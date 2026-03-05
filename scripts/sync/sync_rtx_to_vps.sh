#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# sync_rtx_to_vps.sh — RTX 5090 → VPS (bi-iq.com)
# يُشغّل على RTX 5090 — يرفع training data + LoRA models للـ VPS
# ═══════════════════════════════════════════════════════════════
#
# Usage: ./sync_rtx_to_vps.sh [--full|--models-only|--data-only]
# Cron:  0 * * * * /home/bi/bi-ide-v8/scripts/sync/sync_rtx_to_vps.sh >> /var/log/bi-sync.log 2>&1

set -euo pipefail

# ─── Config ───────────────────────────────────────────────────
VPS_HOST="root@76.13.154.123"
VPS_SYNC_DIR="/opt/bi-iq-app/shared_data"

RTX_TRAINING_DIR="/home/bi/training_data"
RTX_MODELS_DIR="/home/bi/training_data/models/finetuned"
RTX_DATA_DIR="/home/bi/training_data/data"
RTX_INGEST_DIR="/home/bi/training_data/ingest"

LOG_FILE="/var/log/bi-sync.log"
LOCK_FILE="/tmp/bi-sync-rtx.lock"

# ─── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ─── Lock (prevent parallel runs) ────────────────────────────
if [ -f "$LOCK_FILE" ]; then
    log "${YELLOW}⚠️ سنكرونايز قيد التشغيل بالفعل${NC}"
    exit 0
fi
trap "rm -f $LOCK_FILE" EXIT
touch "$LOCK_FILE"

# ─── Parse args ───────────────────────────────────────────────
MODE="${1:---full}"

log "${GREEN}🔄 بدء النسخ المتماثل RTX → VPS ($MODE)${NC}"

# ─── Ensure VPS directory exists ──────────────────────────────
ssh -o ConnectTimeout=10 "$VPS_HOST" "mkdir -p $VPS_SYNC_DIR/{models,data,ingest,metadata}" || {
    log "${RED}❌ فشل الاتصال بالـ VPS${NC}"
    exit 1
}

# ─── Sync Models (LoRA finetuned) ─────────────────────────────
if [[ "$MODE" == "--full" || "$MODE" == "--models-only" ]]; then
    if [ -d "$RTX_MODELS_DIR" ]; then
        log "📦 نقل الموديلات المتعلّمة..."
        rsync -avz --progress --delete \
            "$RTX_MODELS_DIR/" \
            "$VPS_HOST:$VPS_SYNC_DIR/models/"
        log "${GREEN}✅ الموديلات تم نقلها${NC}"
    else
        log "${YELLOW}⚠️ مجلد الموديلات غير موجود: $RTX_MODELS_DIR${NC}"
    fi
fi

# ─── Sync Training Data ──────────────────────────────────────
if [[ "$MODE" == "--full" || "$MODE" == "--data-only" ]]; then
    if [ -d "$RTX_DATA_DIR" ]; then
        log "📊 نقل بيانات التدريب..."
        rsync -avz --progress \
            --exclude='*.tmp' \
            --exclude='__pycache__' \
            "$RTX_DATA_DIR/" \
            "$VPS_HOST:$VPS_SYNC_DIR/data/"
        log "${GREEN}✅ بيانات التدريب تم نقلها${NC}"
    fi
    
    # Sync ingest queue
    if [ -d "$RTX_INGEST_DIR" ]; then
        log "📥 نقل بيانات الاستيعاب..."
        rsync -avz --progress \
            "$RTX_INGEST_DIR/" \
            "$VPS_HOST:$VPS_SYNC_DIR/ingest/"
    fi
fi

# ─── Sync Metadata ───────────────────────────────────────────
# Send machine state info
METADATA=$(cat <<EOF
{
    "source": "rtx5090",
    "sync_time": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "models_count": $(find "$RTX_MODELS_DIR" -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' '),
    "data_size_mb": $(du -sm "$RTX_DATA_DIR" 2>/dev/null | cut -f1 || echo 0),
    "gpu_name": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
}
EOF
)

echo "$METADATA" | ssh "$VPS_HOST" "cat > $VPS_SYNC_DIR/metadata/rtx5090_last_sync.json"

log "${GREEN}🎉 النسخ المتماثل اكتمل — RTX → VPS${NC}"
