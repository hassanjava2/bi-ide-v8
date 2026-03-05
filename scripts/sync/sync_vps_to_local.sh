#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# sync_vps_to_local.sh — VPS → Mac/Windows (سحب أحدث البيانات)
# يُشغّل على الجهاز المحلي (Mac أو Windows WSL)
# ═══════════════════════════════════════════════════════════════
#
# Usage: ./sync_vps_to_local.sh [--full|--models-only|--data-only]

set -euo pipefail

# ─── Config ───────────────────────────────────────────────────
VPS_HOST="root@76.13.154.123"
VPS_SYNC_DIR="/opt/bi-iq-app/shared_data"

# Detect OS and set local paths
case "$(uname -s)" in
    Darwin)  # Mac
        LOCAL_SYNC_DIR="$HOME/training_data"
        OS_NAME="macOS"
        ;;
    Linux)  # Linux or WSL
        if grep -qi microsoft /proc/version 2>/dev/null; then
            # WSL — store in Windows-accessible path
            LOCAL_SYNC_DIR="/mnt/c/Users/BI/training_data"
            OS_NAME="Windows (WSL)"
        else
            LOCAL_SYNC_DIR="$HOME/training_data"
            OS_NAME="Linux"
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)  # Git Bash on Windows
        LOCAL_SYNC_DIR="C:/Users/BI/training_data"
        OS_NAME="Windows"
        ;;
    *)
        echo "❌ نظام تشغيل غير مدعوم"
        exit 1
        ;;
esac

# ─── Colors ───────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ─── Parse args ───────────────────────────────────────────────
MODE="${1:---full}"

log "${GREEN}🔄 سحب البيانات من VPS → $OS_NAME ($MODE)${NC}"
log "📂 المجلد المحلي: $LOCAL_SYNC_DIR"

# ─── Ensure local dirs exist ─────────────────────────────────
mkdir -p "$LOCAL_SYNC_DIR"/{models,data,ingest,metadata}

# ─── Check VPS connectivity ──────────────────────────────────
if ! ssh -o ConnectTimeout=10 "$VPS_HOST" "test -d $VPS_SYNC_DIR" 2>/dev/null; then
    log "${RED}❌ VPS غير متاح أو مجلد البيانات غير موجود${NC}"
    log "${YELLOW}💡 شغّل sync_rtx_to_vps.sh على RTX أولاً${NC}"
    exit 1
fi

# ─── Check last sync metadata ────────────────────────────────
log "📋 فحص حالة آخر سنكرونايز..."
ssh "$VPS_HOST" "cat $VPS_SYNC_DIR/metadata/rtx5090_last_sync.json 2>/dev/null || echo '{\"sync_time\": \"never\"}'" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  آخر نسخ من RTX: {d.get(\"sync_time\", \"غير معروف\")}')" 2>/dev/null || true

# ─── Sync Models ─────────────────────────────────────────────
if [[ "$MODE" == "--full" || "$MODE" == "--models-only" ]]; then
    log "📦 سحب الموديلات المتعلّمة..."
    rsync -avz --progress \
        "$VPS_HOST:$VPS_SYNC_DIR/models/" \
        "$LOCAL_SYNC_DIR/models/"
    log "${GREEN}✅ الموديلات محدّثة${NC}"
fi

# ─── Sync Training Data ──────────────────────────────────────
if [[ "$MODE" == "--full" || "$MODE" == "--data-only" ]]; then
    log "📊 سحب بيانات التدريب..."
    rsync -avz --progress \
        --exclude='*.tmp' \
        --exclude='__pycache__' \
        "$VPS_HOST:$VPS_SYNC_DIR/data/" \
        "$LOCAL_SYNC_DIR/data/"
    
    log "📥 سحب بيانات الاستيعاب..."
    rsync -avz --progress \
        "$VPS_HOST:$VPS_SYNC_DIR/ingest/" \
        "$LOCAL_SYNC_DIR/ingest/"
    
    log "${GREEN}✅ البيانات محدّثة${NC}"
fi

# ─── Sync Metadata ───────────────────────────────────────────
rsync -avz "$VPS_HOST:$VPS_SYNC_DIR/metadata/" "$LOCAL_SYNC_DIR/metadata/" 2>/dev/null || true

# ─── Summary ─────────────────────────────────────────────────
MODELS_COUNT=$(find "$LOCAL_SYNC_DIR/models" -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l | tr -d ' ')
DATA_SIZE=$(du -sh "$LOCAL_SYNC_DIR/data" 2>/dev/null | cut -f1 || echo "0")

log ""
log "${GREEN}🎉 النسخ اكتمل — VPS → $OS_NAME${NC}"
log "   📦 الموديلات: $MODELS_COUNT"
log "   📊 بيانات التدريب: $DATA_SIZE"
log "   📂 المسار: $LOCAL_SYNC_DIR"
