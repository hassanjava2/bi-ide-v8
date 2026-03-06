#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BI-IDE v8 — Emergency Data Downloader Deployment 🚨
# ينشر ويفعّل تنزيل بيانات الطوارئ على سيرفر RTX 5090
#
# الاستخدام:
#   ./deploy_emergency_downloader.sh          # تفعيل كامل
#   ./deploy_emergency_downloader.sh --status # حالة التنزيل
#   ./deploy_emergency_downloader.sh --stop   # إيقاف
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BRAIN_DIR="$PROJECT_ROOT/brain"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="/mnt/4tb/emergency"   # 4TB disk
FALLBACK_DIR="$HOME/emergency_data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ═══════════════════════════════════════════════════════════════════
# ألوان
# ═══════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[INFO]${NC}  $(date '+%H:%M:%S') - $1"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $(date '+%H:%M:%S') - $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') - $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1"; }

# ═══════════════════════════════════════════════════════════════════
# اختيار مجلد البيانات
# ═══════════════════════════════════════════════════════════════════
choose_data_dir() {
    if [ -d "/mnt/4tb" ]; then
        DATA_DIR="/mnt/4tb/emergency"
        ok "4TB disk found → $DATA_DIR"
    elif [ -d "/mnt/data" ]; then
        DATA_DIR="/mnt/data/emergency"
        ok "Data disk found → $DATA_DIR"
    else
        DATA_DIR="$FALLBACK_DIR"
        warn "No external disk → $DATA_DIR"
    fi
    mkdir -p "$DATA_DIR"
}

# ═══════════════════════════════════════════════════════════════════
# فحص المتطلبات
# ═══════════════════════════════════════════════════════════════════
check_requirements() {
    log "Checking requirements..."

    # Python
    if ! command -v python3 &>/dev/null; then
        err "python3 not found!"
        exit 1
    fi
    ok "python3: $(python3 --version)"

    # wget أو curl
    if command -v wget &>/dev/null; then
        ok "wget: available"
    elif command -v curl &>/dev/null; then
        ok "curl: available"
    else
        err "Neither wget nor curl found!"
        exit 1
    fi

    # مساحة القرص
    local avail_gb=$(df "$DATA_DIR" --output=avail -B1G 2>/dev/null | tail -1 | tr -d ' ' || echo "0")
    if [ "$avail_gb" -lt 10 ]; then
        warn "Low disk space: ${avail_gb}GB available"
    else
        ok "Disk space: ${avail_gb}GB available"
    fi

    # إنترنت
    if ping -c 1 -W 3 8.8.8.8 &>/dev/null; then
        ok "Internet: connected"
    else
        warn "Internet: NOT connected — downloads will fail"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# إنشاء systemd service
# ═══════════════════════════════════════════════════════════════════
create_service() {
    log "Creating systemd service..."

    local SERVICE_FILE="/etc/systemd/system/bi-ide-emergency-dl.service"
    local PYTHON_PATH=$(which python3)

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=BI-IDE Emergency Data Downloader 🚨
After=network-online.target
Wants=network-online.target
Documentation=https://github.com/hassanjava2/bi-ide-v8

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONUNBUFFERED=1
Environment=EMERGENCY_DATA_DIR=$DATA_DIR
ExecStart=$PYTHON_PATH $BRAIN_DIR/emergency_downloader.py
Restart=on-failure
RestartSec=60
StandardOutput=append:$LOG_DIR/emergency_dl.log
StandardError=append:$LOG_DIR/emergency_dl_error.log

# حدود الموارد — لا يثقل CPU
CPUQuota=50%
Nice=15

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    ok "Service file created: $SERVICE_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# إنشاء سكربت التشغيل
# ═══════════════════════════════════════════════════════════════════
create_launcher() {
    log "Creating launcher script..."

    local LAUNCHER="$PROJECT_ROOT/scripts/start_emergency_download.sh"
    mkdir -p "$(dirname "$LAUNCHER")"

    cat > "$LAUNCHER" <<'LAUNCHER_EOF'
#!/bin/bash
# BI-IDE Emergency Downloader — Quick Launcher
# الاستخدام: ./start_emergency_download.sh [--bg]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "🚨 BI-IDE Emergency Data Downloader"
echo "════════════════════════════════════"
echo ""

# تحديد مجلد البيانات
if [ -d "/mnt/4tb" ]; then
    export EMERGENCY_DATA_DIR="/mnt/4tb/emergency"
elif [ -d "/mnt/data" ]; then
    export EMERGENCY_DATA_DIR="/mnt/data/emergency"
else
    export EMERGENCY_DATA_DIR="$HOME/emergency_data"
fi

mkdir -p "$EMERGENCY_DATA_DIR"
echo "📂 Data dir: $EMERGENCY_DATA_DIR"
echo "💾 Space: $(df -h "$EMERGENCY_DATA_DIR" --output=avail 2>/dev/null | tail -1 || echo 'unknown')"
echo ""

if [ "$1" = "--bg" ]; then
    echo "▶ Starting in background..."
    nohup python3 "$PROJECT_ROOT/brain/emergency_downloader.py" \
        > "$PROJECT_ROOT/logs/emergency_dl.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: $PROJECT_ROOT/logs/emergency_dl.log"
    echo ""
    echo "  Stop: kill $!"
    echo "  Monitor: tail -f $PROJECT_ROOT/logs/emergency_dl.log"
else
    echo "▶ Starting interactively (Ctrl+C to stop)..."
    echo ""
    python3 "$PROJECT_ROOT/brain/emergency_downloader.py"
fi
LAUNCHER_EOF

    chmod +x "$LAUNCHER"
    ok "Launcher: $LAUNCHER"
}

# ═══════════════════════════════════════════════════════════════════
# تفعيل
# ═══════════════════════════════════════════════════════════════════
activate() {
    log "Activating emergency downloader..."
    mkdir -p "$LOG_DIR"

    sudo systemctl enable bi-ide-emergency-dl
    sudo systemctl start bi-ide-emergency-dl

    ok "Service started!"
    echo ""
    echo -e "  ${CYAN}Status:${NC}  sudo systemctl status bi-ide-emergency-dl"
    echo -e "  ${CYAN}Logs:${NC}    tail -f $LOG_DIR/emergency_dl.log"
    echo -e "  ${CYAN}Stop:${NC}    sudo systemctl stop bi-ide-emergency-dl"
    echo -e "  ${CYAN}Data:${NC}    $DATA_DIR"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# حالة التنزيل
# ═══════════════════════════════════════════════════════════════════
show_status() {
    echo ""
    echo -e "${CYAN}🚨 Emergency Downloader Status${NC}"
    echo "═══════════════════════════════"

    # حالة الخدمة
    if sudo systemctl is-active --quiet bi-ide-emergency-dl 2>/dev/null; then
        ok "Service: RUNNING"
    else
        warn "Service: STOPPED"
    fi

    # مجلد البيانات
    choose_data_dir
    if [ -d "$DATA_DIR" ]; then
        local size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        local files=$(find "$DATA_DIR" -type f 2>/dev/null | wc -l)
        echo -e "  ${CYAN}Data dir:${NC}  $DATA_DIR"
        echo -e "  ${CYAN}Size:${NC}      $size"
        echo -e "  ${CYAN}Files:${NC}     $files"
    fi

    # آخر سطور اللوغ
    if [ -f "$LOG_DIR/emergency_dl.log" ]; then
        echo ""
        echo -e "${CYAN}Last 5 log lines:${NC}"
        tail -5 "$LOG_DIR/emergency_dl.log" 2>/dev/null
    fi
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# إيقاف
# ═══════════════════════════════════════════════════════════════════
stop_service() {
    log "Stopping emergency downloader..."
    sudo systemctl stop bi-ide-emergency-dl 2>/dev/null || true
    sudo systemctl disable bi-ide-emergency-dl 2>/dev/null || true
    ok "Service stopped"
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   BI-IDE v8 — Emergency Data Downloader 🚨${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

case "${1:-deploy}" in
    --status|-s)
        show_status
        ;;
    --stop)
        stop_service
        ;;
    --help|-h)
        echo "Usage: $0 [--status|--stop|--help]"
        echo "  (no args)  Deploy and activate"
        echo "  --status   Show download status"
        echo "  --stop     Stop the service"
        ;;
    *)
        choose_data_dir
        check_requirements
        create_service
        create_launcher
        activate

        echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}   EMERGENCY DOWNLOADER ACTIVATED! 🚨✅${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
        echo ""
        echo "Downloads:"
        echo "  P0: Mistral 7B (~4GB)    — أساسي"
        echo "  P1: Wikipedia AR (~2GB)  — موسوعة"
        echo "  P2: Llama 3 8B (~5GB)    — نموذج كبير"
        echo ""
        ;;
esac
