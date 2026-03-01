#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE v8 — تثبيت التحديث الذاتي
# شغّل مرة وحدة على كل جهاز Linux (RTX 5090, VPS)
# ═══════════════════════════════════════════════════════════════════════════════
# الاستخدام:
#   على الـ 5090:  bash deploy/setup-auto-update.sh
#   على الـ VPS:   bash deploy/setup-auto-update.sh --vps
# ═══════════════════════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[⚠]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ─── Detect machine type ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$1" = "--vps" ]; then
    MACHINE="vps"
    REPO_PATH="/root/bi-ide-v8"
    WORKER_PATH=""
    SERVICES="bi-ide-api"
    USER_NAME="root"
else
    MACHINE="rtx5090"
    REPO_PATH="/home/bi/bi-ide-v8"
    WORKER_PATH="/home/bi/.bi-ide-worker"
    SERVICES="bi-ide-worker"
    USER_NAME="bi"
fi

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   🔄 BI-IDE Auto-Update Setup               ║"
echo "║   Machine: $MACHINE                         ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── 1. Make scripts executable ──────────────────────────────────────────────
log "Making scripts executable..."
chmod +x "$REPO_DIR/deploy/bi-auto-update.sh"

# ─── 2. Create log file ──────────────────────────────────────────────────────
log "Setting up log file..."
sudo touch /var/log/bi-auto-update.log
sudo chown "$USER_NAME:$USER_NAME" /var/log/bi-auto-update.log

# ─── 3. Setup sudoers for passwordless service restart ────────────────────────
log "Setting up passwordless sudo for service restart..."
SUDOERS_FILE="/etc/sudoers.d/bi-auto-update"
sudo tee "$SUDOERS_FILE" > /dev/null << EOF
# BI-IDE Auto-Update: allow $USER_NAME to restart services without password
$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart $SERVICES
$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop $SERVICES
$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl start $SERVICES
$USER_NAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl status $SERVICES
EOF
sudo chmod 440 "$SUDOERS_FILE"
log "Sudoers configured for $SERVICES"

# ─── 4. Install systemd service + timer ───────────────────────────────────────
log "Installing systemd service and timer..."

# Create service file with correct paths
sudo tee /etc/systemd/system/bi-auto-update.service > /dev/null << EOF
[Unit]
Description=BI-IDE Auto-Update Service
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
ExecStart=$REPO_PATH/deploy/bi-auto-update.sh update
WorkingDirectory=$REPO_PATH
StandardOutput=journal
StandardError=journal
Environment=BI_REPO_DIR=$REPO_PATH
Environment=BI_WORKER_DIR=$WORKER_PATH
Environment=BI_BRANCH=main
Environment=BI_SERVICES=$SERVICES
Environment=BI_UPDATE_LOG=/var/log/bi-auto-update.log
TimeoutStartSec=120
Nice=10
EOF

sudo cp "$REPO_DIR/deploy/systemd/bi-auto-update.timer" /etc/systemd/system/

# ─── 5. Enable and start timer ───────────────────────────────────────────────
log "Enabling and starting timer..."
sudo systemctl daemon-reload
sudo systemctl enable bi-auto-update.timer
sudo systemctl start bi-auto-update.timer

# ─── 6. Run first update ─────────────────────────────────────────────────────
log "Running first update check..."
bash "$REPO_DIR/deploy/bi-auto-update.sh" --dry-run

# ─── 7. Show status ──────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}═══ Setup Complete ═══${NC}"
echo ""
log "Timer status:"
systemctl status bi-auto-update.timer --no-pager 2>/dev/null | head -5 || true
echo ""
log "Next trigger:"
systemctl list-timers bi-auto-update.timer --no-pager 2>/dev/null | head -3 || true
echo ""
echo -e "${GREEN}${BOLD}✅ Auto-update is now active!${NC}"
echo -e "   The system will check for updates every 2 minutes."
echo -e "   Log: /var/log/bi-auto-update.log"
echo -e "   Status: systemctl status bi-auto-update.timer"
echo -e "   Logs:   journalctl -u bi-auto-update.service -f"
