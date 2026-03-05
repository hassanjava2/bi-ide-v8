#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup_sync.sh — تنصيب نظام النسخ المتماثل
# يُشغّل مرة واحدة على كل جهاز
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$(hostname)" in
    *rtx*|*5090*|*gpu*)
        echo -e "${GREEN}🔧 تنصيب على RTX 5090...${NC}"
        
        # Make scripts executable
        chmod +x "$SCRIPT_DIR/sync_rtx_to_vps.sh"
        
        # Install systemd service
        sudo cp "$SCRIPT_DIR/bi-sync-watcher.service" /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable bi-sync-watcher
        sudo systemctl start bi-sync-watcher
        
        # Add cron job (hourly sync as backup)
        (crontab -l 2>/dev/null; echo "0 * * * * $SCRIPT_DIR/sync_rtx_to_vps.sh >> /var/log/bi-sync.log 2>&1") | sort -u | crontab -
        
        echo -e "${GREEN}✅ RTX sync installed:${NC}"
        echo "  • systemd service: bi-sync-watcher (مراقب)"
        echo "  • cron: كل ساعة (نسخة احتياطية)"
        ;;
    *)
        echo -e "${GREEN}🔧 تنصيب على جهاز محلي ($(uname -s))...${NC}"
        
        # Make scripts executable
        chmod +x "$SCRIPT_DIR/sync_vps_to_local.sh"
        
        echo -e "${GREEN}✅ جاهز! شغّل:${NC}"
        echo "  ./sync_vps_to_local.sh            # سحب كل البيانات"
        echo "  ./sync_vps_to_local.sh --models-only  # سحب الموديلات فقط"
        ;;
esac
