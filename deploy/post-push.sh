#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Git post-push hook → Auto-Update Notification
# الأجهزة ستتحدث تلقائياً خلال 2 دقيقة عبر self-update timers
# ═══════════════════════════════════════════════════════════════════════════════
# التثبيت:
#   cp deploy/post-push.sh .git/hooks/post-push && chmod +x .git/hooks/post-push
#   أو:
#   ln -sf ../../deploy/post-push.sh .git/hooks/post-push
# ═══════════════════════════════════════════════════════════════════════════════

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║  🚀 Push successful! Auto-update in progress... ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}All machines will auto-update within 2 minutes:${NC}"
echo -e "  📡 VPS (bi-iq.com)     — systemd timer"
echo -e "  🖥️  RTX 5090            — systemd timer"
echo -e "  💻 Windows (RTX 4050)  — scheduled task"
echo ""
echo -e "Check status: ${BOLD}ssh bi@192.168.1.164 'bash /home/bi/bi-ide-v8/deploy/bi-auto-update.sh status'${NC}"
echo ""
