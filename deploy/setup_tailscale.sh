#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE — إعداد Tailscale على كل الأجهزة
# شغّل هذا السكربت قبل ما تروح للبيت!
# ═══════════════════════════════════════════════════════════════════════════════
# الاستخدام:
#   bash deploy/setup_tailscale.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
RED='\033[0;31m'; NC='\033[0m'; BOLD='\033[1m'

echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║   🌐 إعداد Tailscale — اتصال من أي مكان     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mac
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}═══ [1/3] Mac ═══${NC}"

if command -v tailscale &>/dev/null; then
    echo -e "${GREEN}  ✅ Tailscale مثبت${NC}"
    tailscale status 2>/dev/null && echo -e "${GREEN}  ✅ متصل${NC}" || {
        echo -e "${YELLOW}  ⚠️ مثبت لكن غير متصل${NC}"
        echo -e "${YELLOW}  شغّل تطبيق Tailscale من Applications ثم سجل دخول${NC}"
    }
else
    echo -e "${YELLOW}  ⚠️ Tailscale غير مثبت على الماك${NC}"
    echo ""
    echo -e "  ${BOLD}طريقة التثبيت:${NC}"
    echo "  1. افتح هذا الرابط: https://apps.apple.com/app/tailscale/id1475387142"
    echo "     أو حمّل من: https://tailscale.com/download/mac"
    echo ""
    echo "  2. ثبّت التطبيق وسجل دخول بحسابك"
    echo ""
    
    # Try to open the download page
    if command -v open &>/dev/null; then
        echo -e "  ${CYAN}أفتح صفحة التحميل؟ (y/n)${NC}"
        read -r answer
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
            open "https://tailscale.com/download/mac"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RTX 5090
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══ [2/3] RTX 5090 (192.168.1.164) ═══${NC}"

RTX_HOST="bi@192.168.1.164"

if ssh -o ConnectTimeout=3 -o BatchMode=yes "$RTX_HOST" "command -v tailscale" &>/dev/null; then
    echo -e "${GREEN}  ✅ Tailscale مثبت على 5090${NC}"
    ssh "$RTX_HOST" "tailscale status 2>/dev/null" && echo -e "${GREEN}  ✅ متصل${NC}" || echo -e "${YELLOW}  ⚠️ شغّله: sudo tailscale up${NC}"
else
    echo -e "${YELLOW}  ⚠️ يحتاج تثبيت — يطلب كلمة مرور sudo${NC}"
    echo "  جاري التثبيت..."
    echo ""
    
    # This will ask for sudo password interactively
    ssh -t "$RTX_HOST" bash -c '"
        echo \"[*] Installing Tailscale...\"
        curl -fsSL https://tailscale.com/install.sh | sudo sh
        sudo systemctl enable tailscaled
        sudo systemctl start tailscaled
        echo \"\"
        echo \"[*] Starting Tailscale — سجل دخول بهذا الرابط:\"
        sudo tailscale up
        echo \"\"
        echo \"✅ Tailscale installed and connected!\"
        tailscale ip -4
    "'
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Windows (instructions only)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══ [3/3] Windows ═══${NC}"
echo -e "  ${YELLOW}⚠️ Windows يحتاج تثبيت يدوي:${NC}"
echo "  1. حمّل من: https://tailscale.com/download/windows"
echo "  2. ثبّته وسجل دخول بنفس الحساب"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}${BOLD}"
echo "═══════════════════════════════════════════════"
echo "   📋 بعد التثبيت على كل الأجهزة:"
echo "═══════════════════════════════════════════════"
echo -e "${NC}"
echo "  شغّل: tailscale status"
echo "  يطلعلك قائمة بكل أجهزتك + IPs الثابتة"
echo ""
echo "  مثال الاتصال من البيت:"
echo "    ssh bi@<tailscale-ip-of-5090>"
echo ""
echo "  بعدها عدّل deploy workflow:"
echo "    .agent/workflows/deploy.md"
echo "  بدّل RTX5090_HOST من 192.168.1.164 إلى Tailscale IP"
echo ""
echo -e "${GREEN}  ✅ هيجي تكدر تتصل من أي مكان بالعالم!${NC}"
