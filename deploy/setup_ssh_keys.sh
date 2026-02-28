#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# إعداد SSH Key للنشر التلقائي بدون كلمة مرور
# Setup SSH Keys for passwordless auto-deploy
# ═══════════════════════════════════════════════════════════════════════════════
# شغّل هذا السكربت مرة وحدة على الماك:
#   bash deploy/setup_ssh_keys.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}═══ إعداد SSH Keys ═══${NC}"

KEY="$HOME/.ssh/id_ed25519"

# Generate key if not exists
if [ ! -f "$KEY" ]; then
    echo -e "${YELLOW}[*] إنشاء مفتاح SSH جديد...${NC}"
    ssh-keygen -t ed25519 -f "$KEY" -N "" -C "bi-ide-deploy@$(hostname)"
fi

echo -e "\n${GREEN}═══ المفتاح العام ═══${NC}"
cat "${KEY}.pub"

echo -e "\n${YELLOW}═══ إرسال المفتاح للأجهزة ═══${NC}"

# VPS
echo -e "\n[1/2] VPS (bi-iq.com):"
echo "     يطلب منك كلمة مرور root — آخر مرة تحتاجها!"
ssh-copy-id -i "$KEY" root@bi-iq.com 2>/dev/null && echo "     ✅ VPS ready" || echo "     ⚠️ جرب يدوياً: ssh-copy-id root@bi-iq.com"

# 5090
echo -e "\n[2/2] RTX 5090 (192.168.1.164):"
if ssh -o BatchMode=yes bi@192.168.1.164 "echo ok" &>/dev/null; then
    echo "     ✅ 5090 already has key"
else
    ssh-copy-id -i "$KEY" bi@192.168.1.164 2>/dev/null && echo "     ✅ 5090 ready" || echo "     ⚠️ جرب يدوياً: ssh-copy-id bi@192.168.1.164"
fi

echo -e "\n${GREEN}═══ تفعيل الـ sudo بدون كلمة مرور على 5090 ═══${NC}"
echo "شغّل هذا الأمر على الـ 5090:"
echo "  sudo bash -c 'echo \"bi ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart bi-ide-worker\" >> /etc/sudoers.d/bi-ide'"

echo -e "\n${GREEN}═══ تثبيت Git Hook ═══${NC}"
HOOK_DIR="$(git rev-parse --git-dir)/hooks"
cp deploy/post-push.sh "$HOOK_DIR/post-push" 2>/dev/null && chmod +x "$HOOK_DIR/post-push" && echo "✅ Auto-deploy hook installed" || echo "⚠️ Install manually: cp deploy/post-push.sh .git/hooks/post-push"

echo -e "\n${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Done! الآن كل git push → نشر تلقائي     ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
