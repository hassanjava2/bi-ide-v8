#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE v8 — سكربت النشر التلقائي الشامل
# Auto-Deploy: Mac → GitHub → VPS + RTX 5090 + Windows
# ═══════════════════════════════════════════════════════════════════════════════
# الاستخدام:
#   ./deploy/auto_deploy.sh                # نشر كل شي
#   ./deploy/auto_deploy.sh --5090         # نشر على 5090 فقط
#   ./deploy/auto_deploy.sh --vps          # نشر على VPS فقط
#   ./deploy/auto_deploy.sh --push-only    # git push فقط
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ─── الألوان ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

# ─── الإعدادات ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── الأجهزة ─────────────────────────────────────────────────────────────────
VPS_HOST="root@bi-iq.com"
VPS_DIR="/root/bi-ide-v8"
VPS_SERVICE="bi-ide-api"

RTX5090_HOST="bi@192.168.1.164"
RTX5090_REPO="/home/bi/bi-ide-v8"
RTX5090_WORKER="/home/bi/.bi-ide-worker"
RTX5090_SERVICE="bi-ide-worker"

GITHUB_REPO="https://github.com/hassanjava2/bi-ide-v8.git"
BRANCH="main"

# ─── وظائف مساعدة ────────────────────────────────────────────────────────────
log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[⚠]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
header(){ echo -e "\n${CYAN}${BOLD}═══ $1 ═══${NC}"; }

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Git Push
# ═══════════════════════════════════════════════════════════════════════════════
push_to_github() {
    header "📤 Git Push → GitHub"
    cd "$PROJECT_ROOT"
    
    # Check for changes
    if [ -z "$(git status --porcelain)" ]; then
        log "لا توجد تغييرات جديدة"
    else
        git add -A
        CHANGES=$(git status --short | wc -l | tr -d ' ')
        
        # Auto-generate commit message
        MSG="deploy: update $(date +%Y-%m-%d\ %H:%M) — ${CHANGES} files"
        git commit -m "$MSG"
        log "Committed: $MSG"
    fi
    
    git push origin "$BRANCH" 2>&1 && log "✅ Pushed to GitHub" || {
        error "فشل Push — جرب: git push origin $BRANCH --force-with-lease"
        return 1
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Deploy to RTX 5090
# ═══════════════════════════════════════════════════════════════════════════════
deploy_5090() {
    header "🖥️ Deploy → RTX 5090 (192.168.1.164)"
    
    # Check connectivity
    if ! ssh -o ConnectTimeout=3 -o BatchMode=yes "$RTX5090_HOST" "echo ok" &>/dev/null; then
        error "RTX 5090 غير متصل!"
        return 1
    fi
    
    ssh "$RTX5090_HOST" bash <<'REMOTE'
        set -e
        REPO="/home/bi/bi-ide-v8"
        WORKER="/home/bi/.bi-ide-worker"
        
        # Clone if not exists
        if [ ! -d "$REPO/.git" ]; then
            echo "[*] Cloning repo..."
            git clone https://github.com/hassanjava2/bi-ide-v8.git "$REPO"
        fi
        
        # Pull latest
        cd "$REPO"
        echo "[*] Pulling latest..."
        git pull origin main
        
        # Sync worker files
        echo "[*] Syncing worker files..."
        cp -r worker/* "$WORKER/" 2>/dev/null || true
        cp requirements.txt "$WORKER/" 2>/dev/null || true
        cp -r hierarchy/ "$WORKER/" 2>/dev/null || true
        cp -r core/ "$WORKER/" 2>/dev/null || true
        cp -r api/ "$WORKER/" 2>/dev/null || true
        cp -r monitoring/ "$WORKER/" 2>/dev/null || true
        cp -r ai/ "$WORKER/" 2>/dev/null || true
        
        # Install deps
        cd "$WORKER"
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            pip install -r requirements.txt --quiet 2>&1 | tail -2
        fi
        
        # Restart worker (needs sudo)
        echo "[*] Restarting worker..."
        sudo systemctl restart bi-ide-worker 2>/dev/null && echo "[✓] Worker restarted" || echo "[⚠] Restart manually: sudo systemctl restart bi-ide-worker"
        
        echo "[✓] 5090 deploy complete"
REMOTE
    
    log "✅ RTX 5090 deployed"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Deploy to VPS (bi-iq.com)
# ═══════════════════════════════════════════════════════════════════════════════
deploy_vps() {
    header "🌐 Deploy → VPS (bi-iq.com)"
    
    ssh "$VPS_HOST" bash <<REMOTE
        set -e
        
        # Clone if not exists
        if [ ! -d "$VPS_DIR/.git" ]; then
            echo "[*] Cloning repo..."
            git clone $GITHUB_REPO "$VPS_DIR"
        fi
        
        # Pull latest
        cd "$VPS_DIR"
        echo "[*] Pulling latest..."
        git pull origin $BRANCH
        
        # Install deps
        echo "[*] Installing dependencies..."
        pip3 install -r requirements.txt --quiet 2>&1 | tail -3
        
        # Restart services
        echo "[*] Restarting API..."
        systemctl restart $VPS_SERVICE 2>/dev/null && echo "[✓] API restarted" || echo "[⚠] Restart manually"
        
        # Update web files
        if [ -d "/var/www/bi-iq.com" ]; then
            cp -r apps/web/dist/* /var/www/bi-iq.com/ 2>/dev/null || true
            echo "[✓] Web files updated"
        fi
        
        echo "[✓] VPS deploy complete"
REMOTE
    
    log "✅ VPS deployed"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Health Check
# ═══════════════════════════════════════════════════════════════════════════════
health_check() {
    header "🏥 Health Check"
    
    # VPS
    echo -n "  VPS API: "
    curl -s -o /dev/null -w "%{http_code}" "https://bi-iq.com/health" 2>/dev/null && echo " ✅" || echo " ❌"
    
    # 5090
    echo -n "  5090: "
    ssh -o ConnectTimeout=3 "$RTX5090_HOST" "systemctl is-active bi-ide-worker 2>/dev/null" && echo " ✅" || echo " ❌"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
main() {
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════╗"
    echo "║       🚀 BI-IDE v8 — Auto Deploy            ║"
    echo "║       $(date '+%Y-%m-%d %H:%M:%S')                    ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    case "${1:-all}" in
        --push-only) push_to_github ;;
        --5090)      push_to_github; deploy_5090 ;;
        --vps)       push_to_github; deploy_vps ;;
        all|*)
            push_to_github
            deploy_5090
            deploy_vps
            health_check
            ;;
    esac
    
    echo ""
    log "🎉 Deploy complete — $(date '+%H:%M:%S')"
}

main "$@"
