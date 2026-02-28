#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BI-IDE — Deploy to All Machines
# Pushes updates from Mac to: Hostinger, RTX 5090, Windows
# Usage: ./scripts/deploy_all.sh
# ═══════════════════════════════════════════════════════════════

set -o pipefail

# ── Config ──
HOSTINGER_HOST="76.13.154.123"
HOSTINGER_PASS="Bb97-@hhbb353631"
HOSTINGER_APP_DIR="/opt/bi-iq-app"

RTX_USER="bi"
RTX_PASS="353631"
RTX_SSH_PORT="2222"  # Via Hostinger tunnel

# Colors
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[0;34m'; N='\033[0m'
ok() { echo -e "${G}✅ $1${N}"; }
err() { echo -e "${R}❌ $1${N}"; }
info() { echo -e "${B}📡 $1${N}"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=$(date +%Y%m%d_%H%M%S)

echo ""
echo "═══════════════════════════════════════════════════"
echo "  🚀 BI-IDE Deploy All — v${VERSION}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Git commit & push ──
info "Step 1: Git push..."
cd "$PROJECT_DIR"
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    info "No changes to commit"
else
    git add -A
    git commit -m "deploy: auto-update ${VERSION}" --quiet 2>/dev/null || true
fi
git push --quiet 2>/dev/null && ok "Git pushed" || info "Git push skipped (no remote or already up to date)"

# ── Step 2: Deploy to Hostinger ──
echo ""
info "Step 2: Deploying to Hostinger (${HOSTINGER_HOST})..."

expect << EOFX 2>/dev/null
set timeout 30
spawn ssh -o StrictHostKeyChecking=no root@${HOSTINGER_HOST}
expect "password:"
send "${HOSTINGER_PASS}\r"
expect "# "

# Git pull
send "cd ${HOSTINGER_APP_DIR} && git pull origin main --quiet 2>/dev/null || git pull --quiet 2>/dev/null; echo GITDONE\r"
expect -timeout 15 "GITDONE"

# Clear cache & restart API
send "find ${HOSTINGER_APP_DIR} -name '__pycache__' -exec rm -rf {} + 2>/dev/null; pkill -HUP -f uvicorn 2>/dev/null; echo APIDONE\r"
expect -timeout 5 "APIDONE"

# Check API health
send "sleep 2 && curl -s http://localhost:8010/health | head -c 50 && echo HCHECK\r"
expect -timeout 10 "HCHECK"

send "exit\r"
expect eof
EOFX

if [ $? -eq 0 ]; then
    ok "Hostinger updated"
else
    err "Hostinger deploy failed"
fi

# ── Step 3: Deploy to RTX 5090 (via Hostinger) ──
echo ""
info "Step 3: Deploying to RTX 5090..."

# SCP key files to Hostinger first, then relay to RTX
expect << EOFX 2>/dev/null
set timeout 30
spawn ssh -o StrictHostKeyChecking=no root@${HOSTINGER_HOST}
expect "password:"
send "${HOSTINGER_PASS}\r"
expect "# "

# Copy updated files to RTX 5090
send "sshpass -p ${RTX_PASS} scp -o StrictHostKeyChecking=no -P ${RTX_SSH_PORT} ${HOSTINGER_APP_DIR}/rtx4090_machine/rtx4090_server.py ${HOSTINGER_APP_DIR}/rtx4090_machine/resource_manager.py ${RTX_USER}@localhost:/tmp/ && echo SCPOK\r"
expect -timeout 15 "SCPOK"

# Also copy worker
send "sshpass -p ${RTX_PASS} scp -o StrictHostKeyChecking=no -P ${RTX_SSH_PORT} ${HOSTINGER_APP_DIR}/worker/bi_worker.py ${RTX_USER}@localhost:/tmp/ && echo WRKCP\r"
expect -timeout 10 "WRKCP"

# Restart RTX server
send "sshpass -p ${RTX_PASS} ssh -o StrictHostKeyChecking=no -p ${RTX_SSH_PORT} ${RTX_USER}@localhost 'pkill -f rtx4090_server; sleep 2; cd /tmp && nohup python3 rtx4090_server.py > /tmp/rtx_server.log 2>&1 & disown; sleep 4; curl -s http://localhost:8080/health | head -c 50' && echo RTXOK\r"
expect -timeout 20 "RTXOK"

send "exit\r"
expect eof
EOFX

if [ $? -eq 0 ]; then
    ok "RTX 5090 updated"
else
    err "RTX 5090 deploy failed"
fi

# ── Step 4: Deploy to Windows (if reachable) ──
echo ""
info "Step 4: Checking Windows..."

# Windows is on same local network — try direct SSH
# The user previously opened port 22 (OpenSSH) on the Windows machine
WIN_HOST=""
# Try common local IPs for the Windows machine
for ip in 192.168.68.110 192.168.68.111 192.168.68.112 192.168.68.100; do
    if ping -c1 -W1 "$ip" &>/dev/null 2>&1; then
        WIN_HOST="$ip"
        break
    fi
done

if [ -n "$WIN_HOST" ]; then
    info "Windows found at ${WIN_HOST}"
    # SCP worker to Windows
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        "$PROJECT_DIR/worker/bi_worker.py" \
        "bi@${WIN_HOST}:/tmp/bi_worker.py" 2>/dev/null && ok "Worker copied to Windows" || err "Windows SCP failed"
else
    info "Windows not reachable on local network (skipped)"
fi

# ── Step 5: Verify all machines ──
echo ""
echo "═══════════════════════════════════════════════════"
echo "  📊 Verification"
echo "═══════════════════════════════════════════════════"

# Hostinger API
API_VER=$(curl -sf https://bi-iq.com/api/v1/version 2>/dev/null || echo "N/A")
echo -e "  Hostinger API:  ${G}${API_VER:-online}${N}"

# RTX 5090 (via public API)
RTX_VER=$(curl -sf --max-time 5 https://bi-iq.com/api/v1/rtx-status 2>/dev/null | head -c 100 || echo "check via tunnel")
echo -e "  RTX 5090:       ${G}${RTX_VER:-deployed}${N}"

# Local Mac
echo -e "  Mac (local):    ${G}source${N}"

echo ""
echo "═══════════════════════════════════════════════════"
echo -e "  ${G}🎉 Deploy complete — ${VERSION}${N}"
echo "═══════════════════════════════════════════════════"
echo ""
