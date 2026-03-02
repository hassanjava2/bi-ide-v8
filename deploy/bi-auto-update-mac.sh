#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BI-IDE Desktop - Mac Auto-Update Script
# يسحب التحديثات من Git, يعيد البناء, ويثبت التطبيق تلقائياً
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="$HOME/Documents/bi-ide-v8"
APP_DIR="$PROJECT_ROOT/apps/desktop-tauri"
LOG_FILE="$PROJECT_ROOT/logs/mac_auto_update.log"
LOCK_FILE="/tmp/bi-ide-update.lock"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Another update is running (PID: $pid). Exiting."
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$PROJECT_ROOT"

# Check for remote changes
git fetch origin main --quiet 2>/dev/null || { log "Git fetch failed"; exit 1; }

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date ($LOCAL)"
    exit 0
fi

log "Update available: $LOCAL → $REMOTE"

# Pull changes
git pull origin main --quiet 2>/dev/null || { log "Git pull failed"; exit 1; }
log "Pulled latest changes"

# Rebuild desktop app
cd "$APP_DIR"
npm install --silent 2>/dev/null
log "Dependencies installed"

npm run tauri build 2>>"$LOG_FILE" || { log "Build failed"; exit 1; }
log "Build successful"

# Kill running instance
pkill -f "BI-IDE Desktop" 2>/dev/null || true
sleep 1

# Install new version
cp -R "$PROJECT_ROOT/target/release/bundle/macos/BI-IDE Desktop.app" /Applications/
log "Installed to /Applications"

# Relaunch
open "/Applications/BI-IDE Desktop.app" &
log "Relaunched BI-IDE Desktop"

# Notification
osascript -e 'display notification "BI-IDE Desktop updated to latest version" with title "BI-IDE Update" sound name "Glass"' 2>/dev/null || true

log "Update complete: $(git rev-parse --short HEAD)"
