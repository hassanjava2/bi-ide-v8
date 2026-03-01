#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE v8 — سكربت التحديث الذاتي
# يشتغل كل 2 دقيقة عبر systemd timer
# يفحص GitHub ولو فيه تحديث → يسحب ويعيد تشغيل الخدمات
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── الإعدادات ───────────────────────────────────────────────────────────────
REPO_DIR="${BI_REPO_DIR:-/home/bi/bi-ide-v8}"
WORKER_DIR="${BI_WORKER_DIR:-/home/bi/.bi-ide-worker}"
BRANCH="${BI_BRANCH:-main}"
LOG_FILE="${BI_UPDATE_LOG:-/var/log/bi-auto-update.log}"
LOCK_FILE="/tmp/bi-auto-update.lock"
MAX_LOG_SIZE=5242880  # 5MB

# Services to restart (space-separated)
SERVICES="${BI_SERVICES:-bi-ide-worker}"

# ─── وظائف مساعدة ────────────────────────────────────────────────────────────
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(timestamp)] ❌ ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Rotate log if too big
rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(stat -c%s "$LOG_FILE" 2>/dev/null || stat -f%z "$LOG_FILE" 2>/dev/null)" -gt "$MAX_LOG_SIZE" ]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
        log "Log rotated"
    fi
}

# Prevent multiple instances
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        if kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "Another update is running (PID: $LOCK_PID). Skipping."
            exit 0
        fi
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
}

release_lock() {
    rm -f "$LOCK_FILE"
}

trap release_lock EXIT

# ═══════════════════════════════════════════════════════════════════════════════
# Main Update Logic
# ═══════════════════════════════════════════════════════════════════════════════
update() {
    rotate_log
    acquire_lock

    # Check repo exists
    if [ ! -d "$REPO_DIR/.git" ]; then
        log "Repo not found at $REPO_DIR. Cloning..."
        git clone "https://github.com/hassanjava2/bi-ide-v8.git" "$REPO_DIR"
    fi

    cd "$REPO_DIR"

    # Fetch latest from remote
    git fetch origin "$BRANCH" --quiet 2>/dev/null || {
        log_error "git fetch failed (no internet?)"
        return 1
    }

    # Compare local vs remote
    LOCAL_SHA=$(git rev-parse HEAD)
    REMOTE_SHA=$(git rev-parse "origin/$BRANCH")

    if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
        # No update needed — silent (don't spam log)
        return 0
    fi

    # ── Update Available! ─────────────────────────────────────────────────
    COMMITS_BEHIND=$(git rev-list HEAD..origin/$BRANCH --count)
    log "🔄 Update found! $COMMITS_BEHIND commit(s) behind. Updating..."
    log "   Local:  ${LOCAL_SHA:0:8}"
    log "   Remote: ${REMOTE_SHA:0:8}"

    # Save current SHA for rollback
    ROLLBACK_SHA="$LOCAL_SHA"

    # Pull changes
    if ! git pull origin "$BRANCH" --ff-only 2>>"$LOG_FILE"; then
        log_error "git pull failed. Trying hard reset..."
        git reset --hard "origin/$BRANCH"
    fi

    log "✅ Code updated to ${REMOTE_SHA:0:8}"

    # ── Sync worker files ─────────────────────────────────────────────────
    if [ -d "$WORKER_DIR" ]; then
        log "📂 Syncing worker files..."
        for dir in worker hierarchy core api monitoring ai services; do
            if [ -d "$REPO_DIR/$dir" ]; then
                cp -r "$REPO_DIR/$dir" "$WORKER_DIR/" 2>/dev/null || true
            fi
        done
        cp "$REPO_DIR/requirements.txt" "$WORKER_DIR/" 2>/dev/null || true

        # Install Python dependencies
        if [ -f "$WORKER_DIR/venv/bin/activate" ]; then
            log "📦 Installing dependencies..."
            source "$WORKER_DIR/venv/bin/activate"
            pip install -r "$WORKER_DIR/requirements.txt" --quiet 2>&1 | tail -2
            deactivate 2>/dev/null || true
        fi
    fi

    # ── Restart services ──────────────────────────────────────────────────
    for svc in $SERVICES; do
        if systemctl is-enabled "$svc" &>/dev/null; then
            log "🔄 Restarting $svc..."
            if sudo systemctl restart "$svc" 2>/dev/null; then
                sleep 3
                if systemctl is-active "$svc" &>/dev/null; then
                    log "✅ $svc is running"
                else
                    log_error "$svc failed to start! Rolling back..."
                    rollback "$ROLLBACK_SHA"
                    return 1
                fi
            else
                log_error "Failed to restart $svc (sudo needed?)"
            fi
        fi
    done

    log "🎉 Update complete! $(git log -1 --pretty='%s' 2>/dev/null)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Rollback
# ═══════════════════════════════════════════════════════════════════════════════
rollback() {
    local target_sha="${1:-HEAD~1}"
    log "⏪ Rolling back to $target_sha..."

    cd "$REPO_DIR"
    git reset --hard "$target_sha"

    # Re-sync worker
    if [ -d "$WORKER_DIR" ]; then
        for dir in worker hierarchy core api monitoring ai services; do
            [ -d "$REPO_DIR/$dir" ] && cp -r "$REPO_DIR/$dir" "$WORKER_DIR/" 2>/dev/null || true
        done
    fi

    # Restart services
    for svc in $SERVICES; do
        sudo systemctl restart "$svc" 2>/dev/null || true
    done

    log "✅ Rolled back to $target_sha"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
case "${1:-update}" in
    update)     update ;;
    rollback)   rollback "${2:-}" ;;
    status)
        echo "═══ BI-IDE Auto-Update Status ═══"
        cd "$REPO_DIR" 2>/dev/null && {
            echo "Repo:     $REPO_DIR"
            echo "Branch:   $BRANCH"
            echo "Local:    $(git rev-parse --short HEAD)"
            echo "Remote:   $(git rev-parse --short origin/$BRANCH 2>/dev/null || echo 'unknown')"
            echo "Last log: $(tail -1 "$LOG_FILE" 2>/dev/null || echo 'no log')"
            echo ""
            echo "Services:"
            for svc in $SERVICES; do
                STATUS=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
                echo "  $svc: $STATUS"
            done
        } || echo "Repo not found at $REPO_DIR"
        ;;
    --dry-run)
        echo "═══ Dry Run ═══"
        cd "$REPO_DIR" 2>/dev/null && {
            git fetch origin "$BRANCH" --quiet 2>/dev/null
            LOCAL=$(git rev-parse --short HEAD)
            REMOTE=$(git rev-parse --short "origin/$BRANCH" 2>/dev/null)
            if [ "$LOCAL" = "$REMOTE" ]; then
                echo "✅ Already up to date ($LOCAL)"
            else
                BEHIND=$(git rev-list HEAD..origin/$BRANCH --count)
                echo "🔄 Update available: $LOCAL → $REMOTE ($BEHIND commits behind)"
                echo "Changes:"
                git log --oneline HEAD..origin/$BRANCH
            fi
        } || echo "Repo not found at $REPO_DIR"
        ;;
    -h|--help)
        echo "Usage: $0 [update|rollback|status|--dry-run]"
        echo ""
        echo "Environment Variables:"
        echo "  BI_REPO_DIR     Repo path (default: /home/bi/bi-ide-v8)"
        echo "  BI_WORKER_DIR   Worker path (default: /home/bi/.bi-ide-worker)"
        echo "  BI_BRANCH       Branch (default: main)"
        echo "  BI_SERVICES     Services to restart (default: bi-ide-worker)"
        echo "  BI_UPDATE_LOG   Log file (default: /var/log/bi-auto-update.log)"
        ;;
    *)
        echo "Unknown command: $1. Use --help for usage."
        exit 1
        ;;
esac
