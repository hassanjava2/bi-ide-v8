#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Git post-push hook → Auto Deploy
# ينشر تلقائياً بعد كل git push
# ═══════════════════════════════════════════════════════════════════════════════
# التثبيت:
#   cp deploy/post-push.sh .git/hooks/post-push && chmod +x .git/hooks/post-push
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If running from .git/hooks, go up to repo root
if [[ "$SCRIPT_DIR" == *".git/hooks" ]]; then
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
else
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

echo "🚀 Auto-deploying after push..."

# Deploy to 5090 (background)
"$PROJECT_ROOT/deploy/auto_deploy.sh" --5090 &

# Deploy to VPS (background)
"$PROJECT_ROOT/deploy/auto_deploy.sh" --vps &

wait
echo "✅ Auto-deploy complete"
