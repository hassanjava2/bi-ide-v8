#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-}"
TOKEN="${2:-}"
AGENT_NAME="${3:-$(hostname)}"
LABELS="${4:-}"

if [[ -z "$SERVER_URL" ]]; then
  echo "Usage: ./install_linux.sh <SERVER_URL> [TOKEN] [AGENT_NAME] [LABELS]"
  exit 1
fi

AGENT_DIR="$HOME/.bi-ide-agent"
mkdir -p "$AGENT_DIR"

curl -fsSL "$SERVER_URL/api/v1/orchestrator/download/agent.py" -o "$AGENT_DIR/remote_worker_agent.py"

python3 -m pip install --user --upgrade requests >/dev/null 2>&1 || true

TOKEN_ARG=""
if [[ -n "$TOKEN" ]]; then
  TOKEN_ARG=" --token '$TOKEN'"
fi
LABELS_ARG=""
if [[ -n "$LABELS" ]]; then
  LABELS_ARG=" --labels '$LABELS'"
fi

cat > "$AGENT_DIR/run_agent.sh" <<EOF
#!/usr/bin/env bash
cd "$AGENT_DIR"
python3 "$AGENT_DIR/remote_worker_agent.py" --server "$SERVER_URL" --name "$AGENT_NAME"$LABELS_ARG$TOKEN_ARG
EOF
chmod +x "$AGENT_DIR/run_agent.sh"

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/bi-ide-agent.service" <<EOF
[Unit]
Description=BI IDE Remote Worker Agent
After=network-online.target

[Service]
Type=simple
ExecStart=$AGENT_DIR/run_agent.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now bi-ide-agent.service

echo "Agent installed and started with systemd user service: bi-ide-agent.service"
