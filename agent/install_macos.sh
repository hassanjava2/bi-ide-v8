#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-}"
TOKEN="${2:-}"
AGENT_NAME="${3:-$(scutil --get ComputerName 2>/dev/null || hostname)}"
LABELS="${4:-}"

if [[ -z "$SERVER_URL" ]]; then
  echo "Usage: ./install_macos.sh <SERVER_URL> [TOKEN] [AGENT_NAME] [LABELS]"
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

PLIST="$HOME/Library/LaunchAgents/com.biide.workeragent.plist"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.biide.workeragent</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$AGENT_DIR/run_agent.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$AGENT_DIR/agent.out.log</string>
  <key>StandardErrorPath</key>
  <string>$AGENT_DIR/agent.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Agent installed and started with launchd: com.biide.workeragent"
