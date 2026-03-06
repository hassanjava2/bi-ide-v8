#!/bin/bash
# ═══════════════════════════════════════════════════
# BI-IDE Training Worker — One-Line Installer
# ═══════════════════════════════════════════════════
#
# Usage:
#   curl -s https://bi-iq.com/install-worker.sh | bash
#
# This sets up any machine as a BI-IDE training worker:
#   1. Installs Python venv + dependencies
#   2. Downloads training_worker.py
#   3. Registers with the fleet
#   4. Ready to receive capsules for training
#
# Requirements: Python 3.10+, ~2GB disk, GPU optional
# ═══════════════════════════════════════════════════

set -e

WORKER_DIR="$HOME/.bi-ide-worker"
VENV_DIR="$WORKER_DIR/venv"
SCRIPT_URL="https://bi-iq.com/training_worker.py"
FLEET_REGISTER_URL="https://bi-iq.com/api/v1/fleet/register"

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║   BI-IDE Training Worker Installer v1.0   ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# === Check Python ===
if command -v python3 &>/dev/null; then
    PY=$(command -v python3)
    PY_VER=$($PY --version 2>&1 | awk '{print $2}')
    echo "✅ Python: $PY_VER"
else
    echo "❌ Python3 not found. Install it first:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "   macOS: brew install python3"
    exit 1
fi

# === Check GPU ===
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "✅ GPU: $GPU_NAME (${GPU_VRAM}MB)"
    HAS_GPU=1
else
    echo "⚠️  No GPU detected — will train on CPU (slower but works)"
    HAS_GPU=0
fi

# === Create worker directory ===
echo ""
echo "📁 Setting up worker in $WORKER_DIR..."
mkdir -p "$WORKER_DIR"

# === Create venv ===
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating Python virtual environment..."
    $PY -m venv "$VENV_DIR"
fi

# === Install dependencies ===
echo "📦 Installing dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet \
    torch \
    transformers \
    accelerate \
    datasets \
    pyyaml \
    requests

# === Download worker script ===
echo "📥 Downloading training worker..."
curl -sL "$SCRIPT_URL" -o "$WORKER_DIR/training_worker.py"

# === Detect machine info ===
HOSTNAME=$(hostname)
IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")
VRAM_GB=0
if [ "$HAS_GPU" = "1" ]; then
    VRAM_GB=$((GPU_VRAM / 1024))
fi

# === Create worker config ===
cat > "$WORKER_DIR/worker_info.json" << EOF
{
    "name": "$HOSTNAME",
    "host": "$IP",
    "has_gpu": $( [ "$HAS_GPU" = "1" ] && echo "true" || echo "false" ),
    "gpu_name": "${GPU_NAME:-none}",
    "vram_gb": $VRAM_GB,
    "venv": "$VENV_DIR/bin/python3",
    "workspace": "$WORKER_DIR",
    "installed": "$(date -Iseconds)"
}
EOF

# === Register with fleet (optional) ===
echo ""
echo "📡 Registering with fleet..."
curl -s -X POST "$FLEET_REGISTER_URL" \
    -H "Content-Type: application/json" \
    -d @"$WORKER_DIR/worker_info.json" 2>/dev/null && echo "✅ Registered" || echo "⚠️  Fleet registration skipped (offline mode)"

# === Create systemd service (Linux only) ===
if [ -d /etc/systemd/system ] && [ "$(id -u)" = "0" ]; then
    cat > /etc/systemd/system/bi-training-worker.service << UNIT
[Unit]
Description=BI-IDE Training Worker
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$WORKER_DIR
ExecStart=$VENV_DIR/bin/python3 $WORKER_DIR/training_worker.py --listen
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    echo "✅ Systemd service created (bi-training-worker)"
fi

# === Done ===
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║           ✅ Installation Complete!        ║"
echo "╠═══════════════════════════════════════════╣"
echo "║  Worker Dir: $WORKER_DIR"
echo "║  Python:     $VENV_DIR/bin/python3"
echo "║  GPU:        ${GPU_NAME:-CPU only}"
if [ "$HAS_GPU" = "1" ]; then
echo "║  VRAM:       ${VRAM_GB}GB"
fi
echo "╚═══════════════════════════════════════════╝"
echo ""
echo "This machine is now a BI-IDE training worker!"
echo "The dispatcher (RTX 5090) will send capsules here automatically."
echo ""
