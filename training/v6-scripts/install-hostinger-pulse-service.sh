#!/usr/bin/env bash
set -euo pipefail

# Bi IDE - Install Hostinger Pulse Training Service
# ينسخ ملف الخدمة إلى systemd ويفعّله.

SERVICE_NAME="hostinger-pulse-train.service"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "$(id -u)" != "0" ]]; then
  echo "❌ شغّله كروت: sudo $0"
  exit 1
fi

if [[ ! -f "$SRC_DIR/$SERVICE_NAME" ]]; then
  echo "❌ ملف الخدمة غير موجود: $SRC_DIR/$SERVICE_NAME"
  exit 1
fi

echo "📦 Installing systemd unit..."
cp -f "$SRC_DIR/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"

echo "🔄 Reloading systemd..."
systemctl daemon-reload

echo "✅ Enabling + starting..."
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "📊 Status:"
systemctl --no-pager status "$SERVICE_NAME" || true

echo "🧾 Logs:"
echo "  journalctl -u $SERVICE_NAME -f"
