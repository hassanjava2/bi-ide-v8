#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$PWD}"
DOMAIN="${2:-}"
TOKEN="${3:-}"
ADMIN_EMAIL="${4:-admin@${DOMAIN:-example.com}}"
ADMIN_PASS="${5:-}"

if [[ -z "$DOMAIN" ]]; then
  echo "Usage: ./deploy/vps/install.sh <APP_DIR> <DOMAIN> <ORCHESTRATOR_TOKEN> [ADMIN_EMAIL] [ADMIN_PASS]"
  exit 1
fi

if [[ -z "$TOKEN" ]]; then
  echo "ORCHESTRATOR_TOKEN is required"
  exit 1
fi

if [[ ! -d "$APP_DIR" ]]; then
  echo "APP_DIR does not exist: $APP_DIR"
  exit 1
fi

if [[ $EUID -eq 0 ]]; then
  echo "Run this script as a normal user with sudo permissions (not root)."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required"
  exit 1
fi

echo "[1/7] Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx ufw

echo "[2/7] Configuring Python environment..."
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[2.5/7] Generating secure secrets..."
SECRET_KEY_VALUE="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(64))
PY
)"

if [[ -z "$ADMIN_PASS" ]]; then
  ADMIN_PASSWORD_VALUE="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(18))
PY
)"
else
  ADMIN_PASSWORD_VALUE="$ADMIN_PASS"
fi

echo "[3/7] Writing .env..."
cat > "$APP_DIR/.env" <<EOF
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=$SECRET_KEY_VALUE
ADMIN_PASSWORD=$ADMIN_PASSWORD_VALUE
ACCESS_TOKEN_EXPIRE_MINUTES=1440
DATABASE_URL=sqlite+aiosqlite:///./data/bi_ide.db
CORS_ORIGINS=https://$DOMAIN
ORCHESTRATOR_TOKEN=$TOKEN
ORCHESTRATOR_HEARTBEAT_TIMEOUT_SEC=45
EOF

echo "[4/7] Creating systemd service..."
CURRENT_USER="$(whoami)"
sudo tee /etc/systemd/system/bi-ide-api.service >/dev/null <<EOF
[Unit]
Description=BI IDE Unified API
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now bi-ide-api

echo "[5/7] Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/bi-ide-api >/dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    client_max_body_size 100m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/bi-ide-api /etc/nginx/sites-enabled/bi-ide-api
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo "[6/7] Enabling firewall..."
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

echo "[7/7] Issuing HTTPS certificate..."
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$ADMIN_EMAIL" --redirect

echo ""
echo "Deployment complete."
echo "API URL: https://$DOMAIN/health"
echo "Orchestrator URL: https://$DOMAIN/api/v1/orchestrator/health"
echo "Service status: sudo systemctl status bi-ide-api"
echo "Admin username: president"
echo "Generated ADMIN_PASSWORD: $ADMIN_PASSWORD_VALUE"
echo "⚠️ Save this password immediately and rotate it if needed."
