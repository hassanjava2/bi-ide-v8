#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BI-IDE v8 - Remaining Deployment Steps
# Run this on the VPS via ssheasy.com terminal
# Database, Redis, .env are already configured!
# ═══════════════════════════════════════════════════════════════

set -e
echo "=== STEP 1: Clone Repository ==="
rm -rf /opt/bi-ide
git clone https://github.com/hassanjava2/bi-ide-v8.git /opt/bi-ide
echo "CLONE OK"

echo "=== STEP 2: Setup Python ==="
cd /opt/bi-ide
python3.11 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools -q
.venv/bin/pip install -r requirements-prod.txt -q
echo "PIP OK"

echo "=== STEP 3: Setup Directories ==="
mkdir -p data learning_data logs static
id -u biide &>/dev/null || useradd -r -s /bin/bash -d /opt/bi-ide biide
chown -R biide:biide /opt/bi-ide

echo "=== STEP 4: Setup Database ==="
systemctl start postgresql
systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER biide WITH PASSWORD 'Bide2026Prod@Xk9m';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE bi_ide OWNER biide;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bi_ide TO biide;" 2>/dev/null || true
echo "DB OK"

echo "=== STEP 5: Setup Redis ==="
REDIS_PASS="RedisB1ide2026Px"
sed -i "s/^# requirepass .*/requirepass $REDIS_PASS/" /etc/redis/redis.conf
sed -i "s/^requirepass .*/requirepass $REDIS_PASS/" /etc/redis/redis.conf
grep -q "^requirepass" /etc/redis/redis.conf || echo "requirepass $REDIS_PASS" >> /etc/redis/redis.conf
systemctl restart redis-server
echo "REDIS OK"

echo "=== STEP 6: Create .env ==="
cat > /opt/bi-ide/.env << 'ENVFILE'
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
WORKERS=2
DATABASE_URL=postgresql+asyncpg://biide:Bide2026Prod@Xk9m@localhost:5432/bi_ide
REDIS_URL=redis://:RedisB1ide2026Px@localhost:6379/0
REDIS_PASSWORD=RedisB1ide2026Px
CACHE_ENABLED=true
CACHE_TTL=3600
SECRET_KEY=a7f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
ADMIN_PASSWORD=BiAdmin2026@Secure
ORCHESTRATOR_TOKEN=c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3
ACCESS_TOKEN_EXPIRE_MINUTES=1440
CORS_ORIGINS=https://app.bi-iq.com,https://bi-iq.com,http://localhost:5173
RTX4090_HOST=
RTX4090_PORT=8080
AI_CORE_HOST=
AUTO_SYNC_CHECKPOINTS=0
ENVFILE
chmod 600 /opt/bi-ide/.env
chown biide:biide /opt/bi-ide/.env
echo "ENV OK"

echo "=== STEP 7: Build UI ==="
cd /opt/bi-ide/ui
npm install --legacy-peer-deps 2>&1 | tail -5
npm run build 2>&1 | tail -5
echo "UI BUILD OK"

echo "=== STEP 8: Database Migrations ==="
cd /opt/bi-ide
sudo -u biide .venv/bin/python -c "
import asyncio
from api.database import engine, Base
async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('TABLES CREATED')
asyncio.run(init())
" 2>&1 || echo "Migration skipped - will init on first run"

echo "=== STEP 9: Create Systemd Services ==="
cat > /etc/systemd/system/bi-ide-api.service << 'EOF'
[Unit]
Description=BI-IDE v8 API
After=network.target postgresql.service redis-server.service
Wants=postgresql.service redis-server.service

[Service]
Type=exec
User=biide
Group=biide
WorkingDirectory=/opt/bi-ide
Environment=PATH=/opt/bi-ide/.venv/bin:/usr/bin:/bin
EnvironmentFile=/opt/bi-ide/.env
ExecStart=/opt/bi-ide/.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/bi-ide-worker.service << 'EOF'
[Unit]
Description=BI-IDE v8 Celery Worker
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=exec
User=biide
Group=biide
WorkingDirectory=/opt/bi-ide
Environment=PATH=/opt/bi-ide/.venv/bin:/usr/bin:/bin
EnvironmentFile=/opt/bi-ide/.env
ExecStart=/opt/bi-ide/.venv/bin/celery -A api.tasks worker --loglevel=info --concurrency=2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable bi-ide-api bi-ide-worker
systemctl start bi-ide-api
systemctl start bi-ide-worker 2>/dev/null || true
echo "SERVICES OK"

echo "=== STEP 10: Configure Nginx ==="
cat > /etc/nginx/sites-available/bi-ide << 'NGINX'
server {
    listen 80;
    server_name app.bi-iq.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
NGINX

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/bi-ide /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
echo "NGINX OK"

echo "=== STEP 11: Firewall ==="
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
echo "FIREWALL OK"

echo "=== STEP 12: SSL Certificate ==="
certbot --nginx -d app.bi-iq.com --non-interactive --agree-tos -m admin@bi-iq.com 2>&1 || echo "SSL SKIPPED - run manually later"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "   DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Admin Username: president"
echo "Admin Password: BiAdmin2026@Secure"
echo "URL: https://app.bi-iq.com"
echo ""
echo "Check status: systemctl status bi-ide-api"
echo "View logs: journalctl -u bi-ide-api -f"
echo ""
