#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# BI-IDE v8 - Hostinger VPS Deployment Script
# Domain: app.bi-iq.com
# ═══════════════════════════════════════════════════════════════

DOMAIN="app.bi-iq.com"
APP_DIR="/opt/bi-ide"
APP_NAME="bi-ide"
SYS_USER="biide"
ADMIN_EMAIL="admin@bi-iq.com"
GITHUB_REPO="https://github.com/hassanjava2/bi-ide-v8.git"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "   BI-IDE v8 — Deployment to $DOMAIN"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── 1. System Update & Dependencies ────────────────────────────
echo "═══ [1/9] System Setup ═══"

apt-get update -qq
apt-get install -y -qq \
    curl wget git vim htop net-tools \
    software-properties-common apt-transport-https \
    ca-certificates gnupg lsb-release \
    ufw fail2ban openssl bc

# Python 3.11
log_info "Installing Python 3.11..."
if ! command -v python3.11 &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip || {
        apt-get install -y -qq python3 python3-venv python3-dev python3-pip
    }
fi
PYTHON_CMD=$(command -v python3.11 || command -v python3)
log_success "Python: $($PYTHON_CMD --version)"

# PostgreSQL 15
log_info "Installing PostgreSQL..."
if ! command -v psql &>/dev/null; then
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/postgresql.list 2>/dev/null || true
    apt-get update -qq
    apt-get install -y -qq postgresql postgresql-contrib || apt-get install -y -qq postgresql-15 postgresql-contrib-15 || true
fi
log_success "PostgreSQL installed"

# Redis
log_info "Installing Redis..."
apt-get install -y -qq redis-server || true
log_success "Redis installed"

# Nginx
log_info "Installing Nginx..."
apt-get install -y -qq nginx || true
log_success "Nginx installed"

# Certbot
log_info "Installing Certbot..."
apt-get install -y -qq certbot python3-certbot-nginx || true
log_success "Certbot installed"

# Node.js 20 (for UI build)
log_info "Installing Node.js..."
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>/dev/null || true
    apt-get install -y -qq nodejs || true
fi
log_success "Node.js: $(node --version 2>/dev/null || echo 'not found')"

log_success "System setup completed"

# ─── 2. Firewall ────────────────────────────────────────────────
echo ""
echo "═══ [2/9] Firewall Setup ═══"

ufw default deny incoming 2>/dev/null || true
ufw default allow outgoing 2>/dev/null || true
ufw allow 22/tcp 2>/dev/null || true
ufw allow 80/tcp 2>/dev/null || true
ufw allow 443/tcp 2>/dev/null || true
echo "y" | ufw enable 2>/dev/null || true

log_success "Firewall configured"

# ─── 3. Database Setup ──────────────────────────────────────────
echo ""
echo "═══ [3/9] Database Setup ═══"

DB_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)

systemctl start postgresql 2>/dev/null || true
systemctl enable postgresql 2>/dev/null || true
sleep 2

# Create PostgreSQL database and user
sudo -u postgres psql -c "DROP DATABASE IF EXISTS bi_ide;" 2>/dev/null || true
sudo -u postgres psql -c "DROP USER IF EXISTS biide;" 2>/dev/null || true
sudo -u postgres psql -c "CREATE USER biide WITH PASSWORD '${DB_PASSWORD}';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE bi_ide OWNER biide;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bi_ide TO biide;" 2>/dev/null || true

# Configure Redis
systemctl start redis-server 2>/dev/null || true
systemctl enable redis-server 2>/dev/null || true

if [[ -f /etc/redis/redis.conf ]]; then
    sed -i "s/^# requirepass .*/requirepass ${REDIS_PASSWORD}/" /etc/redis/redis.conf
    sed -i "s/^requirepass .*/requirepass ${REDIS_PASSWORD}/" /etc/redis/redis.conf
    if ! grep -q "^requirepass" /etc/redis/redis.conf; then
        echo "requirepass ${REDIS_PASSWORD}" >> /etc/redis/redis.conf
    fi
    systemctl restart redis-server
fi

log_success "Database setup completed"

# ─── 4. Application Setup ───────────────────────────────────────
echo ""
echo "═══ [4/9] Application Setup ═══"

# Create system user
if ! id "$SYS_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d "$APP_DIR" -m "$SYS_USER"
fi

# Clone repository
if [[ -d "$APP_DIR/.git" ]]; then
    log_info "Pulling latest code..."
    cd "$APP_DIR"
    git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
else
    log_info "Cloning repository..."
    rm -rf "$APP_DIR"
    git clone "$GITHUB_REPO" "$APP_DIR"
fi

cd "$APP_DIR"
chown -R "$SYS_USER:$SYS_USER" "$APP_DIR"

# Create virtual environment
log_info "Creating Python virtual environment..."
if [[ ! -d "$APP_DIR/.venv" ]]; then
    sudo -u "$SYS_USER" $PYTHON_CMD -m venv "$APP_DIR/.venv"
fi

# Install dependencies (lightweight production version)
log_info "Installing Python dependencies..."
sudo -u "$SYS_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip wheel setuptools 2>/dev/null

if [[ -f "$APP_DIR/requirements-prod.txt" ]]; then
    sudo -u "$SYS_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements-prod.txt" --quiet 2>&1 | tail -5
else
    log_warn "requirements-prod.txt not found, using requirements.txt"
    sudo -u "$SYS_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" --quiet 2>&1 | tail -5
fi

# Create necessary directories
mkdir -p "$APP_DIR/data" "$APP_DIR/learning_data" "$APP_DIR/logs" "$APP_DIR/static"
chown -R "$SYS_USER:$SYS_USER" "$APP_DIR"

log_success "Application setup completed"

# ─── 5. Build UI ────────────────────────────────────────────────
echo ""
echo "═══ [5/9] Build UI ═══"

if [[ -d "$APP_DIR/ui" ]]; then
    cd "$APP_DIR/ui"
    log_info "Installing Node.js dependencies..."
    npm install --legacy-peer-deps 2>&1 | tail -3
    log_info "Building UI..."
    npm run build 2>&1 | tail -5 || {
        log_warn "TypeScript build failed, trying vite build only..."
        npx vite build 2>&1 | tail -5 || true
    }
    
    if [[ -d "$APP_DIR/ui/dist" ]]; then
        # Copy dist to nginx serving location
        mkdir -p /var/www/bi-ide/ui
        cp -r "$APP_DIR/ui/dist" /var/www/bi-ide/ui/
        log_success "UI built and deployed"
    else
        log_warn "UI dist directory not found after build"
    fi
else
    log_warn "No UI directory found"
fi

cd "$APP_DIR"

# ─── 6. Environment Configuration ───────────────────────────────
echo ""
echo "═══ [6/9] Environment Configuration ═══"

SECRET_KEY=$(openssl rand -hex 64)
ADMIN_PASSWORD="BiAdmin2026@Secure"
ORCHESTRATOR_TOKEN=$(openssl rand -hex 32)

cat > "$APP_DIR/.env" << ENVEOF
# BI-IDE v8 - Production Environment
# Generated: $(date)
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=2

# Database
DATABASE_URL=postgresql+asyncpg://biide:${DB_PASSWORD}@localhost:5432/bi_ide

# Redis
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}
CACHE_ENABLED=true
CACHE_TTL=3600

# Security
SECRET_KEY=${SECRET_KEY}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
ORCHESTRATOR_TOKEN=${ORCHESTRATOR_TOKEN}
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS
CORS_ORIGINS=https://${DOMAIN},https://bi-iq.com,http://localhost:5173

# AI (remote — not on this VPS)
RTX4090_HOST=
RTX4090_PORT=8080
AI_CORE_HOST=
AUTO_SYNC_CHECKPOINTS=0
ENVEOF

chmod 600 "$APP_DIR/.env"
chown "$SYS_USER:$SYS_USER" "$APP_DIR/.env"

log_success "Environment configured"

# ─── 7. Database Migrations ─────────────────────────────────────
echo ""
echo "═══ [7/9] Database Migrations ═══"

cd "$APP_DIR"

# Configure PostgreSQL auth
PG_HBA=$(find /etc/postgresql -name pg_hba.conf 2>/dev/null | head -1)
if [[ -n "$PG_HBA" ]]; then
    sed -i 's/scram-sha-256/md5/g' "$PG_HBA" 2>/dev/null || true
    sed -i 's/peer/md5/g' "$PG_HBA" 2>/dev/null || true
    systemctl restart postgresql
    sleep 2
fi

if [[ -f "$APP_DIR/alembic.ini" ]]; then
    log_info "Running database migrations..."
    cd "$APP_DIR"
    sudo -u "$SYS_USER" bash -c "cd $APP_DIR && source .env && export DATABASE_URL && .venv/bin/alembic upgrade head" 2>&1 | tail -5 || {
        log_warn "Alembic migration failed — trying with init.sql..."
        if [[ -f "$APP_DIR/init.sql" ]]; then
            PGPASSWORD="$DB_PASSWORD" psql -U biide -d bi_ide -h localhost -f "$APP_DIR/init.sql" 2>&1 | tail -5 || true
        fi
    }
fi

# Create default admin user
if [[ -f "$APP_DIR/scripts/create_default_admin.py" ]]; then
    log_info "Creating default admin user..."
    sudo -u "$SYS_USER" bash -c "
        export ADMIN_PASSWORD='${ADMIN_PASSWORD}'
        export DATABASE_URL='postgresql+asyncpg://biide:${DB_PASSWORD}@localhost:5432/bi_ide'
        cd $APP_DIR && .venv/bin/python scripts/create_default_admin.py
    " 2>&1 | tail -5 || log_warn "Admin creation may need manual setup"
fi

log_success "Database initialized"

# ─── 8. Systemd Services & Nginx ────────────────────────────────
echo ""
echo "═══ [8/9] Services & Nginx ═══"

# API Service
cat > /etc/systemd/system/bi-ide-api.service << 'SVCEOF'
[Unit]
Description=BI-IDE API Server
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=biide
Group=biide
WorkingDirectory=/opt/bi-ide
EnvironmentFile=/opt/bi-ide/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=/opt/bi-ide/.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=30

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/bi-ide/data /opt/bi-ide/learning_data /opt/bi-ide/logs

[Install]
WantedBy=multi-user.target
SVCEOF

# Worker Service
cat > /etc/systemd/system/bi-ide-worker.service << 'SVCEOF'
[Unit]
Description=BI-IDE Background Worker
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=biide
Group=biide
WorkingDirectory=/opt/bi-ide
EnvironmentFile=/opt/bi-ide/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=/opt/bi-ide/.venv/bin/celery -A core.celery_config worker --loglevel=info --concurrency=2
Restart=always
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/bi-ide/data /opt/bi-ide/learning_data /opt/bi-ide/logs

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable bi-ide-api
systemctl enable bi-ide-worker

# Nginx Configuration
cat > /etc/nginx/sites-available/bi-ide << NGINXEOF
limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=auth_limit:10m rate=5r/m;

upstream bi_ide_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name ${DOMAIN};

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    access_log /var/log/nginx/bi-ide-access.log;
    error_log /var/log/nginx/bi-ide-error.log;

    client_max_body_size 100m;

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss font/truetype font/opentype image/svg+xml;

    # Static UI files
    location / {
        root /var/www/bi-ide/ui/dist;
        try_files \$uri \$uri/ /index.html;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /assets/ {
        root /var/www/bi-ide/ui/dist;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # Health check
    location /health {
        proxy_pass http://bi_ide_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /ready {
        proxy_pass http://bi_ide_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Auth (stricter rate limit)
    location /api/v1/auth/ {
        limit_req zone=auth_limit burst=10 nodelay;
        proxy_pass http://bi_ide_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://bi_ide_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://bi_ide_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Block hidden files
    location ~ /\\. { deny all; return 404; }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/bi-ide /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t && systemctl reload nginx
log_success "Nginx configured"

# ─── 9. Start Services & SSL ────────────────────────────────────
echo ""
echo "═══ [9/9] Start Services & SSL ═══"

# Start API
systemctl restart bi-ide-api
sleep 3

# SSL Certificate
log_info "Obtaining SSL certificate..."
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$ADMIN_EMAIL" --redirect --hsts 2>&1 | tail -5 || {
    log_warn "Certbot failed — will work on HTTP for now"
    log_info "Run manually: certbot --nginx -d $DOMAIN"
}

# Auto-renew cron
if ! crontab -l 2>/dev/null | grep -q "certbot renew"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * /usr/bin/certbot renew --quiet") | crontab -
fi

# Start worker
systemctl restart bi-ide-worker 2>/dev/null || true

# Health check
sleep 5
log_info "Running health checks..."
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    log_success "Health check PASSED (/health)"
else
    log_warn "Health check failed - checking logs..."
    journalctl -u bi-ide-api -n 20 --no-pager 2>/dev/null || true
fi

# ═══ Summary ═══
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "              DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  🌐 Website:        https://${DOMAIN}"
echo "  🔑 Admin Username: president"
echo "  🔒 Admin Password: ${ADMIN_PASSWORD}"
echo ""
echo "  📋 Commands:"
echo "    View API logs:    journalctl -u bi-ide-api -f"
echo "    Restart API:      systemctl restart bi-ide-api"
echo "    Check status:     systemctl status bi-ide-api"
echo ""
echo "═══════════════════════════════════════════════════════════════"

# Save credentials
cat > "$APP_DIR/.credentials.txt" << CREDEOF
BI-IDE v8 Deployment Credentials
Generated: $(date)
================================
Domain: ${DOMAIN}
Admin Username: president
Admin Password: ${ADMIN_PASSWORD}
DB Password: ${DB_PASSWORD}
Redis Password: ${REDIS_PASSWORD}
================================
CREDEOF
chmod 600 "$APP_DIR/.credentials.txt"
chown "$SYS_USER:$SYS_USER" "$APP_DIR/.credentials.txt"
log_success "Credentials saved to $APP_DIR/.credentials.txt"
