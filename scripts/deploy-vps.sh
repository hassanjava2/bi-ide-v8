#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# BI-IDE v8 - VPS Deployment Script
# ═══════════════════════════════════════════════════════════════

# Configuration
APP_NAME="bi-ide"
APP_DIR="/opt/${APP_NAME}"
USER="biide"
DOMAIN="${1:-}"
ADMIN_EMAIL="${2:-admin@localhost}"
GITHUB_REPO="${GITHUB_REPO:-}"  # Set this to your repo URL

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

if [[ -z "$DOMAIN" ]]; then
    echo "Usage: $0 <domain> [admin-email]"
    echo "Example: $0 bi-ide.example.com admin@example.com"
    exit 1
fi

if [[ "$EUID" -ne 0 ]]; then
    log_error "This script must be run as root or with sudo"
    exit 1
fi

# Detect Ubuntu version
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS_NAME="$NAME"
    OS_VERSION="$VERSION_ID"
else
    log_error "Cannot detect OS version"
    exit 1
fi

log_info "Deploying BI-IDE to $DOMAIN on $OS_NAME $OS_VERSION..."

# ═══════════════════════════════════════════════════════════════
# 1. System Setup
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[1/9] System Setup"
echo "═══════════════════════════════════════════════════════════════"

# Update system
log_info "Updating package lists..."
apt-get update -qq

# Install essential packages
log_info "Installing system dependencies..."
apt-get install -y -qq \
    curl \
    wget \
    git \
    vim \
    htop \
    net-tools \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    ufw \
    fail2ban \
    openssl \
    bc

# Install Python 3.11 (if available) or default Python3
log_info "Installing Python..."
if add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip
    PYTHON_CMD="python3.11"
else
    apt-get install -y -qq python3 python3-venv python3-dev python3-pip
    PYTHON_CMD="python3"
fi

# Install PostgreSQL
log_info "Installing PostgreSQL..."
if ! command -v psql &> /dev/null; then
    # Add PostgreSQL official repository
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg
    echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/postgresql.list
    apt-get update -qq
    apt-get install -y -qq postgresql-15 postgresql-contrib-15
fi

# Install Redis
log_info "Installing Redis..."
if ! command -v redis-cli &> /dev/null; then
    apt-get install -y -qq redis-server
fi

# Install Nginx
log_info "Installing Nginx..."
if ! command -v nginx &> /dev/null; then
    apt-get install -y -qq nginx
fi

# Install Certbot
log_info "Installing Certbot..."
if ! command -v certbot &> /dev/null; then
    apt-get install -y -qq certbot python3-certbot-nginx
fi

# Create user if not exists
if ! id "$USER" &>/dev/null; then
    log_info "Creating user: $USER"
    useradd -r -s /bin/bash -d "$APP_DIR" -m "$USER"
    usermod -aG sudo "$USER" 2>/dev/null || true
else
    log_warn "User $USER already exists"
fi

log_success "System setup completed"

# ═══════════════════════════════════════════════════════════════
# 2. Firewall Setup
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[2/9] Firewall Setup"
echo "═══════════════════════════════════════════════════════════════"

log_info "Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable

log_success "Firewall configured"

# ═══════════════════════════════════════════════════════════════
# 3. Database Setup
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[3/9] Database Setup"
echo "═══════════════════════════════════════════════════════════════"

# Generate secure password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)

# Start PostgreSQL if not running
systemctl start postgresql
systemctl enable postgresql

# Wait for PostgreSQL to be ready
sleep 2

# Create database and user
log_info "Creating database and user..."
sudo -u postgres psql << EOF
DROP DATABASE IF EXISTS ${APP_NAME};
DROP USER IF EXISTS ${USER};
CREATE USER ${USER} WITH PASSWORD '${DB_PASSWORD}';
CREATE DATABASE ${APP_NAME} OWNER ${USER};
GRANT ALL PRIVILEGES ON DATABASE ${APP_NAME} TO ${USER};
\q
EOF

# Configure PostgreSQL for local connections
if ! grep -q "^local.*${USER}" /etc/postgresql/15/main/pg_hba.conf 2>/dev/null; then
    log_info "Configuring PostgreSQL authentication..."
    # Allow local connections with md5
    sed -i 's/scram-sha-256/md5/g' /etc/postgresql/15/main/pg_hba.conf 2>/dev/null || true
    sed -i 's/peer/md5/g' /etc/postgresql/15/main/pg_hba.conf 2>/dev/null || true
    systemctl restart postgresql
fi

# Start and enable Redis
log_info "Configuring Redis..."
systemctl start redis-server
systemctl enable redis-server

# Generate Redis password
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)

# Update Redis configuration
if [[ -f /etc/redis/redis.conf ]]; then
    sed -i "s/^# requirepass .*/requirepass ${REDIS_PASSWORD}/" /etc/redis/redis.conf
    sed -i "s/^requirepass .*/requirepass ${REDIS_PASSWORD}/" /etc/redis/redis.conf
    # Add requirepass if not present
    if ! grep -q "^requirepass" /etc/redis/redis.conf; then
        echo "requirepass ${REDIS_PASSWORD}" >> /etc/redis/redis.conf
    fi
    systemctl restart redis-server
fi

log_success "Database setup completed"

# ═══════════════════════════════════════════════════════════════
# 4. Application Setup
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[4/9] Application Setup"
echo "═══════════════════════════════════════════════════════════════"

# Create application directory
log_info "Creating application directory..."
mkdir -p "$APP_DIR"
chown "$USER:$USER" "$APP_DIR"

# Determine repository URL
if [[ -z "$GITHUB_REPO" ]]; then
    # Try to detect from git remote
    if [[ -d ".git" ]]; then
        GITHUB_REPO=$(git remote get-url origin 2>/dev/null || echo "")
    fi
fi

if [[ -z "$GITHUB_REPO" ]]; then
    log_warn "No GitHub repository URL provided"
    log_info "Please set GITHUB_REPO environment variable or run from a git repository"
    log_info "Skipping git clone - you'll need to manually copy the code"
else
    # Clone or pull code
    if [[ -d "$APP_DIR/.git" ]]; then
        log_info "Pulling latest code..."
        sudo -u "$USER" git -C "$APP_DIR" pull origin main 2>/dev/null || sudo -u "$USER" git -C "$APP_DIR" pull origin master 2>/dev/null || true
    else
        log_info "Cloning repository..."
        sudo -u "$USER" git clone "$GITHUB_REPO" "$APP_DIR"
    fi
fi

# Create virtual environment
log_info "Creating Python virtual environment..."
if [[ ! -d "$APP_DIR/.venv" ]]; then
    sudo -u "$USER" $PYTHON_CMD -m venv "$APP_DIR/.venv"
fi

# Install dependencies
log_info "Installing Python dependencies..."
sudo -u "$USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip wheel setuptools

if [[ -f "$APP_DIR/requirements.txt" ]]; then
    sudo -u "$USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" --quiet
else
    log_warn "No requirements.txt found in $APP_DIR"
fi

# Create necessary directories
mkdir -p "$APP_DIR/data" "$APP_DIR/learning_data" "$APP_DIR/logs" "$APP_DIR/static"
chown -R "$USER:$USER" "$APP_DIR/data" "$APP_DIR/learning_data" "$APP_DIR/logs" "$APP_DIR/static"

log_success "Application setup completed"

# ═══════════════════════════════════════════════════════════════
# 5. Environment Configuration
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[5/9] Environment Configuration"
echo "═══════════════════════════════════════════════════════════════"

log_info "Creating environment configuration..."

# Generate secure secrets
SECRET_KEY=$(openssl rand -hex 64)
ADMIN_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
ORCHESTRATOR_TOKEN=$(openssl rand -hex 32)

# Get RTX 4090 settings from user input or use defaults
RTX4090_HOST="${RTX4090_HOST:-192.168.68.125}"
RTX4090_PORT="${RTX4090_PORT:-8080}"

# Create .env file
sudo -u "$USER" tee "$APP_DIR/.env" > /dev/null << EOF
# BI-IDE v8 - Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://${USER}:${DB_PASSWORD}@localhost:5432/${APP_NAME}

# Redis
REDIS_URL=redis://:${REDIS_PASSWORD}@localhost:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}

# Security
SECRET_KEY=${SECRET_KEY}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
ORCHESTRATOR_TOKEN=${ORCHESTRATOR_TOKEN}
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS
CORS_ORIGINS=https://${DOMAIN},http://localhost:5173,http://localhost:3000

# RTX 4090 AI Server
RTX4090_HOST=${RTX4090_HOST}
RTX4090_PORT=${RTX4090_PORT}

# AI Core
AI_CORE_HOST=${AI_CORE_HOST:-}

# Auto Sync Checkpoints
AUTO_SYNC_CHECKPOINTS=1
EOF

chmod 600 "$APP_DIR/.env"
chown "$USER:$USER" "$APP_DIR/.env"

log_success "Environment configuration created"

# ═══════════════════════════════════════════════════════════════
# 6. Database Migrations & Initialization
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[6/9] Database Migrations"
echo "═══════════════════════════════════════════════════════════════"

if [[ -f "$APP_DIR/alembic.ini" ]]; then
    log_info "Running database migrations..."
    cd "$APP_DIR"
    sudo -u "$USER" "$APP_DIR/.venv/bin/alembic" upgrade head 2>/dev/null || log_warn "Migration may have failed - continuing anyway"
else
    log_warn "No alembic.ini found - skipping migrations"
fi

# Create default admin
if [[ -f "$APP_DIR/scripts/create_default_admin.py" ]]; then
    log_info "Creating default admin user..."
    cd "$APP_DIR"
    sudo -u "$USER" bash -c "export ADMIN_PASSWORD='$ADMIN_PASSWORD' && export DATABASE_URL='postgresql+asyncpg://${USER}:${DB_PASSWORD}@localhost:5432/${APP_NAME}' && '$APP_DIR/.venv/bin/python' scripts/create_default_admin.py" 2>/dev/null || log_warn "Admin creation may have failed - continuing anyway"
else
    log_warn "No create_default_admin.py script found"
fi

log_success "Database initialization completed"

# ═══════════════════════════════════════════════════════════════
# 7. Systemd Services
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[7/9] Systemd Services"
echo "═══════════════════════════════════════════════════════════════"

log_info "Creating systemd services..."

# API Service
tee /etc/systemd/system/${APP_NAME}-api.service > /dev/null << 'EOF'
[Unit]
Description=BI-IDE API Server
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=USER_PLACEHOLDER
Group=USER_PLACEHOLDER
WorkingDirectory=APP_DIR_PLACEHOLDER
EnvironmentFile=APP_DIR_PLACEHOLDER/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=APP_DIR_PLACEHOLDER/.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=30

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=APP_DIR_PLACEHOLDER/data APP_DIR_PLACEHOLDER/learning_data APP_DIR_PLACEHOLDER/logs

[Install]
WantedBy=multi-user.target
EOF

# Worker Service
tee /etc/systemd/system/${APP_NAME}-worker.service > /dev/null << 'EOF'
[Unit]
Description=BI-IDE Background Worker
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=USER_PLACEHOLDER
Group=USER_PLACEHOLDER
WorkingDirectory=APP_DIR_PLACEHOLDER
EnvironmentFile=APP_DIR_PLACEHOLDER/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=APP_DIR_PLACEHOLDER/.venv/bin/celery -A core.celery_config worker --loglevel=info --concurrency=2
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
Restart=always
RestartSec=5
TimeoutStartSec=30
TimeoutStopSec=30

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=APP_DIR_PLACEHOLDER/data APP_DIR_PLACEHOLDER/learning_data APP_DIR_PLACEHOLDER/logs

[Install]
WantedBy=multi-user.target
EOF

# Replace placeholders
sed -i "s|USER_PLACEHOLDER|$USER|g" /etc/systemd/system/${APP_NAME}-api.service
sed -i "s|APP_DIR_PLACEHOLDER|$APP_DIR|g" /etc/systemd/system/${APP_NAME}-api.service
sed -i "s|USER_PLACEHOLDER|$USER|g" /etc/systemd/system/${APP_NAME}-worker.service
sed -i "s|APP_DIR_PLACEHOLDER|$APP_DIR|g" /etc/systemd/system/${APP_NAME}-worker.service

# Reload systemd
systemctl daemon-reload
systemctl enable ${APP_NAME}-api
systemctl enable ${APP_NAME}-worker

log_success "Systemd services created and enabled"

# ═══════════════════════════════════════════════════════════════
# 8. Nginx Configuration
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[8/9] Nginx Configuration"
echo "═══════════════════════════════════════════════════════════════"

log_info "Configuring Nginx..."

# Create Nginx configuration
tee /etc/nginx/sites-available/${APP_NAME} > /dev/null << EOF
# Rate limiting zones
limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=auth_limit:10m rate=5r/m;

# Upstream for API
upstream ${APP_NAME}_api {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name ${DOMAIN};
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Logging
    access_log /var/log/nginx/${APP_NAME}-access.log;
    error_log /var/log/nginx/${APP_NAME}-error.log;
    
    # Large uploads for AI models
    client_max_body_size 500m;
    client_body_timeout 300s;
    
    # Static files
    location /static {
        alias ${APP_DIR}/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://${APP_NAME}_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Auth endpoints (stricter rate limiting)
    location /api/v1/auth/ {
        limit_req zone=auth_limit burst=10 nodelay;
        
        proxy_pass http://${APP_NAME}_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://${APP_NAME}_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://${APP_NAME}_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # All other requests
    location / {
        proxy_pass http://${APP_NAME}_api;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/${APP_NAME} /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t || {
    log_error "Nginx configuration test failed"
    exit 1
}

# Reload Nginx
systemctl reload nginx

log_success "Nginx configured"

# ═══════════════════════════════════════════════════════════════
# 9. SSL Certificate
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[9/9] SSL Certificate"
echo "═══════════════════════════════════════════════════════════════"

log_info "Obtaining SSL certificate from Let's Encrypt..."

# Obtain SSL certificate
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$ADMIN_EMAIL" --redirect --hsts || {
    log_warn "Certbot failed - SSL will need to be configured manually"
    log_info "You can run: sudo certbot --nginx -d $DOMAIN"
}

# Setup auto-renewal cron job
if ! crontab -l 2>/dev/null | grep -q "certbot renew"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * /usr/bin/certbot renew --quiet --no-self-upgrade") | crontab -
    log_info "Added certbot auto-renewal to crontab"
fi

log_success "SSL certificate configured"

# ═══════════════════════════════════════════════════════════════
# 10. Start Services & Verification
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Starting Services & Verification"
echo "═══════════════════════════════════════════════════════════════"

log_info "Starting services..."

# Start services
systemctl restart ${APP_NAME}-api
systemctl restart ${APP_NAME}-worker

# Wait for API to start
log_info "Waiting for API to start..."
sleep 5

# Health check
log_info "Running health checks..."
HEALTH_PASSED=0
HEALTH_FAILED=0

if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    log_success "Health check passed (/health)"
    ((HEALTH_PASSED++))
else
    log_error "Health check failed (/health)"
    ((HEALTH_FAILED++))
fi

if curl -sf http://localhost:8000/ready > /dev/null 2>&1; then
    log_success "Ready check passed (/ready)"
    ((HEALTH_PASSED++))
else
    log_warn "Ready check failed (/ready) - may need more time"
fi

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ $HEALTH_FAILED -eq 0 ]]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
else
    echo -e "${YELLOW}⚠️ Deployment completed with warnings${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ACCESS INFORMATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🌐 Website:        https://${DOMAIN}"
echo "  🔑 Admin Username: president"
echo "  🔒 Admin Password: ${ADMIN_PASSWORD}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SERVICE COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  View API logs:    sudo journalctl -u ${APP_NAME}-api -f"
echo "  View Worker logs: sudo journalctl -u ${APP_NAME}-worker -f"
echo "  Restart API:      sudo systemctl restart ${APP_NAME}-api"
echo "  Restart Worker:   sudo systemctl restart ${APP_NAME}-worker"
echo "  Check status:     sudo systemctl status ${APP_NAME}-api"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  IMPORTANT: SAVE THESE CREDENTIALS SECURELY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  The admin password is displayed above. Copy it now - it won't"
echo "  be shown again!"
echo ""

if [[ $HEALTH_FAILED -gt 0 ]]; then
    echo -e "${YELLOW}Note: Some health checks failed. Services may need more time${NC}"
    echo -e "${YELLOW}to start. Check logs with: sudo journalctl -u ${APP_NAME}-api -f${NC}"
    echo ""
fi

# Save credentials to file for reference
cat > "$APP_DIR/.credentials.txt" << EOF
BI-IDE v8 Deployment Credentials
Generated: $(date)
================================

Domain: ${DOMAIN}
Admin Username: president
Admin Password: ${ADMIN_PASSWORD}

Database:
  User: ${USER}
  Password: ${DB_PASSWORD}
  Database: ${APP_NAME}

Redis Password: ${REDIS_PASSWORD}

Environment File: ${APP_DIR}/.env
================================
KEEP THIS FILE SECURE!
================================
EOF

chmod 600 "$APP_DIR/.credentials.txt"
chown "$USER:$USER" "$APP_DIR/.credentials.txt"

log_info "Credentials saved to: $APP_DIR/.credentials.txt"
