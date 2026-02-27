#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# BI-IDE v8 - Docker Production Deployment Script
# ═══════════════════════════════════════════════════════════════

# Configuration
APP_NAME="bi-ide"
DEPLOY_DIR="${1:-/opt/${APP_NAME}}"
DOMAIN="${2:-}"
ADMIN_EMAIL="${3:-admin@localhost}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo "Usage: $0 <deploy-dir> <domain> [admin-email]"
    echo "Example: $0 /opt/bi-ide bi-ide.example.com admin@example.com"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

log_info "Deploying BI-IDE Docker setup to $DEPLOY_DIR for domain $DOMAIN..."

# ═══════════════════════════════════════════════════════════════
# 1. System Setup
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[1/6] System Setup"
echo "═══════════════════════════════════════════════════════════════"

# Install Docker if not present (Ubuntu/Debian)
if ! command -v docker &> /dev/null; then
    log_info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker $USER || true
    log_success "Docker installed. You may need to logout and login again."
fi

# Create deployment directory
log_info "Creating deployment directory..."
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

log_success "System setup completed"

# ═══════════════════════════════════════════════════════════════
# 2. Generate Secrets
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[2/6] Generating Secrets"
echo "═══════════════════════════════════════════════════════════════"

DB_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/' | cut -c1-32)
SECRET_KEY=$(openssl rand -hex 64)
ADMIN_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
ORCHESTRATOR_TOKEN=$(openssl rand -hex 32)
GRAFANA_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')

log_success "Secrets generated"

# ═══════════════════════════════════════════════════════════════
# 3. Environment Configuration
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[3/6] Environment Configuration"
echo "═══════════════════════════════════════════════════════════════"

log_info "Creating .env file..."

cat > "$DEPLOY_DIR/.env" << EOF
# BI-IDE v8 - Docker Production Environment
VERSION=latest

# Database
DB_PASSWORD=${DB_PASSWORD}

# Redis
REDIS_PASSWORD=${REDIS_PASSWORD}

# Security
SECRET_KEY=${SECRET_KEY}
ADMIN_PASSWORD=${ADMIN_PASSWORD}
ORCHESTRATOR_TOKEN=${ORCHESTRATOR_TOKEN}

# CORS
CORS_ORIGINS=https://${DOMAIN}

# RTX 4090 AI Server
RTX4090_HOST=192.168.68.125
RTX4090_PORT=8080

# Grafana
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
GRAFANA_ROOT_URL=https://${DOMAIN}/grafana
EOF

chmod 600 "$DEPLOY_DIR/.env"
log_success "Environment configuration created"

# ═══════════════════════════════════════════════════════════════
# 4. Copy Configuration Files
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[4/6] Copying Configuration Files"
echo "═══════════════════════════════════════════════════════════════"

log_info "Please copy the following files to $DEPLOY_DIR:"
echo "  - docker-compose.prod.yml"
echo "  - nginx/nginx.conf"
echo "  - monitoring/prometheus.yml (optional)"
echo "  - monitoring/grafana/ (optional)"
echo "  - Dockerfile"
echo "  - Dockerfile.worker"
echo "  - init.sql"
echo "  - static/ directory"

read -p "Press Enter when files are copied..."

# Create required directories
mkdir -p nginx/conf.d
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p static

log_success "Configuration files ready"

# ═══════════════════════════════════════════════════════════════
# 5. Deploy with Docker Compose
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[5/6] Deploying Services"
echo "═══════════════════════════════════════════════════════════════"

log_info "Starting services..."

# Determine docker compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose -f docker-compose.prod.yml --env-file .env"
else
    COMPOSE_CMD="docker-compose -f docker-compose.prod.yml --env-file .env"
fi

# Pull and build images
$COMPOSE_CMD pull || true
$COMPOSE_CMD build

# Start services
$COMPOSE_CMD up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 15

# ═══════════════════════════════════════════════════════════════
# 6. SSL Certificate
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "[6/6] SSL Certificate"
echo "═══════════════════════════════════════════════════════════════"

log_info "Obtaining SSL certificate..."

# Run certbot
$COMPOSE_CMD run --rm certbot certonly \
    --webroot \
    --webroot-path /var/www/certbot \
    --email "$ADMIN_EMAIL" \
    --agree-tos \
    --no-eff-email \
    -d "$DOMAIN" || {
    log_warn "Certbot failed - SSL will need to be configured manually"
}

# Restart nginx to pick up new certificate
$COMPOSE_CMD restart nginx

log_success "SSL certificate configured"

# ═══════════════════════════════════════════════════════════════
# 7. Health Check
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Health Check"
echo "═══════════════════════════════════════════════════════════════"

log_info "Running health checks..."

sleep 5

if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    log_success "Health check passed"
else
    log_warn "Health check failed - checking service logs..."
    $COMPOSE_CMD logs --tail=50 api
fi

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ Docker deployment successful!${NC}"
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
echo "  DOCKER COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  View logs:        $COMPOSE_CMD logs -f"
echo "  View API logs:    $COMPOSE_CMD logs -f api"
echo "  Restart:          $COMPOSE_CMD restart"
echo "  Stop:             $COMPOSE_CMD down"
echo "  Update:           $COMPOSE_CMD pull && $COMPOSE_CMD up -d"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  IMPORTANT: SAVE THESE CREDENTIALS SECURELY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save credentials
cat > "$DEPLOY_DIR/.credentials.txt" << EOF
BI-IDE v8 Docker Deployment Credentials
Generated: $(date)
================================

Domain: ${DOMAIN}
Admin Username: president
Admin Password: ${ADMIN_PASSWORD}

Database Password: ${DB_PASSWORD}
Redis Password: ${REDIS_PASSWORD}
Grafana Password: ${GRAFANA_PASSWORD}

Environment File: ${DEPLOY_DIR}/.env
================================
KEEP THIS FILE SECURE!
================================
EOF

chmod 600 "$DEPLOY_DIR/.credentials.txt"
log_info "Credentials saved to: $DEPLOY_DIR/.credentials.txt"
