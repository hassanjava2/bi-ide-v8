#!/bin/bash
# BI-IDE v8 Production Deployment Script
# Usage: ./deploy-production.sh [domain] [email]

set -e

DOMAIN=${1:-"bi-ide.example.com"}
EMAIL=${2:-"admin@example.com"}
INSTALL_DIR="/opt/bi-ide"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 BI-IDE v8 Production Deployment${NC}"
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root or with sudo${NC}"
    exit 1
fi

# Update system
echo -e "${YELLOW}📦 Updating system packages...${NC}"
apt-get update && apt-get upgrade -y

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
apt-get install -y \
    docker.io \
    docker-compose \
    nginx \
    certbot \
    python3-certbot-nginx \
    git \
    curl \
    htop \
    ufw

# Setup firewall
echo -e "${YELLOW}🛡️ Configuring firewall...${NC}"
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow http
ufw allow https
ufw allow 9090/tcp  # Prometheus (localhost only)
ufw allow 3001/tcp  # Grafana (localhost only)
ufw --force enable

# Create installation directory
echo -e "${YELLOW}📁 Creating installation directory...${NC}"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone repository (if not already present)
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}📥 Cloning repository...${NC}"
    git clone https://github.com/yourusername/bi-ide-v8.git .
fi

# Create environment file
echo -e "${YELLOW}⚙️ Creating environment configuration...${NC}"
cat > .env << EOF
# Database
POSTGRES_USER=bi_ide_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=bi_ide

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
SECRET_KEY=$(openssl rand -hex 64)
ADMIN_PASSWORD=$(openssl rand -base64 24)

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 24)

# Domain
DOMAIN=$DOMAIN

# RTX 4090
RTX4090_HOST=192.168.68.125
RTX4090_PORT=8080
EOF

# Create required directories
echo -e "${YELLOW}📁 Creating data directories...${NC}"
mkdir -p data/ssl
mkdir -p data/certbot
mkdir -p logs/nginx
mkdir -p logs/api
mkdir -p learning_data

# Build UI
echo -e "${YELLOW}🔨 Building UI...${NC}"
cd ui
npm ci
npm run build
cd ..

# Obtain SSL certificate
echo -e "${YELLOW}🔒 Obtaining SSL certificate...${NC}"
certbot certonly --standalone -d $DOMAIN --agree-tos -m $EMAIL --non-interactive || true

# Copy SSL certificates
if [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    cp -r /etc/letsencrypt/live/$DOMAIN/* data/ssl/
fi

# Update nginx config with domain
sed -i "s/bi-ide.example.com/$DOMAIN/g" deploy/nginx.conf

# Start services
echo -e "${YELLOW}🐳 Starting Docker services...${NC}"
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# Wait for database
echo -e "${YELLOW}⏳ Waiting for database...${NC}"
sleep 10

# Run migrations
echo -e "${YELLOW}🔄 Running database migrations...${NC}"
docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head || true

# Create admin user
echo -e "${YELLOW}👤 Creating admin user...${NC}"
docker-compose -f docker-compose.prod.yml exec -T api python -c "
import asyncio
from core.database import db_manager
from core.user_service import UserService, DEFAULT_ROLES

async def setup():
    await db_manager.initialize()
    async with db_manager.get_session() as session:
        # Create roles
        role_service = RoleService(session)
        for role_name, role_data in DEFAULT_ROLES.items():
            existing = await role_service.get_role_by_name(role_name)
            if not existing:
                await role_service.create_role(
                    name=role_name,
                    description=role_data['description'],
                    permissions=role_data['permissions']
                )
                print(f'Created role: {role_name}')
        
        # Create admin user
        user_service = UserService(session)
        existing = await user_service.get_user_by_username('admin')
        if not existing:
            import os
            await user_service.create_user(
                username='admin',
                email='admin@$DOMAIN',
                password=os.getenv('ADMIN_PASSWORD'),
                is_superuser=True,
                role_names=['admin']
            )
            print('Created admin user')

asyncio.run(setup())
" || true

# Setup SSL auto-renewal
echo -e "${YELLOW}🔄 Setting up SSL auto-renewal...${NC}"
(crontab -l 2>/dev/null; echo "0 12 * * * certbot renew --quiet --deploy-hook 'docker-compose -f $INSTALL_DIR/docker-compose.prod.yml exec nginx nginx -s reload'") | crontab -

# Health check
echo -e "${YELLOW}🏥 Running health check...${NC}"
sleep 5
if curl -sf http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ API is healthy${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    exit 1
fi

# Print summary
echo ""
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo ""
echo "🌐 Website: https://$DOMAIN"
echo "📊 Grafana: https://$DOMAIN:3001"
echo "🔍 Prometheus: https://$DOMAIN:9090"
echo ""
echo "📁 Installation Directory: $INSTALL_DIR"
echo "📄 Environment File: $INSTALL_DIR/.env"
echo "📝 Logs: $INSTALL_DIR/logs/"
echo ""
echo "🛠️ Useful Commands:"
echo "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "  Restart: docker-compose -f docker-compose.prod.yml restart"
echo "  Update:  docker-compose -f docker-compose.prod.yml pull && docker-compose -f docker-compose.prod.yml up -d"
echo ""
echo -e "${GREEN}🎉 BI-IDE v8 is now live!${NC}"
