# BI-IDE v8 - Deployment Guide

Complete deployment guide for BI-IDE v8 production environment.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [VPS Deployment (Native)](#vps-deployment-native)
3. [Docker Deployment](#docker-deployment)
4. [CI/CD Setup](#cicd-setup)
5. [Post-Deployment](#post-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Server Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 50 GB SSD | 100+ GB SSD |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 LTS |

### Domain Requirements

- A registered domain name
- DNS A record pointing to your VPS IP
- Port 80 and 443 open in firewall

### Required Secrets

Generate these before deployment:

```bash
# Secret key (64 hex chars)
openssl rand -hex 64

# Passwords
openssl rand -base64 24

# Orchestrator token
openssl rand -hex 32
```

---

## VPS Deployment (Native)

### Quick Start

```bash
# 1. SSH to your VPS
ssh root@your-vps-ip

# 2. Clone the repository
git clone https://github.com/yourusername/bi-ide-v8.git /opt/bi-ide
cd /opt/bi-ide

# 3. Run the deployment script
sudo chmod +x scripts/deploy-vps.sh
sudo ./scripts/deploy-vps.sh bi-ide.yourdomain.com admin@yourdomain.com
```

### What the Script Does

1. **System Setup**
   - Installs Python 3.11, PostgreSQL 15, Redis, Nginx, Certbot
   - Creates `biide` system user
   - Configures UFW firewall

2. **Database Setup**
   - Creates PostgreSQL database and user
   - Configures Redis with authentication
   - Generates secure passwords

3. **Application Setup**
   - Creates virtual environment
   - Installs Python dependencies
   - Sets up directory structure

4. **Environment Configuration**
   - Generates `.env` file with secure secrets
   - Configures database and Redis connections

5. **Database Migrations**
   - Runs Alembic migrations
   - Creates default admin user

6. **Systemd Services**
   - Creates `bi-ide-api.service` (FastAPI/UVicorn)
   - Creates `bi-ide-worker.service` (Celery)
   - Enables auto-start on boot

7. **Nginx Configuration**
   - Sets up reverse proxy
   - Configures rate limiting
   - Sets up security headers

8. **SSL Certificate**
   - Obtains Let's Encrypt certificate
   - Configures auto-renewal

9. **Service Verification**
   - Starts all services
   - Runs health checks
   - Displays access credentials

### Service Management

```bash
# Check service status
sudo systemctl status bi-ide-api
sudo systemctl status bi-ide-worker

# View logs
sudo journalctl -u bi-ide-api -f
sudo journalctl -u bi-ide-worker -f

# Restart services
sudo systemctl restart bi-ide-api
sudo systemctl restart bi-ide-worker

# Stop services
sudo systemctl stop bi-ide-api
sudo systemctl stop bi-ide-worker
```

---

## Docker Deployment

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bi-ide-v8.git
cd bi-ide-v8

# 2. Run the Docker deployment script
sudo chmod +x scripts/deploy-docker.sh
sudo ./scripts/deploy-docker.sh /opt/bi-ide bi-ide.yourdomain.com admin@yourdomain.com
```

### Manual Docker Deployment

```bash
# 1. Create deployment directory
mkdir -p /opt/bi-ide
cd /opt/bi-ide

# 2. Copy required files
cp /path/to/bi-ide-v8/docker-compose.prod.yml .
cp /path/to/bi-ide-v8/Dockerfile .
cp /path/to/bi-ide-v8/Dockerfile.worker .
cp /path/to/bi-ide-v8/init.sql .
cp -r /path/to/bi-ide-v8/nginx .
cp -r /path/to/bi-ide-v8/static .

# 3. Create .env file
cat > .env << 'EOF'
VERSION=latest
DB_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/')
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '=+/')
SECRET_KEY=$(openssl rand -hex 64)
ADMIN_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
ORCHESTRATOR_TOKEN=$(openssl rand -hex 32)
CORS_ORIGINS=https://bi-ide.yourdomain.com
RTX4090_HOST=192.168.68.125
RTX4090_PORT=8080
GRAFANA_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
EOF

# 4. Deploy
docker-compose -f docker-compose.prod.yml up -d

# 5. Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head

# 6. Create admin
docker-compose -f docker-compose.prod.yml exec api python scripts/create_default_admin.py
```

### Docker Compose Commands

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs -f api

# Restart services
docker-compose -f docker-compose.prod.yml restart

# Stop services
docker-compose -f docker-compose.prod.yml down

# Update to latest version
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

---

## CI/CD Setup

### GitHub Actions Configuration

1. **Add Secrets to GitHub Repository**:
   - Go to Settings → Secrets and variables → Actions
   - Add the following secrets:

| Secret Name | Description |
|-------------|-------------|
| `VPS_HOST` | Your VPS IP address or hostname |
| `VPS_USER` | SSH username (e.g., `biide`) |
| `VPS_SSH_KEY` | Private SSH key for deployment |
| `VPS_DOMAIN` | Your domain name |
| `SLACK_WEBHOOK` | (Optional) Slack webhook URL for notifications |

2. **SSH Key Setup**:

```bash
# On your VPS, create deploy key
sudo -u biide mkdir -p ~/.ssh
sudo -u biide ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N ""

# Add public key to GitHub Deploy Keys
cat ~/.ssh/github_deploy.pub

# Add private key to GitHub Secrets
cat ~/.ssh/github_deploy
```

3. **The workflow will**:
   - Run tests on every push/PR
   - Build Docker images
   - Deploy to VPS on main branch pushes
   - Send Slack notifications

### Manual Trigger

You can also trigger deployment manually:

1. Go to Actions tab in GitHub
2. Select "Deploy to Production VPS"
3. Click "Run workflow"

---

## Post-Deployment

### Verify Installation

```bash
# Check all services are running
curl https://your-domain.com/health
curl https://your-domain.com/ready

# View logs
sudo journalctl -u bi-ide-api -n 100

# Check resource usage
htop
```

### First Login

1. Navigate to `https://your-domain.com`
2. Login with:
   - Username: `president`
   - Password: (from deployment output)

### Change Admin Password

```bash
# Using the API
curl -X POST https://your-domain.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"president","password":"YOUR_CURRENT_PASSWORD"}'

# Then use the token to change password
```

### Configure RTX 4090 Server

Edit `.env` on the VPS:

```bash
sudo nano /opt/bi-ide/.env
# Update RTX4090_HOST and RTX4090_PORT

sudo systemctl restart bi-ide-api
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check for errors
sudo journalctl -u bi-ide-api -n 50 --no-pager

# Check Python dependencies
sudo -u biide /opt/bi-ide/.venv/bin/pip list

# Test database connection
sudo -u biide /opt/bi-ide/.venv/bin/python -c "
import asyncio
from core.database import db_manager
asyncio.run(db_manager.initialize())
print('DB OK')
"
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
sudo -u postgres psql -d bi_ide -c "\dt"

# Reset database (DANGER: deletes all data)
sudo -u postgres psql -c "DROP DATABASE bi_ide;"
sudo -u postgres psql -c "CREATE DATABASE bi_ide OWNER biide;"
cd /opt/bi-ide && sudo -u biide .venv/bin/alembic upgrade head
```

### SSL Certificate Issues

```bash
# Renew certificate manually
sudo certbot renew --force-renewal

# Check certificate status
sudo certbot certificates

# Debug Nginx SSL
sudo nginx -t
sudo systemctl restart nginx
```

### High Memory Usage

```bash
# Check memory usage
free -h

# Check which processes use most memory
ps aux --sort=-%mem | head -20

# Adjust worker processes in systemd service
sudo nano /etc/systemd/system/bi-ide-api.service
# Change --workers 2 to --workers 1
sudo systemctl daemon-reload
sudo systemctl restart bi-ide-api
```

### Firewall Issues

```bash
# Check UFW status
sudo ufw status

# Allow specific port
sudo ufw allow 8080/tcp

# Disable UFW (not recommended for production)
sudo ufw disable
```

---

## Security Checklist

- [ ] Changed default admin password
- [ ] Generated unique SECRET_KEY
- [ ] Generated unique ORCHESTRATOR_TOKEN
- [ ] Firewall enabled (UFW)
- [ ] Fail2ban installed and running
- [ ] SSL certificate installed
- [ ] Database passwords are strong
- [ ] Redis password is set
- [ ] `.env` file has 600 permissions
- [ ] SSH key authentication only (no password)
- [ ] Regular backups configured
- [ ] Log monitoring enabled

---

## Backup Strategy

### Automated Backups

```bash
# Add to crontab
0 2 * * * /opt/bi-ide/scripts/backup.sh
```

### Manual Backup

```bash
# Database backup
sudo -u postgres pg_dump bi_ide > backup_$(date +%Y%m%d).sql

# Data directory backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz /opt/bi-ide/data /opt/bi-ide/learning_data
```

---

## Support

For issues and questions:
- Check logs: `sudo journalctl -u bi-ide-api -f`
- Run health check: `./scripts/health_check.sh`
- Review this documentation
