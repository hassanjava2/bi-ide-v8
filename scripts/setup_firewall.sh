#!/bin/bash
#
# Firewall Setup Script - BI-IDE v8
# سكربت إعداد جدار الحماية لـ Ubuntu
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PORTS=(8000 8080 9090 6379 5432)
PORT_NAMES=(
    "BI-IDE API Server"
    "RTX 4090 Inference Server"
    "Prometheus Metrics"
    "Redis Cache"
    "PostgreSQL Database"
)

echo "=========================================="
echo "BI-IDE v8 Firewall Setup"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    echo -e "${RED}Cannot detect OS${NC}"
    exit 1
fi

echo "Detected OS: $OS"

# Install UFW if not present
if ! command -v ufw &> /dev/null; then
    echo -e "${YELLOW}Installing UFW...${NC}"
    apt-get update
    apt-get install -y ufw
fi

# Reset UFW (optional - ask user)
echo ""
read -p "Reset UFW to default? (y/N): " reset_ufw
if [[ $reset_ufw =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Resetting UFW...${NC}"
    ufw --force reset
fi

# Default policies
echo ""
echo "Setting default policies..."
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (important!)
echo "Allowing SSH (port 22)..."
ufw allow 22/tcp comment 'SSH Access'

# Allow required ports
echo ""
echo "Allowing required ports..."
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name=${PORT_NAMES[$i]}
    echo "  - Port $port: $name"
    ufw allow $port/tcp comment "$name"
    ufw allow $port/udp comment "$name (UDP)"
done

# Allow local network (optional)
echo ""
read -p "Allow local network access (192.168.x.x)? (Y/n): " allow_local
if [[ ! $allow_local =~ ^[Nn]$ ]]; then
    echo -e "${GREEN}Allowing local network...${NC}"
    ufw allow from 192.168.0.0/16 comment 'Local Network'
    ufw allow from 10.0.0.0/8 comment 'Private Network'
    ufw allow from 172.16.0.0/12 comment 'Private Network'
fi

# Enable UFW
echo ""
echo -e "${YELLOW}Enabling UFW...${NC}"
ufw --force enable

# Show status
echo ""
echo "=========================================="
echo -e "${GREEN}Firewall Status:${NC}"
echo "=========================================="
ufw status verbose

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Required ports allowed:"
for i in "${!PORTS[@]}"; do
    echo "  - Port ${PORTS[$i]}: ${PORT_NAMES[$i]}"
done
echo ""
echo "To check status later, run: sudo ufw status"
echo "To disable firewall: sudo ufw disable"
echo "To allow additional port: sudo ufw allow <port>"
