#!/bin/bash
# BI-IDE Desktop Development Setup Script
# This script sets up the development environment for BI-IDE Desktop

set -e

echo "🚀 BI-IDE Desktop Development Setup"
echo "===================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}Rust not found! Please install Rust: https://rustup.rs/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust: $(rustc --version)${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found! Please install Node.js 20+${NC}"
    exit 1
fi
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 20 ]; then
    echo -e "${RED}Node.js version must be 20+. Current: $(node -v)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js: $(node -v)${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm: $(npm -v)${NC}"

# Check Tauri CLI
if ! command -v tauri &> /dev/null; then
    echo -e "${YELLOW}Tauri CLI not found. Installing...${NC}"
    cargo install tauri-cli
fi
echo -e "${GREEN}✓ Tauri CLI installed${NC}"

# Setup directories
echo -e "${BLUE}Setting up directories...${NC}"
cd "$(dirname "$0")/.."

# Install UI dependencies
echo -e "${BLUE}Installing UI dependencies...${NC}"
cd apps/desktop-tauri
npm install

# Install Rust dependencies
echo -e "${BLUE}Installing Rust dependencies...${NC}"
cd src-tauri
cargo fetch

cd ../../..

# Create development environment file
echo -e "${BLUE}Creating development environment...${NC}"
if [ ! -f .env.dev ]; then
    cat > .env.dev << 'EOF'
# BI-IDE Development Environment
DATABASE_URL=sqlite+aiosqlite:///./bi_ide_dev.db
SECRET_KEY=dev-secret-key-not-for-production
DEBUG=true
ENVIRONMENT=development
PORT=8000

# Desktop specific
DESKTOP_DEV=true
SYNC_ENABLED=false
TRAINING_ENABLED=false
EOF
    echo -e "${GREEN}✓ Created .env.dev${NC}"
fi

echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Setup complete! 🎉${NC}"
echo ""
echo "Next steps:"
echo "  1. Start the API server: make dev"
echo "  2. In another terminal, start the desktop app:"
echo "     cd apps/desktop-tauri"
echo "     npm run tauri:dev"
echo ""
echo "Or use the convenience scripts:"
echo "  ./scripts/dev-up.sh - Start everything"
echo "  ./scripts/dev-check.sh - Check status"
echo ""
