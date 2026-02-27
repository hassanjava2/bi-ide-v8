#!/bin/bash
# BI-IDE Desktop Development - Check Status

echo "🔍 BI-IDE Development Environment Status"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check API
echo -e "${BLUE}API Server:${NC}"
if curl -s http://localhost:8000/api/v1/status > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Running${NC} (http://localhost:8000)"
    VERSION=$(curl -s http://localhost:8000/api/v1/status | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo -e "  Version: $VERSION"
else
    echo -e "  ${RED}✗ Not running${NC}"
fi

# Check Desktop
echo ""
echo -e "${BLUE}Desktop App:${NC}"
# Check if Tauri dev server is running
if lsof -i :5173 > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ Dev server running${NC} (http://localhost:5173)"
else
    echo -e "  ${YELLOW}○ Dev server not running${NC}"
fi

# Check Rust
echo ""
echo -e "${BLUE}Rust Environment:${NC}"
if command -v rustc &> /dev/null; then
    echo -e "  ${GREEN}✓ $(rustc --version)${NC}"
    echo -e "  Cargo: $(cargo --version)"
else
    echo -e "  ${RED}✗ Rust not installed${NC}"
fi

# Check Node.js
echo ""
echo -e "${BLUE}Node.js Environment:${NC}"
if command -v node &> /dev/null; then
    echo -e "  ${GREEN}✓ Node.js $(node -v)${NC}"
    echo -e "  npm: $(npm -v)"
else
    echo -e "  ${RED}✗ Node.js not installed${NC}"
fi

# Check dependencies
echo ""
echo -e "${BLUE}Dependencies:${NC}"
cd "$(dirname "$0")/../apps/desktop-tauri"

if [ -d node_modules ]; then
    NODE_DEPS=$(ls node_modules | wc -l)
    echo -e "  ${GREEN}✓ Node modules installed${NC} ($NODE_DEPS packages)"
else
    echo -e "  ${RED}✗ Node modules missing${NC}"
fi

if [ -f src-tauri/Cargo.lock ]; then
    echo -e "  ${GREEN}✓ Rust dependencies fetched${NC}"
else
    echo -e "  ${YELLOW}○ Rust dependencies not fetched${NC}"
fi

echo ""
echo "========================================"
echo "Useful commands:"
echo "  Start API:     make dev"
echo "  Start Desktop: cd apps/desktop-tauri && npm run tauri:dev"
echo "  Build Desktop: cd apps/desktop-tauri && npm run tauri:build"
