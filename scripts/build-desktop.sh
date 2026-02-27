#!/bin/bash
# BI-IDE Desktop - Build Script

set -e

echo "🔨 Building BI-IDE Desktop"
echo "=========================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")/../apps/desktop-tauri"

# Parse arguments
RELEASE=false
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            RELEASE=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
if [ ! -d node_modules ]; then
    echo -e "${YELLOW}Installing Node dependencies...${NC}"
    npm install
fi

echo -e "${BLUE}Building...${NC}"

if [ "$RELEASE" = true ]; then
    echo -e "${YELLOW}Release build${NC}"
    npm run tauri:build
else
    echo -e "${YELLOW}Debug build${NC}"
    npm run tauri:build -- --debug
fi

echo ""
echo -e "${GREEN}==========================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Output location:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  src-tauri/target/release/bundle/macos/"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "  src-tauri/target/release/bundle/msi/"
else
    echo "  src-tauri/target/release/bundle/deb/"
fi
