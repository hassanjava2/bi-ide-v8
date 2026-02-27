#!/bin/bash
# BI-IDE Desktop Development - Start All Services

set -e

echo "🚀 Starting BI-IDE Desktop Development Environment"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")/.."

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Shutting down...${NC}"
    if [ -n "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# Check if API is already running
echo -e "${BLUE}Checking API server...${NC}"
if curl -s http://localhost:8000/api/v1/status > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server already running${NC}"
else
    echo -e "${BLUE}Starting API server...${NC}"
    
    # Activate virtual environment if it exists
    if [ -d .venv ]; then
        source .venv/bin/activate
    fi
    
    # Start API in background
    python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    
    # Wait for API to be ready
    echo -e "${YELLOW}Waiting for API to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:8000/api/v1/status > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API server started${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:8000/api/v1/status > /dev/null 2>&1; then
        echo -e "${RED}✗ API server failed to start${NC}"
        exit 1
    fi
fi

# Start Desktop App
echo -e "${BLUE}Starting Desktop App...${NC}"
cd apps/desktop-tauri

# Install dependencies if needed
if [ ! -d node_modules ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
fi

# Run Tauri dev
echo -e "${GREEN}Starting Tauri development server...${NC}"
npm run tauri:dev

# Cleanup
cleanup
