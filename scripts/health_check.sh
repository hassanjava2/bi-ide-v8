#!/bin/bash
#
# Health Check Script - BI-IDE v8
# سكربت فحص صحة النظام
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8000}"
RTX4090_HOST="${RTX4090_HOST:-192.168.68.125}"
RTX4090_PORT="${RTX4090_PORT:-8080}"
TIMEOUT="${TIMEOUT:-5}"

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0

# Functions
check_passed() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_failed() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check HTTP endpoint
check_http() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    echo -n "Checking $name... "
    
    if command -v curl &> /dev/null; then
        response=$(curl -s -o /dev/null -w "%{http_code},%{time_total}" --max-time $TIMEOUT "$url" 2>/dev/null || echo "000,0")
    elif command -v wget &> /dev/null; then
        response=$(wget -q -O /dev/null --timeout=$TIMEOUT --server-response "$url" 2>&1 | grep "HTTP/" | tail -1 | awk '{print $2}' || echo "000")
        response="${response},0"
    else
        check_warning "Neither curl nor wget available"
        return
    fi
    
    http_code=$(echo "$response" | cut -d',' -f1)
    response_time=$(echo "$response" | cut -d',' -f2)
    response_time_ms=$(echo "$response_time * 1000" | bc 2>/dev/null || echo "0")
    
    if [ "$http_code" = "$expected_code" ]; then
        check_passed "$name (HTTP $http_code, ${response_time_ms%.*}ms)"
    elif [ "$http_code" = "000" ]; then
        check_failed "$name (Connection failed)"
    else
        check_failed "$name (Expected $expected_code, got $http_code)"
    fi
}

# Check TCP port
check_port() {
    local name=$1
    local host=$2
    local port=$3
    
    echo -n "Checking $name (port $port)... "
    
    if command -v nc &> /dev/null; then
        if nc -z -w $TIMEOUT "$host" "$port" 2>/dev/null; then
            check_passed "$name (port $port open)"
        else
            check_failed "$name (port $port closed)"
        fi
    elif command -v timeout &> /dev/null && command -v bash &> /dev/null; then
        if timeout $TIMEOUT bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null; then
            check_passed "$name (port $port open)"
        else
            check_failed "$name (port $port closed)"
        fi
    else
        check_warning "Cannot check port $port (nc or timeout not available)"
    fi
}

# Check process
check_process() {
    local name=$1
    local process_name=$2
    
    echo -n "Checking process $name... "
    
    if pgrep -x "$process_name" > /dev/null 2>&1 || pgrep -f "$process_name" > /dev/null 2>&1; then
        check_passed "$name (running)"
    else
        check_failed "$name (not running)"
    fi
}

# Check disk space
check_disk() {
    local threshold=${1:-90}
    
    echo -n "Checking disk space... "
    
    usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$usage" -lt "$threshold" ]; then
        check_passed "Disk usage ${usage}% (threshold ${threshold}%)"
    else
        check_failed "Disk usage ${usage}% exceeds threshold ${threshold}%"
    fi
}

# Check memory
check_memory() {
    local threshold=${1:-90}
    
    echo -n "Checking memory... "
    
    if command -v free &> /dev/null; then
        usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [ "$usage" -lt "$threshold" ]; then
            check_passed "Memory usage ${usage}% (threshold ${threshold}%)"
        else
            check_failed "Memory usage ${usage}% exceeds threshold ${threshold}%"
        fi
    else
        check_warning "Cannot check memory (free not available)"
    fi
}

# Main
echo "=========================================="
echo "BI-IDE v8 Health Check"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  API: $API_HOST:$API_PORT"
echo "  RTX 4090: $RTX4090_HOST:$RTX4090_PORT"
echo "  Timeout: ${TIMEOUT}s"
echo ""

# System checks
echo "------------------------------------------"
echo "System Checks"
echo "------------------------------------------"
check_disk 85
check_memory 85
echo ""

# Port checks
echo "------------------------------------------"
echo "Port Checks"
echo "------------------------------------------"
check_port "BI-IDE API" "$API_HOST" "$API_PORT"
check_port "RTX 4090" "$RTX4090_HOST" "$RTX4090_PORT"
check_port "Redis" "localhost" "6379"
check_port "PostgreSQL" "localhost" "5432"
echo ""

# HTTP checks
echo "------------------------------------------"
echo "HTTP Endpoint Checks"
echo "------------------------------------------"
check_http "BI-IDE Status" "http://$API_HOST:$API_PORT/api/v1/system/status"
check_http "RTX 4090 Status" "http://$RTX4090_HOST:$RTX4090_PORT/status"
check_http "Gateway Status" "http://$API_HOST:$API_PORT/gateway/status"
echo ""

# Process checks (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "------------------------------------------"
    echo "Process Checks"
    echo "------------------------------------------"
    check_process "Python API" "python"
    check_process "Redis" "redis-server"
    echo ""
fi

# Summary
echo "=========================================="
echo "Health Check Summary"
echo "=========================================="
echo -e "Checks Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Checks Failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed!${NC}"
    exit 1
fi
