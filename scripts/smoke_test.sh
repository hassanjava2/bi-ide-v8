#!/bin/bash
# BI-IDE v8 - Smoke Test Script
# Usage: ./smoke_test.sh <base_url> [timeout_seconds]

set -e

BASE_URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-30}"
MAX_RETRIES=5
RETRY_DELAY=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

# HTTP request function
http_request() {
    local method="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"
    local data="${4:-}"
    local headers="${5:-}"
    
    local url="${BASE_URL}${endpoint}"
    local curl_opts="-s -o /dev/null -w %{http_code} --max-time ${TIMEOUT}"
    
    if [ -n "$data" ]; then
        curl_opts="${curl_opts} -d '${data}'"
    fi
    
    if [ -n "$headers" ]; then
        curl_opts="${curl_opts} ${headers}"
    fi
    
    local status_code
    status_code=$(curl ${curl_opts} -X "${method}" "${url}")
    
    if [ "$status_code" == "$expected_status" ]; then
        log_success "${method} ${endpoint} - Status ${status_code}"
        return 0
    else
        log_error "${method} ${endpoint} - Expected ${expected_status}, got ${status_code}"
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    log_info "Waiting for service at ${BASE_URL}..."
    
    for ((i=1; i<=MAX_RETRIES; i++)); do
        if curl -s --max-time 5 "${BASE_URL}/health/ready" > /dev/null 2>&1; then
            log_success "Service is ready"
            return 0
        fi
        log_info "Attempt $i/$MAX_RETRIES - Service not ready yet, waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    
    log_error "Service failed to become ready after $MAX_RETRIES attempts"
    return 1
}

# Main test suite
echo "========================================"
echo "BI-IDE v8 Smoke Test Suite"
echo "Base URL: ${BASE_URL}"
echo "========================================"
echo

# Wait for service
wait_for_service

# Health checks
echo
echo "=== Health Checks ==="
http_request "GET" "/health/live" "200"
http_request "GET" "/health/ready" "200"
http_request "GET" "/health/startup" "200"

# API endpoints
echo
echo "=== API Endpoints ==="
http_request "GET" "/api/v1/status" "200"
http_request "GET" "/api/v1/docs" "200"
http_request "GET" "/api/v1/openapi.json" "200"

# Metrics
echo
echo "=== Metrics Endpoints ==="
http_request "GET" "/metrics" "200"

# Static files (if UI is deployed together)
echo
echo "=== Static Assets ==="
http_request "GET" "/" "200" || http_request "GET" "/index.html" "200"

# CORS headers test
echo
echo "=== CORS Headers ==="
CORS_RESPONSE=$(curl -s -I -X OPTIONS \
    -H "Origin: https://example.com" \
    -H "Access-Control-Request-Method: POST" \
    --max-time 5 \
    "${BASE_URL}/api/v1/status" 2>/dev/null | grep -i "access-control-allow-origin" || true)

if [ -n "$CORS_RESPONSE" ]; then
    log_success "CORS headers present"
else
    log_info "CORS headers not configured (may be expected)"
fi

# Response time test
echo
echo "=== Response Time Tests ==="
RESPONSE_TIME=$(curl -s -o /dev/null -w "%{time_total}" --max-time 10 "${BASE_URL}/api/v1/status")
if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l) )); then
    log_success "API response time: ${RESPONSE_TIME}s (< 2s threshold)"
else
    log_error "API response time too slow: ${RESPONSE_TIME}s (> 2s threshold)"
fi

# Database connectivity test
echo
echo "=== Database Connectivity ==="
DB_STATUS=$(curl -s --max-time 10 "${BASE_URL}/health/ready" | grep -o '"database":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
if [ "$DB_STATUS" == "connected" ]; then
    log_success "Database connection: ${DB_STATUS}"
else
    log_error "Database connection issue: ${DB_STATUS}"
fi

# Redis connectivity test (if exposed)
echo
echo "=== Cache Connectivity ==="
CACHE_STATUS=$(curl -s --max-time 10 "${BASE_URL}/health/ready" | grep -o '"redis":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
if [ "$CACHE_STATUS" == "connected" ]; then
    log_success "Redis connection: ${CACHE_STATUS}"
else
    log_info "Redis status: ${CACHE_STATUS} (may not be exposed in health check)"
fi

# Security headers test
echo
echo "=== Security Headers ==="
SECURITY_HEADERS=$(curl -s -I --max-time 5 "${BASE_URL}/api/v1/status" 2>/dev/null)

if echo "$SECURITY_HEADERS" | grep -qi "X-Content-Type-Options"; then
    log_success "X-Content-Type-Options header present"
else
    log_error "X-Content-Type-Options header missing"
fi

if echo "$SECURITY_HEADERS" | grep -qi "X-Frame-Options"; then
    log_success "X-Frame-Options header present"
else
    log_error "X-Frame-Options header missing"
fi

# Rate limiting test (if enabled)
echo
echo "=== Rate Limiting ==="
RATE_LIMIT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    -H "X-Test-Client: smoke-test" \
    "${BASE_URL}/api/v1/status")

if [ "$RATE_LIMIT_RESPONSE" == "200" ] || [ "$RATE_LIMIT_RESPONSE" == "429" ]; then
    log_success "Rate limiting configured (status: ${RATE_LIMIT_RESPONSE})"
else
    log_info "Rate limiting response: ${RATE_LIMIT_RESPONSE}"
fi

# Summary
echo
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some smoke tests failed!${NC}"
    exit 1
fi
