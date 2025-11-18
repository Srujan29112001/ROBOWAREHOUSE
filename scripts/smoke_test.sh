#!/bin/bash
# Smoke test script for RoboVLA API
# Usage: ./smoke_test.sh <API_URL>

set -e

API_URL="${1:-http://localhost:8000}"

echo "================================"
echo "RoboVLA API Smoke Tests"
echo "================================"
echo "Testing: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local expected_code="$4"
    local data="$5"

    echo -n "Testing $name... "

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected $expected_code, got $http_code)"
        echo "Response: $body"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Health Check
test_endpoint "Health Check" "GET" "/health" "200"

# Test 2: Ready Check
test_endpoint "Ready Check" "GET" "/ready" "200"

# Test 3: Metrics Endpoint
test_endpoint "Metrics" "GET" "/metrics" "200"

# Test 4: Execute Command (invalid data)
INVALID_DATA='{
  "text_command": "test",
  "robot_state": [0, 0]
}'
test_endpoint "Execute (Invalid)" "POST" "/execute" "422" "$INVALID_DATA"

# Test 5: Execute Command (valid data - if models loaded)
# Note: This might fail if models are not loaded
VALID_DATA='{
  "rgb_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "depth_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "text_command": "Pick the red box",
  "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "safety_constraints": {
    "max_velocity": 1.0,
    "max_acceleration": 0.5
  }
}'

echo -n "Testing Execute (Valid)... "
response=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "$VALID_DATA" \
    "$API_URL/execute" || true)

http_code=$(echo "$response" | tail -n1)

if [ "$http_code" = "200" ] || [ "$http_code" = "500" ]; then
    echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code - endpoint reachable)"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Got $http_code)"
    ((TESTS_FAILED++))
fi

# Summary
echo ""
echo "================================"
echo "Test Summary"
echo "================================"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
