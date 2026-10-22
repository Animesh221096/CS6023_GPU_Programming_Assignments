#!/bin/bash
# Runs ./main on every input/<name>.in.bin and compares the result with
# output/<name>.out.bin byte for byte.

SOLVER_EXEC="./main"
INPUT_DIR="input"
EXPECTED_DIR="output"
TEMP_DIR="temp_output"
TIME_LIMIT=60   # seconds per test case

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -f "$SOLVER_EXEC" ]; then
    echo -e "${RED}Error: Executable '$SOLVER_EXEC' not found! Build your code first.${NC}"
    exit 1
fi
mkdir -p "$TEMP_DIR"

TOTAL=0; PASSED=0; FAILED=0
echo "=========================================="
echo "      Running CUDA Wildfire Tests         "
echo "=========================================="

for input_file in "$INPUT_DIR"/*.in.bin; do
    [ -e "$input_file" ] || continue
    name=$(basename "$input_file" .in.bin)
    expected_file="$EXPECTED_DIR/$name.out.bin"
    temp_file="$TEMP_DIR/$name.out.bin"
    ((TOTAL++))

    if [ ! -f "$expected_file" ]; then
        echo -e "[${YELLOW}SKIP${NC}] $name -> Missing expected output '$expected_file'"
        ((TOTAL--))
        continue
    fi

    rm -f "$temp_file"
    time_line=$(timeout "$TIME_LIMIT" $SOLVER_EXEC "$input_file" "$temp_file" 2>&1 >/dev/null | grep "Simulation time"; exit ${PIPESTATUS[0]})
    status=$?

    if [ $status -eq 124 ]; then
        echo -e "[${RED}FAIL${NC}] $name -> Timed out after ${TIME_LIMIT}s"
        ((FAILED++))
    elif [ ! -f "$temp_file" ]; then
        echo -e "[${RED}FAIL${NC}] $name -> No output (crashed?)"
        ((FAILED++))
    elif cmp -s "$temp_file" "$expected_file"; then
        echo -e "[${GREEN}PASS${NC}] $name   ${time_line#Simulation time: }"
        ((PASSED++))
    else
        echo -e "[${RED}FAIL${NC}] $name"
        ((FAILED++))
    fi
done

echo "=========================================="
echo -e "Summary: Total: $TOTAL | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
echo "=========================================="

if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    rm -rf "$TEMP_DIR"
fi
