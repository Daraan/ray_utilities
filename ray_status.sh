#!/bin/bash
# This utility files puts out ray status for all running ray clusters.

# Do not exit on error
set +e

echo "[TRACE] Running: ray status"
output=$(ray status 2>&1)
status=$?
echo "[TRACE] ray status exited with code $status"

if [[ $status -eq 0 ]]; then
    echo "[TRACE] Success path"
    echo "$output"
    exit 0
fi

if echo "$output" | grep -q "Found multiple active Ray instances:"; then
    echo "[TRACE] Multiple Ray instances detected"
    # Extract the line with the addresses
    line=$(echo "$output" | grep "Found multiple active Ray instances:")
    echo "[TRACE] Error line: $line"
    # Extract addresses inside curly braces
    addresses=$(echo "$line" | grep -oP "\{[^}]+\}" | tr -d '{} ')
    echo "[TRACE] Addresses: $addresses"
    # Split by comma
    IFS=',' read -ra ADDR_ARRAY <<< "$addresses"
    for addr in "${ADDR_ARRAY[@]}"; do
        addr_trimmed=$(echo "$addr" | xargs)
        echo "[TRACE] Trying: ray status --address=$addr_trimmed"
        ray status --address="$addr_trimmed"
    done
else
    echo "[TRACE] Error path"
    echo "$output"
    exit $status
fi
