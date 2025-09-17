#!/bin/bash

echo "=== Inference Performance Comparison ==="
echo ""

# Create a simple test prompt file if it doesn't exist
echo "Creating test prompt..."
echo "The quick brown fox jumps over the lazy dog" > single_prompt.txt

echo "Testing CPU-only inference..."
echo "----------------------------------------"
time ./inference --cpu --prompts single_prompt.txt 2>/dev/null | grep "Time:"

echo ""
echo "Testing GPU-accelerated inference..."
echo "----------------------------------------"
time ./inference --gpu --prompts single_prompt.txt 2>/dev/null | grep "Time:"

echo ""
echo "Testing Auto GPU/CPU inference..."
echo "----------------------------------------"
time ./inference --auto-gpu --prompts single_prompt.txt 2>/dev/null | grep "Time:"

echo ""
echo "=== Performance Comparison Complete ==="

# Clean up
rm -f single_prompt.txt