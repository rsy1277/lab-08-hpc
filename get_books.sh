#!/bin/bash
set -euo pipefail

# Get the bucket name and book directory
BUCKET="$1"
BOOK_DIR="$2"

# Convert the directory to an absolute path
mkdir -p "$BOOK_DIR"
BOOK_DIR=$(realpath "$BOOK_DIR")

# Copy only .txt files from S3
aws s3 cp "s3://$BUCKET/" "$BOOK_DIR/" \
    --recursive \
    --exclude "*" \
    --include "*.txt"

# Create book_list.txt with a header
echo "File path" > book_list.txt

# Find all book files and sort them
find "$BOOK_DIR" -type f -name "*.txt" | sort >> book_list.txt

# Count the books (excluding the header)
BOOK_COUNT=$(($(wc -l < book_list.txt) - 1))

echo "Wrote $BOOK_COUNT book files to book_list.txt"
