#!/bin/bash
set -e

BASE_URL="https://r0k.us/graphics/kodak/kodak"
OUTPUT_DIR="datasets/kodak"

mkdir -p "$OUTPUT_DIR"

echo "Downloading Kodak dataset into $OUTPUT_DIR ..."

for i in $(seq -w 1 24); do
    file="kodim${i}.png"
    url="${BASE_URL}/${file}"
    
    echo "Downloading $file ..."
    
    curl -L --fail -o "${OUTPUT_DIR}/${file}" "$url"
done

echo "Download completed! Files are saved in: $OUTPUT_DIR"