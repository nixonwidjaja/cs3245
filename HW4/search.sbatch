#!/bin/bash
#SBATCH --partition=medium

DIRECTORY="./queries"
OUTPUT="output"

for file in "$DIRECTORY"/*; do
  QUERY_FILE="$DIRECTORY/$(basename "$file")"
  OUTPUT_FILE="$OUTPUT$(basename "$file")"
  python search.py -d dictionary.txt -p postings.txt -q "$QUERY_FILE" -o "$OUTPUT_FILE"
done
