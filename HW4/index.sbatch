#!/bin/bash
#SBATCH --partition=medium

FILE=./dataset.csv
if [ ! -f "$FILE" ]; then
  echo "Dataset not found, downloading"
  bash ./download_dataset.sh
fi

python index.py -i ./dataset.csv -d dictionary.txt -p postings.txt
