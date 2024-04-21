#!/bin/bash
# Download the `dataset.csv` file.
# It has to be downloaded seperately as it's too big for GitHub.

set -e

# Get the directory of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Change to the script directory
cd "$script_dir"
mkdir cache
cd cache

# Download the zip file using curl
curl -L -o content_tokens.lemma_wo_pos.zip "https://github.com/EvitanRelta/testing-grounds/releases/download/cs3245-hw4-cache/content_tokens.lemma_wo_pos.zip"

# Extract the specific files from the zip
unzip -j content_tokens.lemma_wo_pos.zip "content_tokens.lemma_wo_pos.csv" -d .

echo "Downloaded and extracted cache successfully."
