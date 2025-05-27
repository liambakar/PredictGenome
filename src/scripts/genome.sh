#!/bin/bash

# Set the path to the folder outside the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Run the Python script
python3 "$PARENT_DIR/baseline_genomic.py" "$@"

