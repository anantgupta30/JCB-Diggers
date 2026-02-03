#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

python3 "$SCRIPT_DIR/vectorizer.py" "$1" "$2" "$3"