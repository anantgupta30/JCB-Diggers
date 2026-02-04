#!/bin/bash
# env.sh - Setup for Task 3

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

python3 -c "import networkx; import numpy" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required python libraries (networkx, numpy) are not installed."
    exit 1
fi

# Compile Gaston from source if binary is missing or incompatible
if [ ! -f "./gaston" ]; then
    echo "Gaston binary missing. Compiling from source..."
    if [ -d "./gaston_src" ]; then
        (cd gaston_src && make clean && make && cp gaston ..)
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compile Gaston."
            exit 1
        fi
        echo "Gaston compiled successfully."
    else
        echo "Error: gaston_src directory not found."
        exit 1
    fi
else
    # Check if the binary works on current OS
    ./gaston 1 /dev/null /dev/null &> /dev/null
    if [ $? -eq 126 ] || [ $? -eq 127 ]; then
        echo "Gaston binary incompatible. Re-compiling from source..."
        (cd gaston_src && make clean && make && cp gaston ..)
    fi
fi

if [ -f "./gaston" ]; then
    chmod +x ./gaston
fi

echo "Environment check complete."
