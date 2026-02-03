#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

# 1. Clean Duplicates
echo "Step 1: Preprocessing..."
python3 "$SCRIPT_DIR/preprocess.py" "$1" "clean_db_temp.txt"

# 2. Pass A: "The Safety Net" (Coverage)
# Settings: 30 Features, Length 5, Support 10%-50%
# Goal: Ensure even small/simple graphs get a vector (avoids the 40k bug).
echo "Step 2: Mining Safety Features (Coverage)..."
python3 "$SCRIPT_DIR/miner.py" "clean_db_temp.txt" "feats_coverage.txt" --min_r 0.10 --max_r 0.50 --len 5 --k 30

# 3. Pass B: "The Spear" (Precision)
# Settings: 20 Features, Length 10, Support 2%-10%
# Goal: Use rare, long paths to drastically cut candidates for complex queries.
echo "Step 3: Mining Specific Features (Precision)..."
python3 "$SCRIPT_DIR/miner.py" "clean_db_temp.txt" "feats_precision.txt" --min_r 0.02 --max_r 0.10 --len 10 --k 20

# 4. Combine
echo "Step 4: Combining Features..."
cat "feats_coverage.txt" "feats_precision.txt" > "features.txt"

# Cleanup
rm "clean_db_temp.txt" "feats_coverage.txt" "feats_precision.txt"
echo "Done! features.txt generated with 30 Coverage + 20 Precision features."