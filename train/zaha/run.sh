#!/usr/bin/env bash

# Exit immediately if any command exits with a non-zero status (fail fast).
set -e

echo "== [RUN] Step 0: Preprocessing S3DIS =="
bash /train/zaha/0-preprocess.sh
echo "== [RUN] Step 0: DONE =="
echo

# echo "== [RUN] Step 0: Stats =="
# bash /train/zaha/0-stats.sh
# echo "== [RUN] Step 0: DONE =="
# echo

echo "== [RUN] Step 1: Training PTv3 =="
bash /train/zaha/1-train.sh
echo "== [RUN] Step 1: DONE =="
echo

# echo "== [RUN] Step 2: Testing PTv3 =="
# bash /train/zaha/2-test.sh
# echo "== [RUN] Step 2: DONE =="
# echo

echo "== [RUN] All steps finished successfully. =="
