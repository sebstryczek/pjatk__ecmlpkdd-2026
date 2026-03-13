#!/usr/bin/env bash

# Exit immediately if any command exits with a non-zero status (fail fast).
set -e

echo "== [RUN] Step 0: Preprocessing S3DIS =="
bash /train/0-preprocess.sh
echo "== [RUN] Step 0: DONE =="
echo

echo "== [RUN] Step 1: Training PTv3 on S3DIS =="
bash /train/1-train.sh
echo "== [RUN] Step 1: DONE =="
echo

echo "== [RUN] All steps finished successfully. =="
