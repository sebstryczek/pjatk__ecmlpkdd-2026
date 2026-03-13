#!/usr/bin/env bash

# Exit immediately if any command exits with a non-zero status (fail fast).
set -e

echo "== [S3DIS] Preprocessing start =="

# Path to the raw S3DIS dataset inside the container
RAW_S3DIS_DIR="/data/S3DIS/raw/Stanford3dDataset_v1.2"
# Path where preprocessed data will be written (also visible on the host)
PROCESSED_S3DIS_DIR="/data/S3DIS/preprocess"

echo "RAW_S3DIS_DIR       = $RAW_S3DIS_DIR"
echo "PROCESSED_S3DIS_DIR = $PROCESSED_S3DIS_DIR"
echo

# Sanity check: make sure raw dataset exists
if [ ! -d "$RAW_S3DIS_DIR" ]; then
  echo "ERROR: Raw S3DIS directory not found: $RAW_S3DIS_DIR"
  echo "Make sure your host folder $RAW_S3DIS_DIR is mounted correctly."
  exit 1
fi

# Skip preprocessing if the output directory already exists
if [ -d "$PROCESSED_S3DIS_DIR" ]; then
  echo "Preprocessed S3DIS directory already exists. Skipping preprocessing."
  exit 0
fi

# Fix the known S3DIS numeric corruption bug in Area_5/office_19/Annotations/ceiling_1.txt
echo "Running S3DIS ceiling_1.txt fix script (if needed)..."
python /train/0-fix_s3dis_ceiling_bug.py "$RAW_S3DIS_DIR"
echo "Fix script finished."
echo

mkdir -p "$PROCESSED_S3DIS_DIR"

cd /Pointcept
echo "Running Pointcept S3DIS preprocessing script..."
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
  --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6 \
  --dataset_root "$RAW_S3DIS_DIR" \
  --output_root "$PROCESSED_S3DIS_DIR" \
  --align_angle \
  --parse_normal \
  --num_workers 8

echo
echo "== [S3DIS] Preprocessing DONE =="
echo "Preprocessed files are in: $PROCESSED_S3DIS_DIR"
echo

# List a small part of the result to confirm something was written
ls -R "$PROCESSED_S3DIS_DIR" | head -n 80 || true
