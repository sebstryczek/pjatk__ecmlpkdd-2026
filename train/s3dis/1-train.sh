#!/usr/bin/env bash

# Exit immediately if any command exits with a non-zero status (fail fast).
set -e

# Magic bash timer start
SECONDS=0

echo "== [PTv3 / S3DIS] Training start =="

PROCESSED_S3DIS_DIR="/data/S3DIS/preprocess"
POINTCEPT_ROOT="/Pointcept"

# CONFIG_NAME="semseg-pt-v3m1-0-base"
# EXPERIMENT_NAME="semseg-pt-v3m1-0-base-exp"
CONFIG_NAME="semseg-pt-v3m1-0-base--local"
EXPERIMENT_NAME="semseg-pt-v3m1-0-base--local-exp"

echo "PROCESSED_S3DIS_DIR = $PROCESSED_S3DIS_DIR"
echo "POINTCEPT_ROOT      = $POINTCEPT_ROOT"
echo "CONFIG_NAME         = $CONFIG_NAME"
echo "EXPERIMENT_NAME     = $EXPERIMENT_NAME"
echo

# Check preprocessed data
if [ ! -d "$PROCESSED_S3DIS_DIR" ]; then
  echo "ERROR: Preprocessed S3DIS directory not found: $PROCESSED_S3DIS_DIR"
  echo "Run 0-preprocess.sh first."
  exit 1
fi


# Create symlink: /Pointcept/data/s3dis -> /data/S3DIS/preprocess
# data_root in config is "data/s3dis"
# https://github.com/Pointcept/Pointcept/blob/main/configs/s3dis/semseg-pt-v3m1-0-base.py#L69
POINTCEPT_DATA_DIR="${POINTCEPT_ROOT}/data"
POINTCEPT_S3DIS_LINK="${POINTCEPT_DATA_DIR}/s3dis"

# Ensure parent directory exists
if [ ! -d "$POINTCEPT_DATA_DIR" ]; then
  echo "Creating Pointcept data directory: $POINTCEPT_DATA_DIR"
  mkdir -p "$POINTCEPT_DATA_DIR"
fi

# Always reset link target to be safe / idempotent
echo "Resetting symlink: $POINTCEPT_S3DIS_LINK -> $PROCESSED_S3DIS_DIR"
rm -rf "$POINTCEPT_S3DIS_LINK"      # remove existing dir or symlink
ln -s "$PROCESSED_S3DIS_DIR" "$POINTCEPT_S3DIS_LINK"

cd "$POINTCEPT_ROOT"

# -g 1       - use 1 GPU (matches docker-compose)
# -d s3dis   - dataset
# -c ...     - config name (without .py)
# -n ...     - experiment name (output folder under exp/s3dis/)
sh scripts/train.sh -g 1 -d s3dis -c "${CONFIG_NAME}" -n "${EXPERIMENT_NAME}"

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
SECS=$((ELAPSED % 60))
printf "\nTotal training time: %02d:%02d:%02d (hh:mm:ss)\n" "$HOURS" "$MINS" "$SECS"

echo
echo "== [PTv3 / S3DIS] Training DONE =="
echo "Check logs and checkpoints under: ${POINTCEPT_ROOT}/exp/s3dis/${EXPERIMENT_NAME}"
echo
