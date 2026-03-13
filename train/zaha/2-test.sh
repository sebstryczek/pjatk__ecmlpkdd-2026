#!/usr/bin/env bash

set -e

echo "== Test start =="

PROCESSED_ZAHA_DIR="/data/ZAHA/preprocessed"
POINTCEPT_ROOT="/Pointcept"

CONFIG_NAME="semseg-pt-v3m1-0-zaha"
EXPERIMENT_NAME="${CONFIG_NAME}-exp"

WEIGHT_NAME="model_best"
WEIGHT_FULL="/Pointcept/exp/zaha/${EXPERIMENT_NAME}/model/${WEIGHT_NAME}.pth"


echo "CONFIG_NAME     = ${CONFIG_NAME}"
echo "EXPERIMENT_NAME = ${EXPERIMENT_NAME}"
echo "WEIGHT          = ${WEIGHT}"
echo

# 1) Sprawdzenie preprocessed ZAHA
if [ ! -d "${PROCESSED_ZAHA_DIR}" ]; then
  echo "ERROR: Preprocessed ZAHA directory not found: ${PROCESSED_ZAHA_DIR}"
  exit 1
fi

# 2) Symlink
POINTCEPT_DATA_DIR="${POINTCEPT_ROOT}/data"
POINTCEPT_ZAHA_LINK="${POINTCEPT_DATA_DIR}/zaha"
mkdir -p "${POINTCEPT_DATA_DIR}"
rm -rf "${POINTCEPT_ZAHA_LINK}"
ln -s "${PROCESSED_ZAHA_DIR}" "${POINTCEPT_ZAHA_LINK}"

cd "${POINTCEPT_ROOT}"

if [ ! -f "${WEIGHT_FULL}" ]; then
  echo "ERROR: Checkpoint not found: ${WEIGHT_FULL}"
  ls -la "/Pointcept/exp/zaha/${EXPERIMENT_NAME}/model/" 2>/dev/null || echo "  (brak katalogu)"
  exit 1
fi

sh scripts/test.sh -g 1 -d zaha -c "${CONFIG_NAME}" -n "${EXPERIMENT_NAME}" \
    -w "${WEIGHT_NAME}"

echo
echo "== Test DONE =="