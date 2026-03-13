#!/usr/bin/env bash

# Fail fast
set -e

echo "== Training start =="

PROCESSED_ZAHA_DIR="/data/ZAHA/preprocessed"
POINTCEPT_ROOT="/Pointcept"

# CONFIG_NAME="semseg-pt-v1-zaha"
CONFIG_NAME="semseg-pt-v3m1-0-zaha"
EXPERIMENT_NAME="${CONFIG_NAME}-exp"

echo "PROCESSED_ZAHA_DIR = ${PROCESSED_ZAHA_DIR}"
echo "POINTCEPT_ROOT      = ${POINTCEPT_ROOT}"
echo "CONFIG_NAME         = ${CONFIG_NAME}"
echo "EXPERIMENT_NAME     = ${EXPERIMENT_NAME}"
echo

          # RESOURCES_LOGS_DIR="${POINTCEPT_ROOT}/exp/zaha/${EXPERIMENT_NAME}/resources_logs"
          # mkdir -p "${RESOURCES_LOGS_DIR}"
          # bash train/zaha/monitor_resources.sh "${RESOURCES_LOGS_DIR}" 10 &
          # MONITOR_PID=$!
          # echo "Monitor zasobów wystartował (PID=${MONITOR_PID}), log: ${RESOURCES_LOGS_DIR}"

# 1) Sprawdzenie preprocessed ZAHA
if [ ! -d "${PROCESSED_ZAHA_DIR}" ]; then
  echo "ERROR: Preprocessed ZAHA directory not found: ${PROCESSED_ZAHA_DIR}"
  echo "Uruchom najpierw preprocessing (0-preprocess.sh / train/zaha/run.sh)."
  exit 1
fi

# 2) Symlink: /Pointcept/data/zaha -> /data/ZAHA/preprocessed
POINTCEPT_DATA_DIR="${POINTCEPT_ROOT}/data"
POINTCEPT_ZAHA_LINK="${POINTCEPT_DATA_DIR}/zaha"

if [ ! -d "${POINTCEPT_DATA_DIR}" ]; then
  echo "Creating Pointcept data directory: ${POINTCEPT_DATA_DIR}"
  mkdir -p "${POINTCEPT_DATA_DIR}"
fi

echo "Resetting symlink: ${POINTCEPT_ZAHA_LINK} -> ${PROCESSED_ZAHA_DIR}"
rm -rf "${POINTCEPT_ZAHA_LINK}"      # usuń istniejący katalog / symlink
ln -s "${PROCESSED_ZAHA_DIR}" "${POINTCEPT_ZAHA_LINK}"

cd "${POINTCEPT_ROOT}"

# 3) Trening
# -g 1      : 1 GPU
# -d zaha   : dataset → configs/zaha/${CONFIG_NAME}.py
# -c ...    : nazwa configu (bez .py)
# -n ...    : nazwa eksperymentu (exp/zaha/${EXPERIMENT_NAME})
start_time=$(date +%s)

sh scripts/train.sh -g 1 -d zaha -c "${CONFIG_NAME}" -n "${EXPERIMENT_NAME}"

STATUS=$?  # zapamiętujemy kod wyjścia treningu

# "$!" → PID ostatniego procesu uruchomionego w tle w tej samej powłoce
# "$?" → kod wyjścia ostatniego polecenia / potoku, które się zakończyło
# Zatrzymanie monitora (jeśli jeszcze żyje)
          # if ps -p "${MONITOR_PID}" > /dev/null 2>&1; then
          #   kill "${MONITOR_PID}" >/dev/null 2>&1 || true
          # fi

exit "${STATUS}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo
printf "== Training DONE ==\n"
printf "Total training time: %02d:%02d:%02d (hh:mm:ss)\n" "${hours}" "${minutes}" "${seconds}"
echo "Check logs and checkpoints under: ${POINTCEPT_ROOT}/exp/zaha/${EXPERIMENT_NAME}"
echo
