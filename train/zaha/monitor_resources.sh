#!/usr/bin/env bash
set -euo pipefail

# 1) Argumenty
LOG_DIR=${1:-/tmp}
INTERVAL=${2:-10}   # co ile sekund robić pomiar

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resource_usage_$(date +%Y%m%d_%H%M%S).log"

echo "Monitoring zasobów -> ${LOG_FILE}"
echo "Interwał: ${INTERVAL}s"

log_snapshot() {
  {
    echo "====== SNAPSHOT ($(date)) ======"

    echo
    echo "---- Memory (free -h) ----"
    free -h || echo "free not available"

    echo
    echo "---- Top processes by RAM (ps aux) ----"
    ps aux --sort=-%mem | head -n 10 || echo "ps not available"

    echo
    echo "---- GPU (nvidia-smi) ----"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi || echo "nvidia-smi failed"
    else
      echo "nvidia-smi not found"
    fi

    echo "==============================="
    echo
  } >> "${LOG_FILE}" 2>&1
}

# 2) Pętla monitorująca
while true; do
  log_snapshot
  sleep "${INTERVAL}"
done
