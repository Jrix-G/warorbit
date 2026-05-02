#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECK_INTERVAL="${CHECK_INTERVAL:-1800}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
LOG_DIR="${LOG_DIR:-neural_network/logs}"
BASE_CONFIG="neural_network/configs/default_config.json"
CURRENT_CONFIG="${BASE_CONFIG}"
TRAIN_LOG="${LOG_DIR}/notebook_4p_training.jsonl"
SUPERVISOR_LOG="${LOG_DIR}/supervisor.log"
TRAIN_STDOUT="${LOG_DIR}/training_supervised.out"
LOCK_FILE="${LOG_DIR}/supervisor.lock"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another supervisor instance is already running." >&2
  exit 1
fi

TRAIN_PID=""
RESTARTS=0

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_event() {
  echo "$(timestamp) $*" | tee -a "${SUPERVISOR_LOG}"
}

cleanup() {
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log_event "STOPPING training pid=${TRAIN_PID}"
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

launch_training() {
  log_event "START training config=${CURRENT_CONFIG}"
  python neural_network/scripts/run_notebook_4p_training.py \
    --config "${CURRENT_CONFIG}" >>"${TRAIN_STDOUT}" 2>&1 &
  TRAIN_PID=$!
  log_event "TRAINING_PID ${TRAIN_PID}"
}

run_health_check() {
  local output status
  set +e
  output="$(python neural_network/src/health_check.py --log "${TRAIN_LOG}" --window 20 2>&1)"
  status=$?
  set -e
  echo "${output}"
  return "${status}"
}

final_check() {
  local output status
  set +e
  output="$(python neural_network/src/health_check.py --log "${TRAIN_LOG}" --window 20 2>&1)"
  status=$?
  set -e
  log_event "FINAL_CHECK status=${status} metrics=${output}"
  echo "${output}"
}

set -e
log_event "SUPERVISOR_START interval=${CHECK_INTERVAL}s max_restarts=${MAX_RESTARTS}"
launch_training

while true; do
  sleep "${CHECK_INTERVAL}" &
  SLEEP_PID=$!
  while kill -0 "${SLEEP_PID}" 2>/dev/null; do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      wait "${TRAIN_PID}" || true
      log_event "TRAINING_EXITED pid=${TRAIN_PID}"
      final_check
      log_event "SUPERVISOR_END"
      exit 0
    fi
    sleep 1
  done
  wait "${SLEEP_PID}" || true

  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    wait "${TRAIN_PID}" || true
    log_event "TRAINING_EXITED pid=${TRAIN_PID}"
    final_check
    log_event "SUPERVISOR_END"
    exit 0
  fi

  set +e
  HEALTH_OUTPUT="$(run_health_check)"
  HEALTH_STATUS=$?
  set -e

  if [[ "${HEALTH_STATUS}" -eq 0 ]]; then
    log_event "CHECK_OK metrics=${HEALTH_OUTPUT}"
    continue
  fi

  if [[ "${HEALTH_STATUS}" -ne 1 ]]; then
    log_event "CHECK_ERROR status=${HEALTH_STATUS} output=${HEALTH_OUTPUT}"
    continue
  fi

  log_event "CHECK_UNHEALTHY metrics=${HEALTH_OUTPUT}"
  if [[ "${RESTARTS}" -ge "${MAX_RESTARTS}" ]]; then
    log_event "MAX_RESTARTS_REACHED restarts=${RESTARTS}"
    cleanup
    exit 2
  fi

  cleanup
  set +e
  AUTOCORRECT_OUTPUT="$(python neural_network/src/autocorrect.py \
    --config "${BASE_CONFIG}" \
    --log "${TRAIN_LOG}" 2>&1)"
  AUTOCORRECT_STATUS=$?
  set -e
  log_event "AUTOCORRECT status=${AUTOCORRECT_STATUS} output=${AUTOCORRECT_OUTPUT}"

  if [[ "${AUTOCORRECT_STATUS}" -eq 0 ]]; then
    CURRENT_CONFIG="neural_network/configs/autocorrected_config.json"
  fi
  RESTARTS=$((RESTARTS + 1))
  launch_training
done
