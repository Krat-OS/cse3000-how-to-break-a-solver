#!/bin/bash
#SBATCH --job-name="cpog-verify-instances-cse3000-finding-different-ways-to-break-a-solver"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=65536
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

################################################################################
# CONFIG
################################################################################

PROJECT_PATH="/home/$USER/cse3000-how-to-break-a-solver"

# Minimal user arguments:
#   1) CSV path (relative or absolute)
#   2) MAX_WORKERS
#   3) THREAD_TIMEOUT
#   4) BATCH_SIZE
#   5) MEMORY_LIMIT_GB
CSV_FILE="$1"
MAX_WORKERS="$2"
THREAD_TIMEOUT="$3"
BATCH_SIZE="$4"
MEMORY_LIMIT_GB="$5"

if [[ -z "$CSV_FILE" || -z "$MAX_WORKERS" || -z "$THREAD_TIMEOUT" \
      || -z "$BATCH_SIZE" || -z "$MEMORY_LIMIT_GB" ]]; then
  echo "Usage: sbatch cpog_verify.sh <CSV_FILE> <MAX_WORKERS> <THREAD_TIMEOUT> <BATCH_SIZE> <MEMORY_LIMIT_GB>"
  echo "Example: sbatch cpog_verify.sh /path/to/fuzz-results.csv 4 120 10 8"
  exit 1
fi

################################################################################
# GRACEFUL SHUTDOWN
################################################################################

declare -a PIDS_TO_CLEANUP=()

cleanup() {
    local signal="$1"
    echo "[cpog_verify.sh] Cleanup invoked..."

    for (( idx=${#PIDS_TO_CLEANUP[@]}-1 ; idx>=0 ; idx-- )) ; do
        local pid="${PIDS_TO_CLEANUP[idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[cpog_verify.sh] Terminating PID: $pid..."
            kill -TERM "$pid" 2>/dev/null
            for _ in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[cpog_verify.sh] PID $pid still alive. Sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done

    echo "[cpog_verify.sh] Deactivate venv"
    deactivate 2>/dev/null || true

    echo "[cpog_verify.sh] Cleanup complete."
    if [[ -n "$signal" ]]; then
        exit $((128 + signal))
    fi
}

run_with_tracking() {
    "$@" &
    local pid=$!
    PIDS_TO_CLEANUP+=("$pid")
    wait "$pid"
    local rc=$?
    PIDS_TO_CLEANUP=("${PIDS_TO_CLEANUP[@]/$pid}")
    return $rc
}

trap 'cleanup 2' INT
trap 'cleanup 15' TERM
trap 'cleanup' EXIT

################################################################################
# VENV
################################################################################

if [[ ! -d "$PROJECT_PATH/.venv" ]]; then
  echo "[cpog_verify.sh] Creating Python 3.11 venv..."
  python3.11 -m venv "$PROJECT_PATH/.venv" || {
    echo "Error: Unable to create venv"
    exit 1
  }
  source "$PROJECT_PATH/.venv/bin/activate"
  pip install -e "$PROJECT_PATH"
else
  source "$PROJECT_PATH/.venv/bin/activate"
fi

################################################################################
# MAIN
################################################################################

echo "[cpog_verify.sh] Verifying CSV: $CSV_FILE"

run_with_tracking \
  python "$PROJECT_PATH/global_cli.py" cpog_verify \
  --csv-path "$CSV_FILE" \
  --max-workers "$MAX_WORKERS" \
  --thread-timeout "$THREAD_TIMEOUT" \
  --batch-size "$BATCH_SIZE" \
  --memory-limit-gb "$MEMORY_LIMIT_GB" \
  --output-dir "$(dirname "$CSV_FILE")" \
  --verifier-dir "$PROJECT_PATH/cpog_verifier/cpog" \

if [[ -f "$OUTPUT_FILE" ]]; then
  echo "[cpog_verify.sh] Verification complete. Output saved to $OUTPUT_FILE"
else
  echo "[cpog_verify.sh] Verification failed or output not generated."
  exit 1
fi

echo "[cpog_verify.sh] Done."