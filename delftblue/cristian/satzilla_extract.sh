#!/bin/bash
#SBATCH --job-name="satzilla-extract-features-cse3000-finding-different-ways-to-break-a-solver"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

################################################################################
# CONFIG
################################################################################

PROJECT_PATH="/home/$USER/cse3000-how-to-break-a-solver"

# Minimal user arguments:
#   1) INSTANCES_DIR  (CNF folder)
INSTANCES_DIR="$1"

if [[ -z "$INSTANCES_DIR" ]]; then
  echo "Usage: sbatch satzilla_extract.sh <INSTANCES_DIR>"
  echo "Example: sbatch satzilla_extract.sh $PROJECT_PATH/SharpVelvet/out/instances/cnf"
  exit 1
fi

OUT_DIR=$(dirname $(dirname "$INSTANCES_DIR"))
BINARY_PATH="$PROJECT_PATH/satzilla_feature_extractor/binaries/features"

################################################################################
# GRACEFUL SHUTDOWN
################################################################################

declare -a PIDS_TO_CLEANUP=()

cleanup() {
    local signal="$1"
    echo "[satzilla_extract.sh] Cleanup invoked..."

    for (( idx=${#PIDS_TO_CLEANUP[@]}-1 ; idx>=0 ; idx-- )) ; do
        local pid="${PIDS_TO_CLEANUP[idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[satzilla_extract.sh] Terminating PID: $pid..."
            kill -TERM "$pid" 2>/dev/null
            for _ in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[satzilla_extract.sh] PID $pid still alive. Sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done

    echo "[satzilla_extract.sh] Deactivate venv"
    deactivate 2>/dev/null || true

    echo "[satzilla_extract.sh] Cleanup complete."
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
  echo "[satzilla_extract.sh] Creating Python 3.11 venv..."
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

echo "[satzilla_extract.sh] Extracting Satzilla features from $INSTANCES_DIR => $OUT_DIR"

run_with_tracking \
  python "$PROJECT_PATH/global_cli.py" satzilla_extract \
  --instances "$INSTANCES_DIR" \
  --out-dir "$OUT_DIR" \
  --satzilla-binary-path "$BINARY_PATH"

echo "[satzilla_extract.sh] Done."
