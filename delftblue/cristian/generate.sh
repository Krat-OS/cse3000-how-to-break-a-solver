#!/bin/bash
#SBATCH --job-name="generate-instances-cse3000-finding-different-ways-to-break-a-solver"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

################################################################################
# BASIC CONFIG
################################################################################

# Hard-coded path to your project
PROJECT_PATH="/home/$USER/cse3000-how-to-break-a-solver"

# Below: parse minimal user arguments from command line:
#   1) Generator name (without .json)
#   2) Input seeds file path
#   3) num-iter (integer)
#   4) output directory
GENERATOR_NAME="$1"
INPUT_SEEDS="$2"
NUM_ITER="$3"
OUT_DIR="$4"

if [[ -z "$GENERATOR_NAME" || -z "$INPUT_SEEDS" || -z "$NUM_ITER" || -z "$OUT_DIR" ]]; then
  echo "Usage: sbatch generate.sh <GENERATOR_NAME> <INPUT_SEEDS> <NUM_ITER> <OUT_DIR>"
  echo "Example: sbatch generate.sh MyGenerator input.txt 10 /home/user/out"
  exit 1
fi

################################################################################
# GRACEFUL SHUTDOWN SETUP
################################################################################

declare -a PIDS_TO_CLEANUP=()

cleanup() {
    local signal="$1"
    echo "[generate.sh] Cleanup invoked..."

    # Kill tracked processes in reverse order
    for (( idx=${#PIDS_TO_CLEANUP[@]}-1 ; idx>=0 ; idx-- )) ; do
        local pid="${PIDS_TO_CLEANUP[idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[generate.sh] Terminating PID: $pid..."
            kill -TERM "$pid" 2>/dev/null
            for _ in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[generate.sh] PID $pid still alive. Sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done

    echo "[generate.sh] Deactivate venv"
    deactivate 2>/dev/null || true

    echo "[generate.sh] Cleanup complete."
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
# VENV SETUP
################################################################################

if [[ ! -d "$PROJECT_PATH/.venv" ]]; then
  echo "[generate.sh] Creating Python 3.11 venv at $PROJECT_PATH/.venv"
  python3.11 -m venv "$PROJECT_PATH/.venv" || {
    echo "Error: Unable to create venv with python3.11"
    exit 1
  }
  source "$PROJECT_PATH/.venv/bin/activate"
  pip install -e "$PROJECT_PATH"
else
  echo "[generate.sh] Activating existing venv at $PROJECT_PATH/.venv"
  source "$PROJECT_PATH/.venv/bin/activate"
fi

################################################################################
# MAIN: CALL GLOBAL_CLI.PY generate
################################################################################

GENERATOR_JSON="$PROJECT_PATH/SharpVelvet/$GENERATOR_NAME.json"

if [[ ! -f "$GENERATOR_JSON" ]]; then
  echo "Error: generator JSON not found at $GENERATOR_JSON"
  exit 1
fi

if [[ ! -f "$INPUT_SEEDS" ]]; then
  echo "Error: Input seeds file not found at $INPUT_SEEDS"
  exit 1
fi

echo "[generate.sh] Using generator JSON: $GENERATOR_JSON"
echo "[generate.sh] Using input seeds file: $INPUT_SEEDS"

run_with_tracking \
"$PROJECT_PATH/global_cli.py" --use-slurm generate \
  --generators "$GENERATOR_JSON" \
  --input-seeds "$INPUT_SEEDS" \
  --num-iter "$NUM_ITER" \
  --out-dir "$OUT_DIR"

echo "[generate.sh] Done."
