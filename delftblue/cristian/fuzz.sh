#!/bin/bash
#SBATCH --job-name="fuzz-instances-cse3000-finding-different-ways-to-break-a-solver"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

################################################################################
# BASIC CONFIG
################################################################################

PROJECT_PATH="/home/$USER/cse3000-how-to-break-a-solver"

# Usage: sbatch fuzz.sh <SOLVER_TIMEOUT> <INSTANCES_PATH> <OUTPUT_DIR> <SOLVER_NAME1> [SOLVER_NAME2 SOLVER_NAME3 ...]
# Example:
#   sbatch fuzz.sh 60 \
#       /home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf \
#       /home/$USER/cse3000-how-to-break-a-solver/fuzz-results \
#       MySolver1 MySolver2

SOLVER_TIMEOUT="$1"
INSTANCES_PATH="$2"
OUTPUT_DIR="$3"
shift 3

# The remaining arguments are solver names
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch fuzz.sh <SOLVER_TIMEOUT> <INSTANCES_PATH> <OUTPUT_DIR> <SOLVER_NAME1> [SOLVER_NAME2 ...]"
  exit 1
fi

SOLVER_NAMES=("$@")  # Array of solver base names (no .json extension)

################################################################################
# GRACEFUL SHUTDOWN
################################################################################

declare -a PIDS_TO_CLEANUP=()

cleanup() {
    local signal="$1"
    echo "[fuzz.sh] Cleanup invoked..."

    # Terminate all tracked processes
    for (( idx=${#PIDS_TO_CLEANUP[@]}-1 ; idx>=0 ; idx-- )); do
        local pid="${PIDS_TO_CLEANUP[idx]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "[fuzz.sh] Terminating PID: $pid..."
            kill -TERM "$pid" 2>/dev/null
            for _ in {1..5}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[fuzz.sh] PID $pid still alive. Sending SIGKILL..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
    done

    echo "[fuzz.sh] Deactivate venv"
    deactivate 2>/dev/null || true

    echo "[fuzz.sh] Cleanup complete."
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
  echo "[fuzz.sh] Creating Python 3.11 venv..."
  python3.11 -m venv "$PROJECT_PATH/.venv" || {
    echo "Error: Unable to create venv"
    exit 1
  }
  source "$PROJECT_PATH/.venv/bin/activate"
  pip install -e "$PROJECT_PATH"
else
  echo "[fuzz.sh] Activating existing venv..."
  source "$PROJECT_PATH/.venv/bin/activate"
fi

################################################################################
# Build list of solver JSON paths
################################################################################

SOLVER_JSONS=()
for sname in "${SOLVER_NAMES[@]}"; do
    sjson="$PROJECT_PATH/SharpVelvet/${sname}.json"
    if [[ ! -f "$sjson" ]]; then
      echo "Error: solver JSON not found: $sjson"
      exit 1
    fi
    SOLVER_JSONS+=("$sjson")
done

################################################################################
# MAIN: CALL fuzz
################################################################################

echo "[fuzz.sh] Fuzzing with solvers: ${SOLVER_JSONS[*]}"
echo "[fuzz.sh] Instances path: $INSTANCES_PATH, solver timeout: $SOLVER_TIMEOUT"
echo "[fuzz.sh] Output directory: $OUTPUT_DIR"

run_with_tracking \
  "$PROJECT_PATH/global_cli.py" --use-slurm fuzz \
  --instances "$INSTANCES_PATH" \
  --solvers "${SOLVER_JSONS[@]}" \
  --solver-timeout "$SOLVER_TIMEOUT" \
  --out-dir "$OUTPUT_DIR"

echo "[fuzz.sh] Done."
