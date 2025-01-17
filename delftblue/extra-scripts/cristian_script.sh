#!/bin/bash
#SBATCH --job-name="cse3000-experiments"
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=compute-p2
#SBATCH --account=education-eemcs-courses-cse3000

# Global path variable
PROJECT_PATH="/home/$USER/cse3000-how-to-break-a-solver"

# Define the new output location
OUT_DIR="$PROJECT_PATH/../out"

# Record start time
START_TIME=$(date +%s)

# Array to store all PIDs that need cleanup
declare -a PIDS_TO_CLEANUP=()

# Enhanced cleanup function that handles signals
cleanup() {
    local signal=$1
    echo "Starting cleanup process..."

    # Kill processes in reverse order (children first)
    for ((idx=${#PIDS_TO_CLEANUP[@]}-1; idx>=0; idx--)); do
        pid=${PIDS_TO_CLEANUP[idx]}
        if kill -0 $pid 2>/dev/null; then
            echo "Terminating process (PID: $pid)..."
            kill -TERM $pid 2>/dev/null
            
            # Give each process up to 5 seconds to terminate gracefully
            for i in {1..5}; do
                if ! kill -0 $pid 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if kill -0 $pid 2>/dev/null; then
                echo "Process $pid still running, sending SIGKILL..."
                kill -9 $pid 2>/dev/null
            fi
        fi
    done

    echo "Copying output files to $OUT_DIR..."
    mkdir -p "$OUT_DIR"
    if [[ -d "$PROJECT_PATH/SharpVelvet/out" ]]; then
        cp -r "$PROJECT_PATH/SharpVelvet/out/." "$OUT_DIR"
        echo "Output files copied to $OUT_DIR."
        echo "Removing $PROJECT_PATH/SharpVelvet/out..."
        rm -rf "$PROJECT_PATH/SharpVelvet/out"
    else
        echo "No output directory found to copy."
    fi

    echo "Deactivating virtual environment."
    deactivate 2>/dev/null || true
    echo "Cleanup complete."

    # If we received a signal, exit with special code
    if [ -n "$signal" ]; then
        exit $((128 + signal))
    fi
}

# Function to run a command and track its PID
run_with_tracking() {
    "$@" &  # Run command in background
    local pid=$!
    PIDS_TO_CLEANUP+=($pid)
    wait $pid
    local return_code=$?
    # Remove PID from array after process completes
    PIDS_TO_CLEANUP=(${PIDS_TO_CLEANUP[@]/$pid})
    return $return_code
}

# Set up signal handling
trap 'cleanup 2' INT   # Ctrl+C (SIGINT)
trap 'cleanup 15' TERM # scancel (SIGTERM)
trap 'cleanup' EXIT    # Normal exit

# Check if .venv exists, create if not, then activate
if [[ ! -d "$PROJECT_PATH/.venv" ]]; then
    echo ".venv not found. Creating virtual environment with Python 3.11..."
    python3.11 -m venv "$PROJECT_PATH/.venv"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create virtual environment. Ensure Python 3.11 is installed."
        exit 1
    fi
    echo "Activating virtual environment and installing dependencies..."
    source "$PROJECT_PATH/.venv/bin/activate"
    pip install -e "$PROJECT_PATH"
else
    echo "Activating existing virtual environment..."
    source "$PROJECT_PATH/.venv/bin/activate"
fi

# Parse arguments
GENERATORS=()
SOLVERS=()
SOLVER_TIMEOUT=()
NUM_ITER=()
TIME_LIMIT=()
MAX_WORKERS=()
THREAD_TIMEOUT=()
BATCH_SIZE=()
MEMORY_LIMIT_GB=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --generators) shift; while [[ "$#" -gt 0 && "$1" != --* ]]; do GENERATORS+=("$1"); shift; done ;;
        --num-iter) shift; NUM_ITER=$1; shift ;;
        --solvers) shift; while [[ "$#" -gt 0 && "$1" != --* ]]; do SOLVERS+=("$1"); shift; done ;;
        --solver-timeout) shift; SOLVER_TIMEOUT=$1; shift;;
        --time) shift; TIME_LIMIT=$1; shift ;;
        --max-workers) shift; MAX_WORKERS=$1; shift ;;
        --thread-timeout) shift; THREAD_TIMEOUT=$1; shift ;;
        --batch-size) shift; BATCH_SIZE=$1; shift ;;
        --memory-limit-gb) shift; MEMORY_LIMIT_GB=$1; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Validate time limit format and convert to seconds
if [[ -z "$TIME_LIMIT" ]]; then
    echo "Error: Time limit not specified. Use --time HH:MM:SS"
    exit 1
fi

if ! [[ $TIME_LIMIT =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Error: Time limit must be in format HH:MM:SS"
    exit 1
fi

TOTAL_TIME_LIMIT=$(echo "$TIME_LIMIT" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')

# Ensure at least one generator and one solver are provided
if [[ ${#GENERATORS[@]} -eq 0 ]]; then
    echo "Error: No generators specified. Use --generators to provide generator names."
    exit 1
fi

if [[ ${#SOLVERS[@]} -eq 0 ]]; then
    echo "Error: No solvers specified. Use --solvers to provide solver names."
    exit 1
fi

# Run each python command with proper tracking and cleanup
for GENERATOR in "${GENERATORS[@]}"; do
    GENERATOR_PATH="$PROJECT_PATH/SharpVelvet/$GENERATOR.json"
    if [[ -f $GENERATOR_PATH ]]; then
        echo "Running generate_instances.py with generator: $GENERATOR_PATH"
        run_with_tracking python "$PROJECT_PATH/SharpVelvet/src/generate_instances.py" \
            --generators "$GENERATOR_PATH" \
            --num-iter "$NUM_ITER"
    else
        echo "Warning: Generator file $GENERATOR_PATH not found. Skipping."
    fi
done

for SOLVER in "${SOLVERS[@]}"; do
    SOLVER_PATH="$PROJECT_PATH/SharpVelvet/$SOLVER.json"
    if [[ -f $SOLVER_PATH ]]; then
        echo "Running run_fuzzer.py with solver: $SOLVER_PATH"
        run_with_tracking python "$PROJECT_PATH/SharpVelvet/src/run_fuzzer.py" \
            --counters "$SOLVER_PATH" \
            --instances "$PROJECT_PATH/SharpVelvet/out/instances/cnf" \
            --timeout "$SOLVER_TIMEOUT"
    else
        echo "Warning: Solver file $SOLVER_PATH not found. Skipping."
    fi
done

# Process CSV files
CSV_PATH="$PROJECT_PATH/SharpVelvet/out"
CSV_FILES=("$CSV_PATH"/*.csv)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "No CSV files found in $CSV_PATH. Skipping CPOG verifier."
else
    for CSV in "${CSV_FILES[@]}"; do
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
        
        if [ $ELAPSED_TIME -ge $TOTAL_TIME_LIMIT ]; then
            echo "Warning: Total time limit reached. Skipping remaining CPOG verification."
            break
        fi

        echo "Running cpog_verifier with CSV file: $CSV"
        run_with_tracking python "$PROJECT_PATH/cpog_verifier/cli.py" \
            --csv-path "$CSV" \
            --max-workers "$MAX_WORKERS" \
            --thread-timeout "$THREAD_TIMEOUT" \
            --batch-size "$BATCH_SIZE" \
            --memory-limit-gb "$MEMORY_LIMIT_GB"
    done
fi

# Extract satzilla features
if [[ -d "$PROJECT_PATH/SharpVelvet/out/instances/cnf" ]]; then
    echo "Extracting Satzilla features from CNF instances..."
    python "$PROJECT_PATH/satzilla_feature_extractor/compute_sat_feature_data_cli.py" compute_features \
        "$PROJECT_PATH/SharpVelvet/out/instances/cnf" \
        "$PROJECT_PATH/SharpVelvet/out" \
        "$PROJECT_PATH/satzilla_feature_extractor/binaries/features"
    
    # Check if the feature extraction succeeded
    if [[ $? -eq 0 ]]; then
        echo "Satzilla feature extraction completed successfully."
    else
        echo "Error: Satzilla feature extraction failed. Please check the logs for more details."
        exit 1
    fi
else
    echo "Error: CNF instances directory not found. Unable to extract Satzilla features."
    exit 1
fi
