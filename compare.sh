#!/bin/bash

# Directory containing mismatched CNF files
INSTANCE_DIR="mismatches"

# Paths to the solvers
GANAK_SOLVER="/home/vjurisic/cse3000-how-to-break-a-solver/binaries/solvers/ganak"
GPMC_SOLVER="/home/vjurisic/cse3000-how-to-break-a-solver/binaries/solvers/gpmc"

# Ganak configuration
GANAK_CONFIG="--verb 0 --bveresolvmaxsz 12 --sbva 1 --maxcache 7800 --tdexpmult 1.1 --tdminw 7 --tdmaxw 60 --arjunoraclefindbins 6 --rdbclstarget 10000"

# Output files
GANAK_OUTPUT="ganak.txt"
GPMC_OUTPUT="gpmc.txt"

# Initialize output files
echo "Ganak Output" > "$GANAK_OUTPUT"
echo "============" >> "$GANAK_OUTPUT"
echo "GPMC Output" > "$GPMC_OUTPUT"
echo "============" >> "$GPMC_OUTPUT"

# Iterate through each CNF file in the mismatches folder
for file in "$INSTANCE_DIR"/*.cnf; do
    echo "Launching solvers for instance: $file"

    echo "Running ganak for $file..."
    $GANAK_SOLVER $GANAK_CONFIG "$file" >> "$GANAK_OUTPUT" 2>&1 || echo "Error running ganak for $file"

    echo "Running gpmc for $file..."
    $GPMC_SOLVER "$file" >> "$GPMC_OUTPUT" 2>&1 || echo "Error running gpmc for $file"

done

echo "All solvers launched. Outputs will be saved in $GANAK_OUTPUT and $GPMC_OUTPUT."
