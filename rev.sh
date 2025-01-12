#!/bin/bash

# Directory containing the files
dir="/home/$USER/cse3000-how-to-break-a-solver/SharpVelvet/instances/500_horn_instances/chevu"

# Ensure the directory exists
if [ ! -d "$dir" ]; then
    echo "Directory $dir does not exist."
    exit 1
fi

# Reverse the order of files and update their timestamps
files=($(ls -1 "$dir" | sort -n -t_ -k2 | tac))  # Reverse the sorted order
for file in "${files[@]}"; do
    touch "$dir/$file"  # Update modification time
    sleep 0.1           # Add delay to ensure distinct timestamps
done

echo "Files reordered based on modification time."

