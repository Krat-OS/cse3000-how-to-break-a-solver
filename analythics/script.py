import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <seed> [solver]")
        sys.exit(1)

    seed = sys.argv[1]
    solver = sys.argv[2] if len(sys.argv) > 2 else "all"

    print(f"Processing seed: {seed}")

    # Call sort.py with the seed
    subprocess.call(["python", "sort.py", seed])

    # Call plot_times.py with the seed and solver
    print(f"Plotting solver: {solver}")
    subprocess.call(["python", "plot_times.py", seed, solver])

if __name__ == "__main__":
    main()