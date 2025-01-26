import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <seed> [solver] [--xmin XMIN] [--xmax XMAX] [--ymin YMIN] [--ymax YMAX] [--y2min Y2MIN] [--y2max Y2MAX]")
        sys.exit(1)

    seed = sys.argv[1]
    
    # Check if a solver is provided or additional arguments start immediately
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        solver = sys.argv[2]
        additional_args = sys.argv[3:]  # Everything after solver
    else:
        solver = "all"  # Default solver
        additional_args = sys.argv[2:]  # Everything else

    # Call sort.py with the seed
    # subprocess.call(["python", "sort.py", seed])

    # Call plot_times.py with the seed, solver, and additional arguments
    print(f"Plotting solver: {solver}")
    subprocess.call(["python", "plot_times.py", seed, solver] + additional_args)

if __name__ == "__main__":
    main()