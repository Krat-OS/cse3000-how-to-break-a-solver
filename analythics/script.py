import sys
import os
import subprocess

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <seed>")
        sys.exit(1)

    seed = sys.argv[1]
    print(f"Processing seed: {seed}")

    # Call sort.py with the seed
    subprocess.call(["python", "sort.py", seed])
    subprocess.call(["python", "plot_times.py", seed])

if __name__ == "__main__":
    main()