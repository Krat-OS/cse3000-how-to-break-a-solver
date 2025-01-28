#!/usr/bin/env python3

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Match


def get_seed_from_instance(instance_name: str) -> str:
    """
    Extract the numeric seed value from an instance name.

    Args:
        instance_name: Name of the instance file (e.g., 'PairSAT-easy-50_015_s10293847561249')

    Returns:
        The numeric seed value without 's' prefix (e.g., '10293847561249')
        Empty string if no seed is found

    Example:
        >>> get_seed_from_instance("PairSAT-easy-50_015_s10293847561249")
        "10293847561249"
    """
    match: Optional[Match[str]] = re.search(r's(\d+)', instance_name)
    return match.group(1) if match else ""


def parse_input_file(file_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse the input file containing instance mappings and base directory path.

    Args:
        file_path: Path to the input file

    Returns:
        Tuple containing:
            - Base directory path (str)
            - Dictionary mapping generator-difficulty keys to instance names

    Format:
        Each line before the last should be: "{generator}-{difficulty}: {instance_name}"
        Last line should be the absolute base directory path

    Example input file:
        PairSAT-easy: PairSAT-easy-50_015_s10293847561249
        PairSAT-hard: PairSAT-hard-50_036_s564738291012381
        /home/user/sat_instances

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines: list[str] = f.readlines()

        if not lines:
            raise ValueError("Input file is empty")

        base_path: str = lines[-1].strip()
        instances: Dict[str, str] = {}

        for line in lines[:-1]:
            if ':' not in line:
                continue
            key, value = map(str.strip, line.split(":", 1))
            instances[key] = value

        return base_path, instances

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing input file: {str(e)}")

def create_horn_output_directory(base_path: str) -> Path:
    """
    Create a single output directory for all HORN formula outputs.

    Args:
        base_path: Base directory where the HORN directory will be created

    Returns:
        Path to the created HORN output directory

    Raises:
        PermissionError: If directory creation fails due to permissions
        OSError: If directory creation fails for other reasons
    """
    horn_dir: Path = Path(base_path) / "HORN"
    try:
        horn_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {horn_dir}")
        return horn_dir
    except PermissionError:
        raise PermissionError(f"Permission denied creating directory: {horn_dir}")
    except OSError as e:
        raise OSError(f"Error creating directory {horn_dir}: {e}")

def run_horn_generator(base_path: str, instances: Dict[str, str], output_dir: Path) -> None:
    """
    Execute horn_gen.py to generate HORN formulas for each instance.

    Args:
        base_path: Base directory containing input instances
        instances: Dictionary mapping generator-difficulty keys to instance names
        output_dir: Path to the single HORN output directory

    Command format:
        python horn_gen.py --input={input_path} --output={output_dir} --seed={seed}

    Raises:
        subprocess.CalledProcessError: If horn_gen.py execution fails
        FileNotFoundError: If horn_gen.py or input files not found
    """
    script_dir: Path = Path(__file__).parent
    horn_gen_path: Path = script_dir.parent / "SharpVelvet" / "generators" / "horn_gen.py"

    if not horn_gen_path.exists():
        raise FileNotFoundError(f"horn_gen.py not found at: {horn_gen_path}")

    for instance_key, instance_name in instances.items():
        generator, difficulty = instance_key.split('-')
        seed: str = get_seed_from_instance(instance_name)
        
        input_path: Path = Path(base_path) / f"{generator}-{difficulty}" / f"{instance_name}.cnf"

        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue

        cmd: list[str] = [
            "python",
            str(horn_gen_path),
            f"--input={input_path}",
            f"--output={output_dir}",
            f"--seed={seed}"
        ]

        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {instance_name}:")
            print(f"Exit code: {e.returncode}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")

def rename_horn_files(output_dir: Path, instances: Dict[str, str]) -> None:
    """
    Rename generated HORN formula files to follow the standardized naming convention.

    Args:
        output_dir: Path to the single HORN output directory
        instances: Dictionary mapping generator-difficulty keys to instance names

    Naming convention:
        Original: {instance_name}_{idx}.cnf
        New: {generator}HORN{difficulty}-{randomness}_{idx}_{seed}.cnf

    Example:
        Original: PairSAT-easy-50_015_s10293847561249_000.cnf
        New: PairSATHORNeasy-50_000_s10293847561249.cnf

    Note:
        - Extracts randomness value from original instance name
        - Preserves index number from HORN generator output
        - Maintains seed identifier with 's' prefix
    """
    for instance_key, instance_name in instances.items():
        generator, difficulty = instance_key.split("-")

        seed_match: Optional[Match[str]] = re.search(r's(\d+)', instance_name)
        rand_match: Optional[Match[str]] = re.search(r'-(\d+)_', instance_name)

        if not seed_match or not rand_match:
            print(f"Warning: Could not extract seed or randomness from {instance_name}")
            continue

        seed: str = seed_match.group(0)
        randomness: str = rand_match.group(1)

        for file in output_dir.glob("*.cnf"):
            if instance_name in file.name:
                idx_match: Optional[Match[str]] = re.search(r'_(\d+)\.cnf$', file.name)
                if idx_match:
                    idx: str = idx_match.group(1)
                    new_name: str = f"{generator}HORN-{difficulty}-{randomness}_{idx}_{seed}.cnf"
                    new_path: Path = output_dir / new_name

                    try:
                        file.rename(new_path)
                        print(f"Renamed: {file.name} -> {new_name}")
                    except OSError as e:
                        print(f"Error renaming {file.name}: {e}")

def main() -> None:
    """
    Main entry point for the HORN formula generation script.

    Usage:
        python generate_with_horn.py input_file.txt

    Process:
        1. Parses input file for instance mappings and base path
        2. Creates a single HORN output directory
        3. Generates HORN formulas using horn_gen.py
        4. Renames output files to follow convention

    Exits with status code 1 if:
        - Incorrect number of arguments
        - Input file parsing fails
        - Critical errors during processing
    """
    if len(sys.argv) != 2:
        print("Usage: python generate_with_horn.py input_file.txt")
        sys.exit(1)

    try:
        input_file: str = sys.argv[1]
        base_path, instances = parse_input_file(input_file)
        
        print(f"Base path: {base_path}")
        print("Processing instances:", instances)

        output_dir = create_horn_output_directory(base_path)
        run_horn_generator(base_path, instances, output_dir)
        rename_horn_files(output_dir, instances)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
