#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

def parse_input_file(file_path: str) -> Tuple[str, Dict[str, str]]:
    """Parse input file and extract base path and instance mappings.
    
    Args:
        file_path: Path to the input file containing instance mappings and base path.
        
    Returns:
        Tuple containing:
            - base_path: String path to the base directory
            - instances: Dictionary mapping generator-difficulty keys to instance names
            
    Format:
        Input file should have instance mappings (key: value) followed by base path on last line
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    base_path = lines[-1].strip()
    instances: Dict[str, str] = {}
    for line in lines[:-1]:
        if ':' in line:
            key, value = line.strip().split(': ')
            instances[key] = value
    
    return base_path, instances

def create_horn_directories(base_path: str, instances: Dict[str, str]) -> None:
    """Create HORN directories for each generator-difficulty combination.
    
    Args:
        base_path: Base directory path where HORN directories will be created
        instances: Dictionary of generator-difficulty keys and their instance names
    """
    for instance_key in instances:
        generator, difficulty = instance_key.split('-')
        horn_dir = Path(base_path) / f"{generator}-HORN-{difficulty}"
        horn_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {horn_dir}")

def run_horn_generator(base_path: str, instances: Dict[str, str]) -> None:
    """Run horn_gen.py for each instance to generate HORN formulas.
    
    Args:
        base_path: Base directory containing input instances
        instances: Dictionary mapping generator-difficulty keys to instance names
        
    Raises:
        subprocess.CalledProcessError: If horn_gen.py execution fails
    """
    script_dir = Path(__file__).parent
    horn_gen_path = script_dir.parent / "SharpVelvet" / "generators" / "horn_gen.py"
    
    for instance_key, instance_name in instances.items():
        generator, difficulty = instance_key.split('-')
        
        input_path = Path(base_path) / f"{generator}-{difficulty}" / f"{instance_name}.cnf"
        output_dir = Path(base_path) / f"{generator}-HORN-{difficulty}"
        
        cmd = [
            "python",
            str(horn_gen_path),
            f"--input={input_path}",
            f"--output={output_dir}"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {instance_name}: {e}")

def rename_horn_files(base_path: str, instances: Dict[str, str]) -> None:
    """Rename generated HORN files to follow the naming convention.
    
    Args:
        base_path: Base directory containing HORN directories
        instances: Dictionary mapping generator-difficulty keys to instance names
        
    Format:
        New filename pattern: {generator}-HORN-{difficulty}-0_{idx}_{seed}.cnf
    """
    for instance_key, instance_name in instances.items():
        generator, difficulty = instance_key.split('-')
        horn_dir = Path(base_path) / f"{generator}-HORN-{difficulty}"
        
        match = re.search(r's\d+', instance_name)
        if not match:
            continue
        seed = match.group(0)
        
        for file in horn_dir.glob("*.cnf"):
            if instance_name in file.name:
                idx_match = re.search(r'_(\d+)\.cnf$', file.name)
                if idx_match:
                    idx = idx_match.group(1)
                    new_name = f"{generator}-HORN-{difficulty}-0_{idx}_{seed}.cnf"
                    new_path = horn_dir / new_name
                    file.rename(new_path)
                    print(f"Renamed: {file.name} -> {new_name}")

def main() -> None:
    """Main entry point for the HORN formula generation script.
    
    Usage:
        python generate_with_horn.py input_file.txt
        
    The input file should contain instance mappings and a base path.
    """
    if len(sys.argv) != 2:
        print("Usage: python generate_with_horn.py input_file.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    base_path, instances = parse_input_file(input_file)
    print(f"Base path: {base_path}")
    print("Instances:", instances)
    
    create_horn_directories(base_path, instances)
    run_horn_generator(base_path, instances)
    rename_horn_files(base_path, instances)

if __name__ == "__main__":
    main()
