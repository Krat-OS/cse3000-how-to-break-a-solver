#!/usr/bin/env python3
"""
A unified CLI tool with 4 subcommands:
  1) generate
  2) fuzz
  3) cpog_verify
  4) satzilla_extract

It references:
  - cpog_verifier.utils (verify_with_cpog, verify_single_instance, etc.)
  - satzilla_feature_extractor.compute_sat_feature_data (compute_features, etc.)

Usage:
    global_cli.py <subcommand> [args...]

Example:
    ./global_cli.py generate --help
    ./global_cli.py fuzz --help
    ./global_cli.py cpog_verify --help
    ./global_cli.py satzilla_extract --help
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import pandas as pd

from cpog_verifier.cli import process_results_and_verify_with_cpog, verify_single_cnf, get_output_path
from satzilla_feature_extractor.compute_sat_feature_data import compute_features, process_csv_files

###############################################################################
# Logging Setup
###############################################################################

class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to show levelname only for WARNING or ERROR,
    and add color to the output.
    """
    RESET = "\033[0m"
    COLORS = {
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            colored_level = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        else:
            colored_level = record.levelname

        # Format: [GLOBAL CLI <LEVEL>], 2024-01-01 12:34:56: message
        return (f"[GLOBAL CLI {colored_level}], "
                f"{self.formatTime(record)}: "
                f"{record.getMessage()}")

def setup_logger() -> None:
    """
    Set up the global logger with a custom formatter.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logger.handlers = [handler]

setup_logger()
logger = logging.getLogger(__name__)

# Global signal handling
should_terminate = False

def signal_handler(signum, frame) -> None:
    global should_terminate
    should_terminate = True
    logger.warning(f"Received signal {signum}. Terminating processes.")

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

###############################################################################
# Subcommand: generate
###############################################################################

def add_generate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'generate' subcommand parser to the top-level subparsers.
    """
    parser_gen = subparsers.add_parser(
        "generate",
        help="Generate CNF instances using SharpVelvet's generate_instances.py."
    )
    parser_gen.add_argument(
        "--input-seeds",
        type=str,
        required=True,
        help="Path to a text file containing a list of seeds."
    )
    parser_gen.add_argument(
        "--generators",
        nargs="+",
        required=True,
        help="Paths to generator JSON files."
    )
    parser_gen.add_argument(
        "--num-iter",
        type=int,
        required=True,
        help="Number of iterations per generator."
    )
    parser_gen.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the final output directory."
    )
    parser_gen.set_defaults(func=command_generate)

def command_generate(args: argparse.Namespace) -> None:
    """
    Execute the 'generate' subcommand:
      1) Reads seeds from the input file.
      2) Runs generate_instances.py as a script for each seed.
    """
    if not os.path.isfile(args.input_seeds):
        logger.error(f"Input seeds file not found: {args.input_seeds}")
        sys.exit(1)

    with open(args.input_seeds, "r") as f:
        seeds: List[str] = [line.strip() for line in f if line.strip().isdigit()]

    if not seeds or not all(seed.isdigit() for seed in seeds):
        logger.error("No valid seeds found in the input seeds file.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    sharpvelvet_dir = os.path.dirname(os.path.abspath(__file__))
    generate_instances_script = os.path.join(sharpvelvet_dir, "SharpVelvet/src/generate_instances.py")

    def run_generation(seed: str):
        try:
            logger.info(f"Generating instances for seed {seed}...")
            cmd = [
                sys.executable,
                generate_instances_script,
                "--generators", *args.generators,
                "--num-iter", str(args.num_iter),
                "--seed", seed,
                "--out-dir", args.out_dir,
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully generated instances for seed {seed}.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating instances for seed {seed}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for seed {seed}: {e}")

    # Run generation for each seed
    for seed in seeds:
        run_generation(seed)

    logger.info("Instance generation complete.")


###############################################################################
# Subcommand: fuzz
###############################################################################

def add_fuzz_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'fuzz' subcommand parser.
    """
    parser_fz = subparsers.add_parser(
        "fuzz",
        help="Fuzz CNF instances using SharpVelvet's run_fuzzer.py with multiple solvers."
    )
    parser_fz.add_argument(
        "--instances",
        type=str,
        required=True,
        help="Path to the folder containing CNF instances."
    )
    parser_fz.add_argument(
        "--solvers",
        nargs="+",
        required=True,
        help="List of solver JSON file paths."
    )
    parser_fz.add_argument(
        "--solver-timeout",
        type=int,
        required=True,
        help="Timeout in seconds for each solver."
    )
    parser_fz.add_argument(
        "--sharpvelvet-fuzzer",
        type=str,
        required=True,
        help="Path to SharpVelvet's run_fuzzer.py."
    )
    parser_fz.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the final output directory."
    )
    parser_fz.set_defaults(func=command_fuzz)

def command_fuzz(args: argparse.Namespace) -> None:
    def create_temp_dir() -> str:
        return tempfile.mkdtemp(prefix="fuzz_out_")

    def run_fuzzer(solver_path: str, out_dir: str) -> None:
        if not os.path.isfile(solver_path):
            logger.warning(f"Solver file not found: {solver_path}")
            return

        cmd = [
            sys.executable,
            args.sharpvelvet_fuzzer,
            "--verbosity", "1",
            "--counters", solver_path,
            "--instances", args.instances,
            "--timeout", str(args.solver_timeout),
            "--out-dir", out_dir,
        ]
        try:
            logger.info(f"Running fuzzer: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running fuzzer for solver {solver_path}: {e}")

    temp_out_dirs = []
    with ThreadPoolExecutor(max_workers=len(args.solvers)) as executor:
        futures = []
        for solver_path in args.solvers:
            out_dir = create_temp_dir()
            temp_out_dirs.append(out_dir)
            futures.append(executor.submit(run_fuzzer, solver_path, out_dir))

        for future in as_completed(futures):
            future.result()

    csv_paths: List[str] = []
    for temp_out_dir in temp_out_dirs:
        for item in os.listdir(temp_out_dir):
            src = os.path.join(temp_out_dir, item)
            dst = os.path.join(args.out_dir, item)
            if item.endswith(".csv"):
                csv_paths.append(src)

    if csv_paths:
        final_csv_name = Path(csv_paths[0]).name
        final_csv_path = os.path.join(args.out_dir, final_csv_name)
        merge_csv_files(csv_paths, final_csv_path)
        logger.info(f"Final merged CSV saved to: {final_csv_path}")

    for out_dir in temp_out_dirs:
        shutil.rmtree(out_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary output directory: {out_dir}")

    logger.info(f"All outputs saved to the specified directory: {args.out_dir}")

def merge_csv_files(csv_paths: List[str], out_csv: str) -> None:
    """
    Merge multiple CSV files into out_csv.
    """
    valid_paths = [p for p in csv_paths if os.path.isfile(p)]
    if not valid_paths:
        logger.warning("No valid CSVs to merge.")
        return

    dfs = [pd.read_csv(p) for p in valid_paths]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(out_csv, index=False)
    logger.info(f"Merged CSV file created at {out_csv}")

###############################################################################
# Subcommand: cpog_verify
###############################################################################

def add_cpog_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'cpog_verify' subcommand parser.
    """
    parser_cpog = subparsers.add_parser(
        "cpog_verify",
        help="Run the CPOG verifier on fuzz-results CSV or a single CNF."
    )
    parser_cpog.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file containing verification instances."
    )
    parser_cpog.add_argument(
        "--cnf-path",
        type=str,
        help="Path to a single CNF file for verification."
    )
    parser_cpog.add_argument(
        "--verifier-dir",
        type=Path,
        default=Path(__file__).parent / "cpog",
        help="Path to directory containing verifier binaries."
    )
    parser_cpog.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save output CPOG files or CSV results."
    )
    parser_cpog.add_argument(
        "--thread-timeout",
        type=int,
        default=3600,
        help="Maximum execution time per verification thread (seconds)."
    )
    parser_cpog.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel verification workers."
    )
    parser_cpog.add_argument(
        "--batch-size",
        type=int,
        help="Number of instances to process in a single batch."
    )
    parser_cpog.add_argument(
        "--memory-limit-gb",
        type=float,
        default=4.0,
        help="Maximum allowed memory usage during processing (GB)."
    )
    parser_cpog.set_defaults(func=command_cpog_verify)

def command_cpog_verify(args: argparse.Namespace) -> None:
    """
    Execute the 'cpog_verify' subcommand by:
      - Processing CSV files with multiple verification instances, OR
      - Verifying a single CNF file.
    """
    try:
        if args.csv_path:
            logger.info(f"Processing CSV file: {args.csv_path}")
            results = process_results_and_verify_with_cpog(
                csv_path=args.csv_path,
                verifier_dir=args.verifier_dir,
                thread_timeout=args.thread_timeout,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                memory_limit_gb=args.memory_limit_gb
            )
            output_path = get_output_path(args.csv_path)
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")

        elif args.cnf_path:
            logger.info(f"Verifying single CNF file: {args.cnf_path}")
            verify_single_cnf(
                cnf_path=args.cnf_path,
                verifier_dir=args.verifier_dir,
                output_dir=args.output_dir,
                timeout=args.thread_timeout
            )

        else:
            logger.error("Either --csv-path or --cnf-path must be provided.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during CPOG verification: {e}")
        sys.exit(1)

###############################################################################
# Subcommand: satzilla_extract
###############################################################################

def add_satzilla_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'satzilla_extract' subcommand parser.
    """
    parser_sz = subparsers.add_parser(
        "satzilla_extract",
        help="Extract Satzilla features from CNF instances."
    )
    parser_sz.add_argument(
        "--instances",
        type=str,
        required=True,
        help="Path to directory containing CNF files."
    )
    parser_sz.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path where feature extraction results are stored."
    )
    parser_sz.add_argument(
        "--satzilla-binary-path",
        type=str,
        required=True,
        help="Path to the Satzilla feature binary (e.g. 'satzilla')."
    )
    parser_sz.set_defaults(func=command_satzilla_extract)

def command_satzilla_extract(args: argparse.Namespace) -> None:
    """
    Execute the 'satzilla_extract' subcommand by calling
    satzilla_feature_extractor.compute_sat_feature_data.compute_features
    on the CNF files, then optionally process them (no duplicates, etc.).
    """
    try:
        logger.info(f"Extracting features for CNFs in {args.instances} ...")
        compute_features(
            cnf_dir=args.instances,
            features_output_dir=args.out_dir,
            satzilla_path=args.satzilla_binary_path
        )
        logger.info("Satzilla feature extraction complete.")
        process_csv_files(args.out_dir)
        logger.info("Post-processing of CSV files complete.")
    except Exception as e:
        logger.error(f"Error during Satzilla extraction: {e}")

###############################################################################
# Main CLI
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level parser and subcommand parsers.
    """
    parser = argparse.ArgumentParser(
        prog="global_cli",
        description="Unified CLI for generating, fuzzing, verifying, "
                    "and extracting Satzilla features."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    add_generate_subparser(subparsers)
    add_fuzz_subparser(subparsers)
    add_cpog_subparser(subparsers)
    add_satzilla_subparser(subparsers)

    return parser

def main() -> None:
    """
    Parse command line arguments and dispatch to the appropriate subcommand handler.
    """
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
