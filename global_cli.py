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
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd

from cpog_verifier.utils import verify_with_cpog, verify_single_instance
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
        "--sharpvelvet-generate",
        type=str,
        required=True,
        help="Path to SharpVelvet's generate_instances.py."
    )
    parser_gen.add_argument(
        "--sharpvelvet-out",
        type=str,
        required=True,
        help="Path to SharpVelvet/out directory."
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
      1) Calls generate_instances.py for each generator
      2) Moves the output to --out-dir.
    """
    for generator_path in args.generators:
        if not os.path.isfile(generator_path):
            logger.warning(f"Generator file not found: {generator_path}")
            continue

        cmd = [
            sys.executable,
            args.sharpvelvet_generate,
            "--generators",
            generator_path,
            "--num-iter",
            str(args.num_iter),
        ]
        logger.info(f"Running generation command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Move everything from sharpvelvet_out -> out_dir
    if not os.path.isdir(args.sharpvelvet_out):
        logger.warning(f"SharpVelvet out dir does not exist: {args.sharpvelvet_out}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for item in os.listdir(args.sharpvelvet_out):
        src = os.path.join(args.sharpvelvet_out, item)
        dst = os.path.join(args.out_dir, item)
        logger.info(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    logger.info("Generation complete.")


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
    parser_fz.set_defaults(func=command_fuzz)

def command_fuzz(args: argparse.Namespace) -> None:
    """
    Execute the 'fuzz' subcommand:
      1) For each solver, create a temp dir, copy instances, run run_fuzzer.py
      2) Merge all _fuzz-results.csv into a single CSV in the instances directory
    """
    csv_paths: List[str] = []

    def prepare_temp_dir(instances_dir: str) -> str:
        """
        Create a temp directory and copy all CNF instances into it.
        """
        tmp_dir = tempfile.mkdtemp(prefix="fuzz_")
        shutil.copytree(instances_dir, os.path.join(tmp_dir, "cnf"))
        return tmp_dir

    def run_fuzzer_on_solver(solver_path: str) -> Optional[str]:
        """
        Run the fuzzer for a single solver in a temp dir. Return CSV path if found.
        """
        if not os.path.isfile(solver_path):
            logger.warning(f"Solver file not found: {solver_path}")
            return None

        tmp_dir = prepare_temp_dir(args.instances)
        cmd = [
            sys.executable,
            args.sharpvelvet_fuzzer,
            "--counters",
            solver_path,
            "--instances",
            os.path.join(tmp_dir, "cnf"),
            "--timeout",
            str(args.solver_timeout),
        ]
        logger.info(f"Running fuzzer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Look for _fuzz-results.csv in tmp_dir
        for f in os.listdir(tmp_dir):
            if f.endswith("_fuzz-results.csv"):
                return os.path.join(tmp_dir, f)
        return None

    futures = []
    with ThreadPoolExecutor(max_workers=len(args.solvers)) as executor:
        for solver_json in args.solvers:
            futures.append(executor.submit(run_fuzzer_on_solver, solver_json))

        for future in as_completed(futures):
            result = future.result()
            if result:
                csv_paths.append(result)

    # Merge all fuzz-results CSVs
    if csv_paths:
        out_csv = os.path.join(args.instances, "fuzz-results-merged.csv")
        logger.info(f"Merging {len(csv_paths)} CSVs into {out_csv}")
        merge_csv_files(csv_paths, out_csv)
        logger.info("Fuzzing complete.")
    else:
        logger.warning("No fuzz-results CSVs were produced.")

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
        "--csv-paths",
        nargs="+",
        help="Paths to fuzz-results CSVs (or directories) to verify with CPOG."
    )
    parser_cpog.add_argument(
        "--cnf-path",
        type=str,
        help="Path to single CNF file for direct verification."
    )
    parser_cpog.add_argument(
        "--verifier-dir",
        type=str,
        default="cpog",
        help="Directory containing verifier binaries (if needed)."
    )
    parser_cpog.add_argument(
        "--thread-timeout",
        type=int,
        default=3600,
        help="Timeout (seconds) for each verification thread."
    )
    parser_cpog.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum parallel verification workers."
    )
    parser_cpog.add_argument(
        "--batch-size",
        type=int,
        help="Number of instances to process in one batch."
    )
    parser_cpog.add_argument(
        "--memory-limit-gb",
        type=float,
        default=4.0,
        help="Maximum memory usage in GB."
    )
    parser_cpog.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save any resulting CPOG file from single-CNF verification."
    )
    parser_cpog.set_defaults(func=command_cpog_verify)

def command_cpog_verify(args: argparse.Namespace) -> None:
    """
    Execute the 'cpog_verify' subcommand by:
      1) If --csv-paths given, run verify_with_cpog on each CSV
      2) If --cnf-path given, run verify_single_instance
    """
    # If user gave CSV paths, batch-verify
    if args.csv_paths:
        # Expand any directory entries into a list of CSVs
        csv_files = collect_csv_files(args.csv_paths)
        if not csv_files:
            logger.error("No CSV files found from --csv-paths.")
            return

        for csv_file in csv_files:
            logger.info(f"Verifying CSV with CPOG: {csv_file}")
            import pandas as pd
            df = pd.read_csv(csv_file)
            verified_df = verify_with_cpog(
                df,
                verifier_dir=args.verifier_dir,
                thread_timeout=args.thread_timeout,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                memory_limit_gb=args.memory_limit_gb
            )
            # Save output CSV
            out_file = get_cpog_verified_out(csv_file)
            verified_df.to_csv(out_file, index=False)
            logger.info(f"Saved verified CSV to: {out_file}")

    # If user gave single CNF path
    elif args.cnf_path:
        cnf_p = Path(args.cnf_path)
        if not cnf_p.is_file():
            logger.error(f"CNF path is not a file: {cnf_p}")
            return
        logger.info(f"Verifying single CNF with cpog_verifier: {cnf_p}")
        try:
            # We'll do a quick workspace approach, or direct call
            # (If you have a special function for single CNF verification, use it.)
            workspace = tempfile.mkdtemp(prefix="cpog_single_")
            verified, error, model_count = verify_single_instance(
                cnf_p,
                Path(workspace),
                thread_id="single_cnfp",
                timeout=args.thread_timeout
            )
            shutil.rmtree(workspace, ignore_errors=True)

            logger.info(f"Model count: {model_count}, Verified: {verified}, Error: {error}")
            if verified and args.output_dir:
                # Optionally copy out the .cpog file if you'd like.
                logger.info("You can copy the .cpog file from the workspace if needed.")
        except Exception as e:
            logger.error(f"Error verifying single CNF: {e}")
    else:
        logger.error("Must provide either --csv-paths or --cnf-path.")
        return

    logger.info("cpog_verify complete.")

def collect_csv_files(paths: List[str]) -> List[str]:
    """
    Given a list of file/dir paths, collect all .csv files.
    """
    collected = []
    for p in paths:
        pth = Path(p)
        if pth.is_file() and pth.suffix == ".csv":
            collected.append(str(pth))
        elif pth.is_dir():
            for f in pth.glob("*.csv"):
                collected.append(str(f))
    return collected

def get_cpog_verified_out(csv_file: str) -> str:
    """
    Return a new CSV path for the verified result, e.g. `original_with_cpog.csv`.
    """
    p = Path(csv_file)
    return str(p.parent / f"{p.stem}_with_cpog.csv")


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
        # Optionally process CSV files
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
