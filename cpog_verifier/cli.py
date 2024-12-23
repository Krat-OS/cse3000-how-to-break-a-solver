import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import logging
import tempfile
import shutil
import uuid
import signal
import sys

from result_processor.utils import process_results
from cpog_verifier.utils import verify_single_instance, verify_with_cpog, should_terminate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_results_and_verify_with_cpog(
    csv_path: str | Path,
    verifier_dir: str | Path,
    thread_timeout: int = 3600,
    max_workers: int = 10,
    batch_size: Optional[int] = None,
    memory_limit_gb: Optional[float] = 4.0
) -> pd.DataFrame:
    """Process results CSV and verify with CPOG without raising exceptions.

    This function always returns results (partial or full), even if interrupted
    or if an error occurs.

    Args:
        csv_path: Path to the CSV file to process.
        verifier_dir: Path to the directory containing verifier binaries.
        thread_timeout: Timeout in seconds for each thread.
        max_workers: Maximum number of parallel workers.
        batch_size: Number of instances to process in a single batch.
        memory_limit_gb: Maximum allowed memory usage during batch processing.

    Returns:
        DataFrame: Results with verification data added or partial results if interrupted.
    """
    # Start with a processed DataFrame from the CSV
    df = process_results(csv_path)

    df_with_cpog = None
    try:
        df_with_cpog = verify_with_cpog(
            df,
            verifier_dir,
            thread_timeout=thread_timeout,
            max_workers=max_workers,
            batch_size=batch_size,
            memory_limit_gb=memory_limit_gb
        )
        return df_with_cpog
    except KeyboardInterrupt:
        # Log and return whatever we have so far
        logging.info("Interrupted by user. Returning partial results.")
        return df_with_cpog if df_with_cpog is not None else df
    except Exception as e:
        # Log error and return partial results if available, else original
        logging.error(f"Error in verification process: {e}. Returning partial results.")
        return df_with_cpog if df_with_cpog is not None else df

def get_output_path(input_path: str | Path) -> Path:
    """Generate output path by appending '_with_cpog' to the original filename.

    Args:
        input_path: Original CSV file path.

    Returns:
        Path: Output file path with modified filename.
    """
    input_path = Path(input_path)
    return input_path.parent / f"{input_path.stem}_with_cpog.csv"

def verify_single_cnf(
    cnf_path: str | Path,
    verifier_dir: str | Path,
    output_dir: Optional[Path] = None,
    timeout: int = 300
) -> None:
    """Verify a single CNF file and optionally save the CPOG file.

    This function performs the following steps:
    1. Creates a temporary workspace
    2. Copies verifier binaries to the workspace
    3. Runs verification on the CNF file
    4. Optionally saves the CPOG file if verification succeeds
    5. Cleans up temporary files

    Args:
        cnf_path: Path to the CNF file to verify.
        verifier_dir: Path to the directory containing verifier binaries.
        output_dir: Directory to save the CPOG file (optional).
        timeout: Maximum execution time in seconds.

    Raises:
        FileNotFoundError: If the CNF file or verifier binaries are not found.
        RuntimeError: If verification fails or times out.
        OSError: If there are permission issues or I/O errors.
    """
    try:
        cnf_path = Path(cnf_path)
        verifier_dir = Path(verifier_dir)
        thread_id = str(uuid.uuid4())

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            new_verifier_dir = workspace / "cpog"
            shutil.copytree(verifier_dir, new_verifier_dir)

            for executable in new_verifier_dir.glob("*"):
                if not executable.name.startswith('.'):
                    executable.chmod(0o755)

            logging.info(f"Verifying CNF file: {cnf_path}")

            verified, error, model_count = verify_single_instance(
                cnf_path=cnf_path,
                workspace=workspace,
                thread_id=thread_id,
                timeout=timeout
            )

            print(f"Model count: {model_count}")
            print(f"Verified: {verified}")
            if error:
                print(f"Error: {error}")

            if verified and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                cpog_output = output_dir / f"{cnf_path.stem}.cpog"
                shutil.copy(workspace / f"temp_{thread_id}.cpog", cpog_output)
                logging.info(f"Saved CPOG file to: {cpog_output}")

    except KeyboardInterrupt:
        logging.info("Verification interrupted by user.")
        raise
    except Exception as e:
        logging.error(f"Error during verification: {e}")
        raise

def main() -> None:
    """
    Main function to parse command-line arguments and process CSV or CNF files.

    This function handles the command-line interface for the verifier tool with
    comprehensive error handling and support for graceful interruption.

    Supports two primary modes of operation:
    1. Processing a CSV file with multiple verification instances
    2. Verifying a single CNF file

    Command-line arguments control various aspects of verification, including:
    - Input file selection (CSV or CNF)
    - Verification tool configuration
    - Parallel processing parameters
    - Memory and timeout constraints

    Raises:
        SystemExit: On unrecoverable errors or CLI usage mistakes
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="CPOG Verification Tool for CNF Instances"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file containing multiple verification instances"
    )
    parser.add_argument(
        "--cnf-path",
        type=str,
        help="Path to single CNF file for direct verification"
    )
    parser.add_argument(
        "--verifier-dir",
        default=Path(__file__).parent / "cpog",
        type=Path,
        help="Directory containing verification tool binaries"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save generated CPOG files"
    )
    parser.add_argument(
        "--thread-timeout",
        type=int,
        default=3600,
        help="Maximum execution time per verification thread (seconds)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel verification workers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of instances to process in a single batch"
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=4.0,
        help="Maximum memory usage allowed during batch processing (GB)"
    )
    args: argparse.Namespace = parser.parse_args()

    results: Optional[pd.DataFrame] = None
    output_path: Optional[Path] = None

    try:
        if args.csv_path:
            logging.info(f"Processing CSV file: {args.csv_path}")
            results = process_results_and_verify_with_cpog(
                args.csv_path,
                args.verifier_dir,
                thread_timeout=args.thread_timeout,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                memory_limit_gb=args.memory_limit_gb
            )
            output_path = get_output_path(args.csv_path)

        elif args.cnf_path:
            verify_single_cnf(
                args.cnf_path,
                args.verifier_dir,
                output_dir=args.output_dir,
                timeout=args.thread_timeout
            )
            return

        else:
            logging.error(
                "Invalid usage: Either --csv-path or --cnf-path must be provided"
            )
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        # If an unexpected error occurs here, we still may have partial results.
        logging.error(f"Unexpected error: {e}")

    finally:
        if results is not None and output_path is not None:
            try:
                results.to_csv(output_path, index=False)
                logging.info(f"Results saved to: {output_path}")
            except Exception as save_error:
                logging.error(f"Failed to save results: {save_error}")
                sys.exit(1)


if __name__ == "__main__":
    main()
