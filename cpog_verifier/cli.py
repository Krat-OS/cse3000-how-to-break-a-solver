import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import logging
import tempfile
import shutil
import uuid

from result_processor.utils import process_results
from cpog_verifier.utils import verify_single_instance, verify_with_cpog

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_results_and_verify_with_cpog(
    csv_path: str | Path,
    verifier_dir: str | Path,
    thread_timeout: int = 3600,
    max_workers: int = 10,
    batch_size: Optional[int] = None,
    memory_limit_gb: Optional[float] = 4.0
) -> pd.DataFrame:
    """Process results CSV and verify with CPOG.

    Args:
        csv_path: Path to the CSV file to process.
        verifier_dir: Path to the directory containing verifier binaries.
        thread_timeout: Timeout in seconds for each thread.
        max_workers: Maximum number of parallel workers.
        batch_size: Number of instances to process in a single batch.
        memory_limit_gb: Maximum allowed memory usage during batch processing.

    Returns:
        DataFrame: Results with verification data added.
    """
    df = process_results(csv_path)
    return verify_with_cpog(
        df,
        verifier_dir,
        thread_timeout=thread_timeout,
        max_workers=max_workers,
        batch_size=batch_size,
        memory_limit_gb=memory_limit_gb
    )


def get_output_path(input_path: str | Path) -> Path:
    """Generate output path by appending '_cristian_tool_output' to the original filename.

    Args:
        input_path: Original CSV file path.

    Returns:
        Path: Output file path with modified filename.
    """
    input_path = Path(input_path)
    return input_path.parent / f"{input_path.stem}_cristian_tool_output.csv"


def verify_single_cnf(
    cnf_path: str | Path,
    verifier_dir: str | Path,
    output_dir: Optional[Path] = None,
    timeout: int = 300
) -> None:
    """Verify a single CNF file and optionally save the CPOG file.

    Args:
        cnf_path: Path to the CNF file to verify.
        verifier_dir: Path to the directory containing verifier binaries.
        output_dir: Directory to save the CPOG file (optional).
        timeout: Maximum execution time in seconds.
    """
    cnf_path = Path(cnf_path)
    verifier_dir = Path(verifier_dir)
    thread_id = str(uuid.uuid4())

    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        # Copy verifier binaries to workspace
        new_verifier_dir = workspace / "cpog"
        shutil.copytree(verifier_dir, new_verifier_dir)
        
        # Set executable permissions
        for executable in new_verifier_dir.glob("*"):
            if not executable.name.startswith('.'):
                executable.chmod(0o755)

        logging.info(f"Verifying CNF file: {cnf_path}")

        # Run verification
        verified, error, model_count = verify_single_instance(
            cnf_path=cnf_path,
            workspace=workspace,
            thread_id=thread_id,
            timeout=timeout
        )

        # Print results
        print(f"Model count: {model_count}")
        print(f"Verified: {verified}")
        if error:
            print(f"Error: {error}")

        # Save CPOG file if verification succeeded and output directory is specified
        if verified and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cpog_output = output_dir / f"{cnf_path.stem}.cpog"
            shutil.copy(workspace / f"temp_{thread_id}.cpog", cpog_output)
            logging.info(f"Saved CPOG file to: {cpog_output}")


def main() -> None:
    """Main function to parse command-line arguments and process CSV or CNF files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", help="Path to CSV file")
    parser.add_argument("--cnf-path", help="Path to CNF file")
    parser.add_argument(
        "--verifier-dir",
        default=Path(__file__).parent / "cpog",
        type=Path,
        help="Path to verifier binaries"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save CPOG file"
    )
    parser.add_argument(
        "--thread-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each thread"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers"
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
        help="Maximum allowed memory usage during batch processing"
    )
    args = parser.parse_args()

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
        results.to_csv(output_path, index=False)
        logging.info(f"Results saved to: {output_path}")
        return

    if args.cnf_path:
        verify_single_cnf(
            args.cnf_path,
            args.verifier_dir,
            output_dir=args.output_dir,
            timeout=args.thread_timeout
        )
        return

    logging.error("Either --csv-path or --cnf-path must be provided.")
    parser.print_help()

if __name__ == "__main__":
    main()
