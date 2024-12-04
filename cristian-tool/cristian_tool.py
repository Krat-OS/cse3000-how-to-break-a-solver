import argparse
from pathlib import Path

import pandas as pd
from cpog_verifier import verify_single_instance, verify_with_cpog
from result_proccessor import process_results


def process_results_and_verify_with_cpog(
    csv_path: str | Path, verifier_dir: str | Path
) -> pd.DataFrame:
    """
    Process results CSV and verify with CPOG.
    
    Args:
        csv_path: Path to the CSV file to process.
        verifier_dir: Path to the directory containing verifier binaries.

    Returns:
        DataFrame: Results with verification data added.
    """
    df = process_results(csv_path)
    return verify_with_cpog(df, verifier_dir)

def main() -> None:
    """
    Main function to parse command-line arguments and process CSV or CNF files.

    Command-line arguments:
    --csv-path: Path to the CSV file to process.
    --cnf-path: Path to the CNF file to verify.
    --verifier-dir: Path to the directory containing verifier binaries 
        (default: "cpog" directory relative to the script).
    --output-dir: Directory to save the CPOG file.

    If --csv-path is provided, processes the results and verifies with CPOG.
    If --cnf-path is provided, reads the CNF file, verifies the single instance, 
        and optionally saves the CPOG file to the specified output directory.
    """
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
    args = parser.parse_args()

    if args.csv_path:
        results = process_results_and_verify_with_cpog(args.csv_path, args.verifier_dir)
        print(results)
        return

    if args.cnf_path:
        cnf_content = Path(args.cnf_path).read_text(encoding="utf-8")
        verified, error, model_count = verify_single_instance(
            cnf_content, args.verifier_dir
        )

        print(f"Model count: {model_count}")
        print(f"Verified: {verified}")
        if error:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()
