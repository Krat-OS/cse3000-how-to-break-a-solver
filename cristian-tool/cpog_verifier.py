import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def run_command(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 300) -> tuple[int, str, str]:
    """Execute a shell command with timeout.

    Args:
        cmd: Command to execute as list of strings
        cwd: Working directory for command execution
        timeout: Maximum execution time in seconds

    Returns:
        tuple: (return_code, stdout, stderr)

    Raises:
        subprocess.TimeoutExpired: If command execution exceeds timeout
    """
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Command timed out"

def extract_model_count(stdout: str) -> int:
    """Extract model count from command output.

    Args:
        stdout: Command output containing model count

    Returns:
        int: Extracted model count, 0 if not found
    """
    for line in stdout.splitlines():
        if "Regular model count = " in line:
            return int(line.split("=")[1].strip())
    return 0

def generate_d4nnf(cnf_path: Path, d4nnf_path: Path, verifier_dir: Path, timeout: int) -> tuple[bool, Optional[str]]:
    """Generate D4NNF from CNF file.

    Args:
        cnf_path: Input CNF file path
        d4nnf_path: Output D4NNF file path
        verifier_dir: Directory containing d4 binary
        timeout: Command timeout in seconds

    Returns:
        tuple: (success, error_message)
    """
    returncode, _, stderr = run_command([
        str(verifier_dir / "d4"),
        str(cnf_path),
        "-dDNNF",
        f"-out={str(d4nnf_path)}"
    ], timeout=timeout)
    return returncode == 0, None if returncode == 0 else f"D4NNF generation failed: {stderr}"

def generate_cpog(cnf_path: Path, d4nnf_path: Path, cpog_path: Path, verifier_dir: Path, timeout: int) -> tuple[bool, Optional[str]]:
    """Generate CPOG from D4NNF file.

    Args:
        cnf_path: Input CNF file path
        d4nnf_path: Input D4NNF file path
        cpog_path: Output CPOG file path
        verifier_dir: Directory containing cpog-gen binary
        timeout: Command timeout in seconds

    Returns:
        tuple: (success, error_message)
    """
    returncode, _, stderr = run_command([
        str(verifier_dir / "cpog-gen"),
        str(cnf_path),
        str(d4nnf_path),
        str(cpog_path)
    ], timeout=timeout)
    return returncode == 0, None if returncode == 0 else f"CPOG generation failed: {stderr}"

def verify_cpog(cnf_path: Path, cpog_path: Path, verifier_dir: Path, timeout: int) -> tuple[bool, Optional[str], int]:
    """Verify CPOG file and get model count.

    Args:
        cnf_path: Input CNF file path
        cpog_path: Input CPOG file path
        verifier_dir: Directory containing cpog-check binary
        timeout: Command timeout in seconds

    Returns:
        tuple: (success, error_message, model_count)
    """
    returncode, stdout, stderr = run_command([
        str(verifier_dir / "cpog-check"),
        str(cnf_path),
        str(cpog_path)
    ], timeout=timeout)
    if returncode != 0:
        return False, f"CPOG verification failed: {stderr}", 0
    return True, None, extract_model_count(stdout)

def verify_single_instance(
    cnf_path: Path, verifier_dir: Path, timeout: int = 300
) -> tuple[bool, Optional[str], int]:
    """Verify a single CNF instance using CPOG.

    Args:
        cnf_path: Path to CNF file
        verifier_dir: Directory containing verification tools
        timeout: Maximum execution time per command in seconds

    Returns:
        tuple: (success, error_message, model_count)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        d4nnf_path = work_dir / "temp.d4nnf"
        cpog_path = work_dir / "temp.cpog"

        success, error = generate_d4nnf(cnf_path, d4nnf_path, verifier_dir, timeout)
        if not success:
            return False, error, 0

        success, error = generate_cpog(cnf_path, d4nnf_path, cpog_path, verifier_dir, timeout)
        if not success:
            return False, error, 0

        return verify_cpog(cnf_path, cpog_path, verifier_dir, timeout)

def create_timeout_result() -> Dict[str, Any]:
    """Create result dictionary for timeout case.

    Returns:
        dict: Timeout result with default values
    """
    return {
        "cpog_error": "TIMEOUT",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def create_missing_instance_result() -> Dict[str, Any]:
    """Create result dictionary for missing instance case.

    Returns:
        dict: Missing instance result with default values
    """
    return {
        "cpog_error": "No instance provided",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def process_row(row: pd.Series, verifier_dir: Path) -> tuple[int, Dict[str, Any]]:
    """Process a single row of instance data.

    Args:
        row: DataFrame row containing instance data
        verifier_dir: Directory containing verification tools

    Returns:
        tuple: (row_index, results_dictionary)
    """
    if pd.isna(row["instance_path"]):
        return row.name, create_missing_instance_result()

    verified, error, cpog_count = verify_single_instance(Path(row["instance_path"]), verifier_dir)
    return row.name, {
        "cpog_error": "NO ERROR" if error is None else error,
        "cpog_count": cpog_count,
        "count_matches": abs(cpog_count - row["count_value"]) < 1e-6,
        "verified": verified
    }

def get_cpog_columns() -> List[str]:
    """Get list of CPOG-related column names in correct order.

    Returns:
        list: Ordered CPOG column names
    """
    return ["cpog_error", "cpog_count", "count_matches", "verified"]

def verify_with_cpog_parallel(
    df: pd.DataFrame,
    verifier_dir: Path | str = Path(__file__).parent / "cpog",
    thread_timeout: int = 3600
) -> pd.DataFrame:
    """Verify multiple CNF instances using CPOG in parallel.

    Args:
        df: DataFrame containing instance data
        verifier_dir: Directory containing verification tools
        thread_timeout: Maximum execution time per instance in seconds

    Returns:
        DataFrame: DataFrame with verification results
    """
    verifier_dir = Path(verifier_dir)
    df = df.copy()

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_row, row, verifier_dir)
            for _, row in df.iterrows()
        ]

        for future in futures:
            try:
                idx, results = future.result(timeout=thread_timeout)
                for key, value in results.items():
                    df.at[idx, key] = value
            except FuturesTimeoutError:
                df.at[future.idx, "cpog_error"] = "TIMEOUT"
                df.at[future.idx, "cpog_count"] = 0
                df.at[future.idx, "count_matches"] = False
                df.at[future.idx, "verified"] = False

    return df

def verify_with_cpog(
    df: pd.DataFrame,
    verifier_dir: Path | str = Path(__file__).parent / "cpog",
) -> pd.DataFrame:
    """Verify multiple CNF instances using CPOG sequentially.

    Args:
        df: DataFrame containing instance data
        verifier_dir: Directory containing verification tools
        timeout: Maximum execution time per instance in seconds

    Returns:
        DataFrame: DataFrame with verification results
    """
    verifier_dir = Path(verifier_dir)
    df = df.copy()
    
    for idx, row in df.iterrows():
        try:
            results = process_row(row, verifier_dir)[1]
            for key, value in results.items():
                df.at[idx, key] = value
        except Exception:
            df.at[idx, "cpog_error"] = "TIMEOUT"
            df.at[idx, "cpog_count"] = 0
            df.at[idx, "count_matches"] = False
            df.at[idx, "verified"] = False

    return df
