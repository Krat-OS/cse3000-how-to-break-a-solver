import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import threading

import pandas as pd

print_lock = threading.Lock()
remaining_count_lock = threading.Lock()
remaining_count = 0

def print_progress(message: str) -> None:
    """Print a timestamped progress message in a thread-safe manner.

    Args:
        message: The message to be printed
    """
    with print_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

def update_remaining_count(delta: int = -1) -> int:
    """Update the global count of remaining instances in a thread-safe manner.

    Args:
        delta: The value to add to the remaining count (default: -1)

    Returns:
        int: The updated count of remaining instances
    """
    global remaining_count
    with remaining_count_lock:
        remaining_count += delta
        return remaining_count

def setup_instance_workspace(instance_path: Path, verifier_dir: Path) -> Path:
    """Create and setup a temporary workspace for processing an instance.

    Args:
        instance_path: Path to the CNF instance file
        verifier_dir: Path to the directory containing verification tools

    Returns:
        Path: Path to the created workspace directory
    """
    instance_name = instance_path.stem
    workspace = Path(tempfile.gettempdir()) / f"cpog_workspace_{instance_name}"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    new_verifier_dir = workspace / "cpog"
    shutil.copytree(verifier_dir, new_verifier_dir)

    for executable in new_verifier_dir.glob("*"):
        if not executable.name.startswith('.'):
            executable.chmod(0o755)

    return workspace

def cleanup_workspace(workspace: Path) -> None:
    """Remove a temporary workspace directory.

    Args:
        workspace: Path to the workspace directory to be cleaned up
    """
    try:
        shutil.rmtree(workspace)
        print_progress(f"Cleaned up workspace: {workspace}")
    except Exception as e:
        print_progress(f"Error cleaning up workspace {workspace}: {e}")

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
    returncode, stdout, stderr = run_command([
        str(verifier_dir / "cpog-gen"),
        str(cnf_path),
        str(d4nnf_path),
        str(cpog_path)
    ], timeout=timeout)

    if "Compiled formula unsatisfiable.  Cannot verify" in stdout:
        return False, "UNSAT"

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
    cnf_path: Path, workspace: Path, timeout: int = 300
) -> tuple[bool, Optional[str], int]:
    """Verify a single CNF instance using CPOG.

    Args:
        cnf_path: Path to CNF file
        workspace: Instance-specific workspace directory
        timeout: Maximum execution time per command in seconds

    Returns:
        tuple: (success, error_message, model_count)
    """
    verifier_dir = workspace / "cpog"
    d4nnf_path = workspace / "temp.d4nnf"
    cpog_path = workspace / "temp.cpog"

    success, error = generate_d4nnf(cnf_path, d4nnf_path, verifier_dir, timeout)
    if not success:
        return False, error, 0

    success, error = generate_cpog(cnf_path, d4nnf_path, cpog_path, verifier_dir, timeout)
    if not success:
        return False, error, 0

    return verify_cpog(cnf_path, cpog_path, verifier_dir, timeout)

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

    if (pd.isna(row["count_value"]) or
        not float(row["count_value"]).is_integer() or
        row.get("_skip_due_to_threshold", False)):
        return row.name, create_invalid_count_result()

    instance_path = Path(row["instance_path"])
    print_progress(f"Starting verification for instance: {instance_path.name} (count_value: {row['count_value']})")

    workspace = setup_instance_workspace(instance_path, verifier_dir)

    try:
        verified, error, cpog_count = verify_single_instance(instance_path, workspace)
        result = {
            "cpog_message": "NO ERROR" if error is None else error,
            "cpog_count": cpog_count,
            "count_matches": (
                (cpog_count > 0 and abs(cpog_count - row["count_value"]) < 1e-6) or
                (cpog_count == 0 and row["count_value"] == 0 and
                 row["satisfiability"] == "UNSATISFIABLE" and error == "UNSAT")
            ),
            "verified": verified
        }
    except Exception as e:
        result = {
            "cpog_message": f"ERROR: {str(e)}",
            "cpog_count": 0,
            "count_matches": False,
            "verified": False
        }
    finally:
        cleanup_workspace(workspace)
        remaining = update_remaining_count()
        print_progress(
            f"Completed verification for {instance_path.name}. Remaining instances: {remaining}")

    return row.name, result

def verify_with_cpog(
    df: pd.DataFrame,
    verifier_dir: Path | str = Path(__file__).parent / "cpog",
    thread_timeout: int = 3600,
    max_workers: int = 10,
    max_cnt_thresh: Optional[int] = None
) -> pd.DataFrame:
    """Verify multiple CNF instances using CPOG in parallel.

    Args:
        df: DataFrame containing instance data
        verifier_dir: Directory containing verification tools
        thread_timeout: Maximum execution time per instance in seconds
        max_workers: Maximum number of concurrent threads
        max_cnt_thresh: Skip instances with count_value above this threshold

    Returns:
        pd.DataFrame: DataFrame with verification results
    """
    verifier_dir = Path(verifier_dir)
    results_df = df.copy()

    df_to_process = df.copy()

    df_to_process["_skip_due_to_threshold"] = False
    if max_cnt_thresh is not None:
        df_to_process.loc[df_to_process["count_value"] > max_cnt_thresh, "_skip_due_to_threshold"] = True

    df_to_process = df_to_process.sort_values('count_value')

    global remaining_count
    remaining_count = len(df_to_process)
    print_progress(f"Starting verification of {remaining_count} instances with {max_workers} workers")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_row, row, verifier_dir): idx
                for idx, row in df_to_process.iterrows()
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    idx, results = future.result(timeout=thread_timeout)
                    for key, value in results.items():
                        if key == "cpog_count":
                            value = int(value)
                        results_df.at[idx, key] = value

                except FuturesTimeoutError:
                    print_progress(f"Timeout occurred for instance at index {idx}")
                    results_df.at[idx, "cpog_message"] = "TIMEOUT"
                    results_df.at[idx, "cpog_count"] = 0
                    results_df.at[idx, "count_matches"] = False
                    results_df.at[idx, "verified"] = False
                    update_remaining_count()

        print_progress("Verification complete for all instances")

    except KeyboardInterrupt:
        print_progress("\nInterrupted by user. Returning partial results...")
        executor.shutdown(wait=False)

    if "_skip_due_to_threshold" in results_df.columns:
        results_df.drop("_skip_due_to_threshold", axis=1, inplace=True)

    return results_df

def create_timeout_result() -> Dict[str, Any]:
    """Create a result dictionary for timeout case."""
    return {
        "cpog_message": "TIMEOUT",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def create_missing_instance_result() -> Dict[str, Any]:
    """Create a result dictionary for missing instance case."""
    return {
        "cpog_message": "No instance provided",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def create_invalid_count_result():
    """Create a result dictionary for invalid count value case."""
    return {
        "cpog_message": "Invalid count value",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }
