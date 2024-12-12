import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import threading
import psutil
import pandas as pd
import gc
import uuid
from os import environ
import time
import signal
import atexit
import logging
import sys

# Global locks and variables
remaining_count_lock: threading.Lock = threading.Lock()
remaining_count: int = 0
batch_executor: Optional[ThreadPoolExecutor] = None
should_terminate: threading.Event = threading.Event()
active_processes: List[subprocess.Popen] = []
processes_lock: threading.Lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_memory_usage_gb() -> float:
    """
    Return current memory usage of this process in gigabytes.

    Returns:
        float: Memory usage in GB.
    """
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 3)


def monitor_memory_usage(memory_limit_gb: float, check_interval: float = 1.0) -> None:
    """
    Monitor memory usage in a separate thread and terminate if limit is exceeded.

    Args:
        memory_limit_gb (float): Maximum allowed memory usage in GB.
        check_interval (float): Time between checks in seconds.
    """
    while not should_terminate.is_set():
        current_usage = get_memory_usage_gb()
        if current_usage > memory_limit_gb:
            logging.warning(
                "Memory limit exceeded: %.2fGB > %.2fGB",
                current_usage, memory_limit_gb
            )
            if batch_executor:
                batch_executor.shutdown(wait=False, cancel_futures=True)
            clear_ram()
        time.sleep(check_interval)


def update_remaining_count(delta: int = -1) -> int:
    """
    Update global count of remaining instances in a thread-safe manner.

    Args:
        delta (int): Value to add to remaining_count.

    Returns:
        int: Updated remaining_count.
    """
    global remaining_count
    with remaining_count_lock:
        remaining_count += delta
        return remaining_count


def setup_instance_workspace(
    instance_path: Path, verifier_dir: Path, thread_id: str
) -> Path:
    """
    Create a unique temporary workspace for an instance.

    Args:
        instance_path (Path): Path to the CNF instance file.
        verifier_dir (Path): Path to directory with verification tools.
        thread_id (str): Unique thread identifier.

    Returns:
        Path: Workspace directory path.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"cpog_workspace_{thread_id}_{instance_path.stem}_")
    workspace = Path(temp_dir)
    new_verifier_dir: Path = workspace / "cpog"
    shutil.copytree(verifier_dir, new_verifier_dir)

    for executable in new_verifier_dir.glob("*"):
        if not executable.name.startswith('.'):
            executable.chmod(0o755)

    return workspace


def cleanup_workspace(workspace: Path) -> None:
    """
    Remove a temporary workspace directory.

    Args:
        workspace (Path): Workspace directory to remove.
    """
    try:
        if workspace.exists():
            shutil.rmtree(workspace)
            logging.info("Cleaned up workspace: %s", workspace)
    except Exception as e:
        logging.error("Error cleaning up workspace %s: %s", workspace, e)


def cleanup_on_exit() -> None:
    """
    Cleanup function registered to run at program exit.
    Terminates subprocesses and clears RAM.
    """
    clear_ram()
    with processes_lock:
        for process in active_processes:
            try:
                process.terminate()
                process.wait(timeout=1)
            except:
                try:
                    process.kill()
                except:
                    pass


def clear_ram() -> None:
    """
    Force clear RAM by terminating subprocesses and invoking garbage collection.
    """
    gc.collect()
    proc: psutil.Process = psutil.Process()
    for child in proc.children(recursive=True):
        try:
            child.terminate()
        except Exception:
            pass

    _, alive = psutil.wait_procs(proc.children(recursive=True), timeout=3)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
    env: Optional[dict] = None
) -> Tuple[int, str, str]:
    """
    Execute a command with timeout and optional env vars.

    Args:
        cmd (List[str]): Command to execute.
        cwd (Optional[Path]): Working directory.
        timeout (int): Timeout in seconds.
        env (Optional[dict]): Environment variables.

    Returns:
        Tuple[int, str, str]: (Return code, stdout, stderr)
    """
    process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env
        )
        with processes_lock:
            active_processes.append(process)
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        return -1, "", "Command timed out"
    finally:
        if process:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            process.wait()
            with processes_lock:
                if process in active_processes:
                    active_processes.remove(process)


def verify_single_instance(
    cnf_path: Path,
    workspace: Path,
    thread_id: str,
    timeout: int = 300
) -> Tuple[bool, Optional[str], int]:
    """
    Verify a single CNF instance using CPOG.

    Args:
        cnf_path (Path): Path to CNF file.
        workspace (Path): Workspace directory.
        thread_id (str): Unique thread ID.
        timeout (int): Timeout in seconds.

    Returns:
        Tuple[bool, Optional[str], int]: (verified, error, model_count)
    """
    verifier_dir: Path = workspace / "cpog"
    d4nnf_path: Path = workspace / f"temp_{thread_id}.d4nnf"
    cpog_path: Path = workspace / f"temp_{thread_id}.cpog"

    try:
        # Generate D4NNF
        returncode, _, stderr = run_command(
            [
                str(verifier_dir / "d4"),
                str(cnf_path),
                "-dDNNF",
                f"-out={str(d4nnf_path)}"
            ],
            cwd=workspace,
            timeout=timeout
        )
        if returncode != 0:
            return False, f"D4NNF generation failed: {stderr}", 0

        # Set environment for cpog-gen and cpog-check
        env = environ.copy()
        env["PATH"] = f"{str(verifier_dir)}:/usr/local/bin:{env.get('PATH', '')}"

        # Generate CPOG
        returncode, stdout, stderr = run_command(
            [
                str(verifier_dir / "cpog-gen"),
                str(cnf_path.resolve()),
                str(d4nnf_path.resolve()),
                str(cpog_path.resolve())
            ],
            cwd=workspace,
            timeout=timeout,
            env=env
        )

        if "Compiled formula unsatisfiable.  Cannot verify" in stdout:
            # Unsat instance
            # Check D4NNF file just in case
            if not d4nnf_path.exists():
                logging.error("D4NNF file not created for unsat instance.")
                return False, "D4NNF file missing", 0
            return False, "UNSAT", 0

        if returncode != 0:
            return False, f"CPOG generation failed: {stderr}", 0

        # Verify CPOG
        returncode, stdout, stderr = run_command(
            [
                str(verifier_dir / "cpog-check"),
                str(cnf_path.resolve()),
                str(cpog_path.resolve())
            ],
            cwd=workspace,
            timeout=timeout,
            env=env
        )

        if returncode != 0:
            return False, f"CPOG verification failed: {stderr}", 0

        if not d4nnf_path.exists():
            logging.error("D4NNF file %s was not created.", d4nnf_path)
            return False, "D4NNF file missing", 0

        verified = True
        if verified and not cpog_path.exists():
            logging.error("CPOG file %s not created despite success.", cpog_path)
            return False, "CPOG file missing after success", 0

        model_count: int = 0
        for line in stdout.splitlines():
            if "Regular model count = " in line:
                model_count = int(line.split("=")[1].strip())
                break

        return True, None, model_count

    except Exception as e:
        return False, f"Verification failed with error: {str(e)}", 0


def process_instance_group(
    group: pd.DataFrame,
    verifier_dir: Path,
    timeout: int
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Process a group of rows with the same instance_path.

    Args:
        group (pd.DataFrame): Rows with same instance_path.
        verifier_dir (Path): Directory of verification tools.
        timeout (int): Timeout in seconds.

    Returns:
        List[Tuple[int, Dict[str, Any]]]: Results for each row in the group.
    """
    if pd.isna(group["instance_path"].iloc[0]):
        return [(idx, create_missing_instance_result()) for idx in group.index]

    thread_id: str = str(uuid.uuid4())
    instance_path: Path = Path(group["instance_path"].iloc[0])
    workspace: Path = setup_instance_workspace(instance_path, verifier_dir, thread_id)
    results: List[Tuple[int, Dict[str, Any]]] = []

    try:
        verified, error, cpog_count = verify_single_instance(
            instance_path, workspace, thread_id, timeout
        )

        for idx, row in group.iterrows():
            # Even if count_value invalid, we still fill in results
            result: Dict[str, Any] = {
                "cpog_message": "NO ERROR" if error is None else error,
                "cpog_count": cpog_count,
                "count_matches": (
                    (cpog_count > 0 and
                     abs(cpog_count - row.get("count_value", 0)) < 1e-6) or
                    (cpog_count == 0 and row.get("count_value", 0) == 0 and
                     row.get("satisfiability", "") == "UNSATISFIABLE" and error == "UNSAT")
                ),
                "verified": verified
            }

            val = row.get("count_value")
            if pd.isna(val) or not isinstance(val, (int, float)) \
               or not float(val).is_integer():
                result["cpog_message"] = "Invalid count value"

            results.append((idx, result))

    except Exception as e:
        err_res = create_cpog_error_result(f"ERROR: {str(e)}")
        results = [(idx, err_res) for idx in group.index]
    finally:
        cleanup_workspace(workspace)
        remaining: int = update_remaining_count()
        logging.info(
            "Completed verification for %s. Remaining instance groups: %d",
            instance_path.name, remaining
        )

    return results


def _setup_signal_handlers() -> None:
    """
    Set up signal handlers for graceful shutdown.
    """
    def signal_handler(signum, frame):
        global batch_executor, should_terminate
        logging.warning("Received signal %d, initiating shutdown...", signum)
        should_terminate.set()
        if batch_executor:
            batch_executor.shutdown(wait=False, cancel_futures=True)
        clear_ram()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def verify_with_cpog(
    df: pd.DataFrame,
    verifier_dir: Path | str = Path(__file__).parent / "cpog",
    thread_timeout: int = 3600,
    max_workers: int = 10,
    batch_size: Optional[int] = None,
    memory_limit_gb: float = 4.0
) -> pd.DataFrame:
    """
    Verify multiple CNF instances with CPOG. Supports batching and threading.

    Args:
        df (pd.DataFrame): Dataframe with instances to verify.
        verifier_dir (Path | str): Path to verification tools.
        thread_timeout (int): Timeout per instance.
        max_workers (int): Max parallel workers.
        batch_size (Optional[int]): Instances per batch.
        memory_limit_gb (float): Memory limit in GB.

    Returns:
        pd.DataFrame: Results with verification data added.
    """
    global batch_executor, should_terminate

    _setup_signal_handlers()
    atexit.register(cleanup_on_exit)

    should_terminate.clear()
    verifier_dir = Path(verifier_dir)
    results_df: pd.DataFrame = df.copy()

    # Check required tools exist
    required_tools = ["d4", "cpog-gen", "cpog-check", "cadical", "drat-trim"]
    for tool in required_tools:
        if not (verifier_dir / tool).exists():
            logging.error("Required tool %s not found in %s", tool, verifier_dir)
            sys.exit(1)

    try:
        grouped = df.groupby("instance_path", group_keys=True)
        groups: List[Tuple[str, pd.DataFrame]] = list(grouped)
        total_groups: int = len(groups)

        global remaining_count
        remaining_count = total_groups

        logging.info(
            "Starting verification of %d unique instances with %d workers",
            total_groups, max_workers
        )

        memory_monitor = threading.Thread(
            target=monitor_memory_usage,
            args=(memory_limit_gb,),
            daemon=True
        )
        memory_monitor.start()

        batch_size = batch_size or total_groups
        batch_count: int = 0

        for i in range(0, total_groups, batch_size):
            if should_terminate.is_set():
                logging.info("Termination requested, stopping after current batch")
                break

            batch_count += 1
            current_batch = groups[i:i + batch_size]
            logging.info(
                "Processing batch %d with %d instance groups",
                batch_count, len(current_batch)
            )

            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_executor = executor
                    future_to_group = {
                        executor.submit(
                            process_instance_group,
                            group,
                            verifier_dir,
                            thread_timeout
                        ): group_name
                        for group_name, group in current_batch
                    }

                    for future in as_completed(future_to_group):
                        group_name = future_to_group[future]
                        try:
                            results = future.result()
                            for idx, result in results:
                                for key, value in result.items():
                                    if key == "cpog_count":
                                        value = int(value)
                                    results_df.at[idx, key] = value
                        except Exception as e:
                            logging.error(
                                "Error processing group %s: %s", group_name, e
                            )
                            mark_group_as_failed(
                                results_df, grouped.get_group(group_name)
                            )

            except Exception as e:
                logging.error("Batch %d interrupted: %s", batch_count, str(e))
                mark_remaining_as_failed(results_df, future_to_group, grouped)
            finally:
                batch_executor = None
                clear_ram()
                logging.info("Batch %d complete", batch_count)

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving partial results...")
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        should_terminate.set()
        if memory_monitor.is_alive():
            memory_monitor.join(timeout=1.0)

        if batch_executor:
            batch_executor.shutdown(wait=False, cancel_futures=True)
            batch_executor = None
        clear_ram()

    logging.info("Verification complete. Processed batches: %d", batch_count)
    return results_df


def mark_group_as_failed(results_df: pd.DataFrame, group: pd.DataFrame) -> None:
    """
    Mark all instances in a group as failed.
    """
    err = create_processing_failed_result()
    for idx in group.index:
        for key, value in err.items():
            results_df.at[idx, key] = value


def mark_remaining_as_failed(
    results_df: pd.DataFrame,
    future_to_group: Dict,
    grouped: pd.core.groupby.DataFrameGroupBy
) -> None:
    """
    Mark all remaining unprocessed instances as failed.
    """
    for future, group_name in future_to_group.items():
        if not future.done():
            group = grouped.get_group(group_name)
            mark_group_as_failed(results_df, group)


def create_timeout_result() -> Dict[str, Any]:
    """
    Create result for a timeout case.

    Returns:
        Dict[str, Any]: Timeout result data.
    """
    return {
        "cpog_message": "TIMEOUT",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }


def create_missing_instance_result() -> Dict[str, Any]:
    """
    Create result for missing instance case.

    Returns:
        Dict[str, Any]: Missing instance result data.
    """
    return {
        "cpog_message": "No instance provided",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }


def create_invalid_count_result() -> Dict[str, Any]:
    """
    Create result for invalid count value case.

    Returns:
        Dict[str, Any]: Invalid count result data.
    """
    return {
        "cpog_message": "Invalid count value",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }


def create_cpog_error_result(message: str) -> Dict[str, Any]:
    """
    Create result for a CPOG-related error.

    Args:
        message (str): Error message.

    Returns:
        Dict[str, Any]: CPOG error result.
    """
    return {
        "cpog_message": message,
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }


def create_processing_failed_result() -> Dict[str, Any]:
    """
    Create result for a generic processing failure.

    Returns:
        Dict[str, Any]: Processing failed result.
    """
    return {
        "cpog_message": "Processing failed",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }
