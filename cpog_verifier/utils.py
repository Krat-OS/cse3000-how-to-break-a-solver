import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

print_lock: threading.Lock = threading.Lock()
remaining_count_lock: threading.Lock = threading.Lock()
remaining_count: int = 0
batch_executor: Optional[ThreadPoolExecutor] = None
should_terminate: threading.Event = threading.Event()
active_processes: List[subprocess.Popen] = []
processes_lock: threading.Lock = threading.Lock()

def get_memory_usage_gb() -> float:
    """Return current memory usage of the process in gigabytes.

    Returns:
        float: Memory usage in GB.
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def monitor_memory_usage(memory_limit_gb: float, check_interval: float = 1.0) -> None:
    """Monitor memory usage in a separate thread and terminate batch if limit exceeded.

    Args:
        memory_limit_gb: Maximum allowed memory usage in GB.
        check_interval: Time between memory checks in seconds.
    """
    while not should_terminate.is_set():
        current_usage = get_memory_usage_gb()
        if current_usage > memory_limit_gb:
            with print_lock:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Memory limit exceeded: {current_usage:.2f}GB > {memory_limit_gb:.2f}GB")
                if batch_executor:
                    batch_executor.shutdown(wait=False, cancel_futures=True)
                clear_ram()
        time.sleep(check_interval)

def print_progress(message: str) -> None:
    """Print a timestamped progress message in a thread-safe manner.

    Args:
        message: The message to be printed.
    """
    with print_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

def update_remaining_count(delta: int = -1) -> int:
    """Update the global count of remaining instances thread-safely.

    Args:
        delta: Value to add to remaining count.

    Returns:
        int: Updated count of remaining instances.
    """
    global remaining_count
    with remaining_count_lock:
        remaining_count += delta
        return remaining_count

def setup_instance_workspace(instance_path: Path, verifier_dir: Path, thread_id: str) -> Path:
    """Create and setup a temporary workspace for processing an instance with thread isolation.

    Args:
        instance_path: Path to the CNF instance file.
        verifier_dir: Path to the directory containing verification tools.
        thread_id: Unique identifier for the thread.

    Returns:
        Path: Path to the created workspace directory.
    """
    instance_name: str = instance_path.stem
    workspace: Path = Path(tempfile.gettempdir()) / f"cpog_workspace_{thread_id}_{instance_name}"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    new_verifier_dir: Path = workspace / "cpog"
    shutil.copytree(verifier_dir, new_verifier_dir)

    for executable in new_verifier_dir.glob("*"):
        if not executable.name.startswith('.'):
            executable.chmod(0o755)

    return workspace

def cleanup_workspace(workspace: Path) -> None:
    """Remove a temporary workspace directory.

    Args:
        workspace: Path to the workspace directory to be cleaned up.
    """
    try:
        if workspace.exists():
            shutil.rmtree(workspace)
            print_progress(f"Cleaned up workspace: {workspace}")
    except Exception as e:
        print_progress(f"Error cleaning up workspace {workspace}: {e}")

def cleanup_on_exit():
    """Cleanup function registered to run when the program exits."""
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
    """Force clear RAM by terminating subprocesses and invoking garbage collection."""
    gc.collect()
    process: psutil.Process = psutil.Process()

    for child in process.children(recursive=True):
        try:
            child.terminate()
        except Exception:
            pass

    _, alive = psutil.wait_procs(process.children(recursive=True), timeout=3)
    for proc in alive:
        try:
            proc.kill()
        except Exception:
            pass

def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
    env: Optional[dict] = None
) -> Tuple[int, str, str]:
    """Execute a shell command with timeout and optional environment variables.

    Args:
        cmd: Command to execute as list of strings.
        cwd: Working directory for command execution.
        timeout: Maximum execution time in seconds.
        env: Environment variables to set for the command.

    Returns:
        Tuple[int, str, str]: Return code, stdout, and stderr from the command.
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

        # Track the process
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
            # Remove process from tracking
            with processes_lock:
                if process in active_processes:
                    active_processes.remove(process)

def verify_single_instance(
    cnf_path: Path,
    workspace: Path,
    thread_id: str,
    timeout: int = 300
) -> Tuple[bool, Optional[str], int]:
    """Verify a single CNF instance using CPOG with thread isolation.

    Args:
        cnf_path: Path to CNF file.
        workspace: Path to workspace directory.
        thread_id: Unique identifier for the thread.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple[bool, Optional[str], int]: Success status, error message (if any), and model count.
    """
    verifier_dir: Path = workspace / "cpog"
    d4nnf_path: Path = workspace / f"temp_{thread_id}.d4nnf"
    cpog_path: Path = workspace / f"temp_{thread_id}.cpog"

    try:
        # Generate D4NNF
        returncode, _, stderr = run_command([
            str(verifier_dir / "d4"),
            str(cnf_path),
            "-dDNNF",
            f"-out={str(d4nnf_path)}"
        ], timeout=timeout)
        if returncode != 0:
            return False, f"D4NNF generation failed: {stderr}", 0

        # Generate CPOG with updated environment
        env = environ.copy()
        env["PATH"] = f"{str(verifier_dir)}:{env.get('PATH', '')}"

        returncode, stdout, stderr = run_command([
            str(verifier_dir / "cpog-gen"),
            str(cnf_path),
            str(d4nnf_path),
            str(cpog_path)
        ], timeout=timeout, env=env)

        if "Compiled formula unsatisfiable.  Cannot verify" in stdout:
            return False, "UNSAT", 0
        if returncode != 0:
            return False, f"CPOG generation failed: {stderr}", 0

        # Verify CPOG
        returncode, stdout, stderr = run_command([
            str(verifier_dir / "cpog-check"),
            str(cnf_path),
            str(cpog_path)
        ], timeout=timeout)

        if returncode != 0:
            return False, f"CPOG verification failed: {stderr}", 0

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
    """Process a group of rows sharing the same instance_path with thread isolation.

    Args:
        group: DataFrame containing rows with the same instance_path.
        verifier_dir: Directory containing verification tools.
        timeout: Maximum execution time in seconds.

    Returns:
        List[Tuple[int, Dict[str, Any]]]: List of results for each row in the group.
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
            if pd.isna(row["count_value"]) or not float(row["count_value"]).is_integer():
                results.append((idx, create_invalid_count_result()))
                continue

            result: Dict[str, Any] = {
                "cpog_message": "NO ERROR" if error is None else error,
                "cpog_count": cpog_count,
                "count_matches": (
                    (cpog_count > 0 and abs(cpog_count - row["count_value"]) < 1e-6) or
                    (cpog_count == 0 and row["count_value"] == 0 and
                     row["satisfiability"] == "UNSATISFIABLE" and error == "UNSAT")
                ),
                "verified": verified
            }
            results.append((idx, result))

    except Exception as e:
        error_result: Dict[str, Any] = {
            "cpog_message": f"ERROR: {str(e)}",
            "cpog_count": 0,
            "count_matches": False,
            "verified": False
        }
        results = [(idx, error_result) for idx in group.index]
    finally:
        cleanup_workspace(workspace)
        remaining: int = update_remaining_count()
        print_progress(
            f"Completed verification for {instance_path.name}. Remaining instance groups: {remaining}"
        )

    return results

def _setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        global batch_executor, should_terminate

        with print_lock:
            print(f"\nReceived signal {signum}, initiating graceful shutdown...")

        should_terminate.set()

        if batch_executor:
            batch_executor.shutdown(wait=False, cancel_futures=True)

        clear_ram()
        exit(128 + signum)

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
    """Verify multiple CNF instances using CPOG with batching and thread isolation."""
    global batch_executor, should_terminate
    
    # Set up signal handlers and cleanup
    _setup_signal_handlers()
    atexit.register(cleanup_on_exit)
    
    should_terminate.clear()
    verifier_dir = Path(verifier_dir)
    results_df: pd.DataFrame = df.copy()

    try:
        grouped = df.groupby("instance_path", group_keys=True)
        groups: List[Tuple[str, pd.DataFrame]] = list(grouped)
        total_groups: int = len(groups)

        global remaining_count
        remaining_count = total_groups

        print_progress(f"Starting verification of {total_groups} unique instances with {max_workers} workers")

        # Start memory monitoring thread
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
                print_progress("Termination requested, stopping after current batch")
                break

            batch_count += 1
            current_batch = groups[i:i + batch_size]
            print_progress(f"Processing batch {batch_count} with {len(current_batch)} instance groups")

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
                        try:
                            results = future.result()
                            for idx, result in results:
                                for key, value in result.items():
                                    if key == "cpog_count":
                                        value = int(value)
                                    results_df.at[idx, key] = value
                        except Exception as e:
                            group_name = future_to_group[future]
                            print_progress(f"Error processing group {group_name}: {e}")
                            # Handle the error for this group but continue processing
                            mark_group_as_failed(results_df, grouped.get_group(group_name))

            except Exception as e:
                print_progress(f"Batch {batch_count} interrupted: {str(e)}")
                # Mark remaining instances in batch as failed
                mark_remaining_as_failed(results_df, future_to_group, grouped)
            finally:
                batch_executor = None
                clear_ram()
                print_progress(f"Batch {batch_count} complete")

    except KeyboardInterrupt:
        print_progress("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        print_progress(f"Unexpected error: {e}")
        raise
    finally:
        should_terminate.set()
        if memory_monitor.is_alive():
            memory_monitor.join(timeout=1.0)
        
        # Additional cleanup
        if batch_executor:
            batch_executor.shutdown(wait=False, cancel_futures=True)
            batch_executor = None
        clear_ram()

    print_progress(f"Verification complete. Processed batches: {batch_count}")
    return results_df

def mark_group_as_failed(results_df: pd.DataFrame, group: pd.DataFrame) -> None:
    """Mark all instances in a group as failed."""
    error_result = {
        "cpog_message": "Processing failed",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }
    for idx in group.index:
        for key, value in error_result.items():
            results_df.at[idx, key] = value

def mark_remaining_as_failed(
    results_df: pd.DataFrame,
    future_to_group: Dict,
    grouped: pd.core.groupby.DataFrameGroupBy
) -> None:
    """Mark all remaining unprocessed instances as failed."""
    for future, group_name in future_to_group.items():
        if not future.done():
            group = grouped.get_group(group_name)
            mark_group_as_failed(results_df, group)

def create_timeout_result() -> Dict[str, Any]:
    """Create a result dictionary for timeout case.

    Returns:
        Dict[str, Any]: Dictionary containing timeout result data.
    """
    return {
        "cpog_message": "TIMEOUT",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def create_missing_instance_result() -> Dict[str, Any]:
    """Create a result dictionary for missing instance case.

    Returns:
        Dict[str, Any]: Dictionary containing missing instance result data.
    """
    return {
        "cpog_message": "No instance provided",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }

def create_invalid_count_result() -> Dict[str, Any]:
    """Create a result dictionary for invalid count value case.

    Returns:
        Dict[str, Any]: Dictionary containing invalid count result data.
    """
    return {
        "cpog_message": "Invalid count value",
        "cpog_count": 0,
        "count_matches": False,
        "verified": False
    }