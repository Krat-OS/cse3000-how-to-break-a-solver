from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd

def get_csv_files(directory: Path) -> List[Path]:
    """Get all CSV files from a directory."""
    return list(directory.glob("*.csv"))

def read_csv_to_dataframe(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a single CSV file into a DataFrame."""
    try:
        return pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

def get_cnf_paths(instance_col: pd.Series, cnf_dir: Path) -> List[Path]:
    """Extract CNF file paths from instance column."""
    instance_paths = instance_col.unique()
    return [cnf_dir / Path(path).name for path in instance_paths if pd.notna(path)]

def read_cnf_file(cnf_path: Path) -> Optional[str]:
    """Read a single CNF file."""
    try:
        return cnf_path.read_text()
    except Exception as e:
        print(f"Error reading CNF file {cnf_path}: {e}")
        return None

def get_cnf_contents(cnf_paths: List[Path]) -> Dict[str, str]:
    """Get contents of all CNF files."""
    cnf_contents = {}
    for path in cnf_paths:
        if path.exists():
            content = read_cnf_file(path)
            if content is not None:
                cnf_contents[path.name] = content
        else:
            print(f"CNF file not found: {path}")
    return cnf_contents

def process_single_csv(csv_path: Path, cnf_dir: Path) -> Optional[Tuple[pd.DataFrame, Dict[str, str]]]:
    """Process a single CSV file and its associated CNF files."""
    df = read_csv_to_dataframe(csv_path)
    if df is None:
        return None

    cnf_paths = get_cnf_paths(df['instance'], cnf_dir)
    cnf_contents = get_cnf_contents(cnf_paths)

    return df, cnf_contents

def process_results(directory: str | Path) -> Dict[str, Tuple[pd.DataFrame, Dict[str, str]]]:
    """
    Process CSV files and their associated CNF files from a directory.

    Args:
        directory: Path to directory containing CSV files and 'instances/cnf' subdirectory

    Returns:
        Dictionary mapping CSV names to tuples of (DataFrame, CNF_contents)
        where CNF_contents maps CNF filenames to their contents

    Example:
        >>> results = process_results("path/to/results")
        >>> df, cnf_files = results['experiment_name']
        >>> print(df.head())
        >>> print(len(cnf_files))
    """
    base_dir = Path(directory)
    cnf_dir = base_dir / "instances" / "cnf"

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    results = {}
    csv_files = get_csv_files(base_dir)

    for csv_path in csv_files:
        result = process_single_csv(csv_path, cnf_dir)
        if result is not None:
            results[csv_path.stem] = result

    return results
