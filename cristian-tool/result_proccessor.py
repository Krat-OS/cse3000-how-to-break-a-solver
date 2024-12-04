from pathlib import Path
from typing import Optional
import pandas as pd

def get_cnf_path(path: str, base_dir: Path) -> Path:
    """Construct path to CNF file in instances directory.

    Args:
        path: Original path string from results
        base_dir: Base directory containing instances folder

    Returns:
        Path: Full path to CNF file
    """
    return base_dir / "instances" / "cnf" / Path(path).name

def read_cnf_file(cnf_path: Path) -> Optional[str]:
    """Read CNF file content and format as single line.

    Args:
        cnf_path: Path to CNF file

    Returns:
        str: File content as single line, None if error
    """
    try:
        content = cnf_path.read_text()
        return content.replace('\n', ' ')
    except Exception as e:
        print(f"Error reading CNF file {cnf_path}: {e}")
        return None

def process_results(csv_path: Path | str) -> pd.DataFrame:
    """Process results CSV and load associated CNF files.

    Args:
        csv_path: Path to results CSV file

    Returns:
        DataFrame: Results with CNF contents added
    """
    csv_path = Path(csv_path)
    base_dir = csv_path.parent
    df = pd.read_csv(csv_path)

    df['instance_path'] = df['instance']
    df['instance'] = None

    for idx, row in df.iterrows():
        if pd.isna(row['instance_path']):
            continue

        cnf_path = get_cnf_path(row['instance_path'], base_dir)
        if not cnf_path.exists():
            print(f"Instance not found: {cnf_path}")
            continue

        content = read_cnf_file(cnf_path)
        if content is not None:
            df.at[idx, 'instance'] = content

    return df
