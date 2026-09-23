"""Writing results safely: other jobs may be reading the same file."""
import os
from pathlib import Path

import pandas as pd


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file, then rename: readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
