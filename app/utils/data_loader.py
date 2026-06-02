"""
Data loading utilities.
Supports CSV, Excel (.xlsx / .xls), and JSON inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


def load_dataset(source: Union[str, Path, bytes], filename: str = "") -> pd.DataFrame:
    """
    Load a dataset from a file path or raw bytes.

    Args:
        source:   File path (str / Path) OR raw file bytes.
        filename: Original filename — used to infer format when source is bytes.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the file format is unsupported.
    """
    ext = _infer_extension(source, filename)

    if isinstance(source, (str, Path)):
        path = Path(source)
        logger.info(f"Loading dataset from {path} (format: {ext})")
        return _read_by_ext(ext, path=path)
    else:
        import io
        buf = io.BytesIO(source)
        logger.info(f"Loading dataset from bytes (format: {ext})")
        return _read_by_ext(ext, buf=buf)


def _infer_extension(source, filename: str) -> str:
    if filename:
        return Path(filename).suffix.lower().lstrip(".")
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower().lstrip(".")
    raise ValueError("Cannot infer file format — supply a filename.")


def _read_by_ext(ext: str, *, path: Path | None = None, buf=None) -> pd.DataFrame:
    target = path or buf

    if ext == "csv":
        return pd.read_csv(target, low_memory=False)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(target)
    elif ext == "json":
        return pd.read_json(target)
    else:
        raise ValueError(
            f"Unsupported file format: .{ext}. Supported: csv, xlsx, xls, json"
        )
