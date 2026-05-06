"""File readers — CSV and Excel, with sheet enumeration for Excel."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import io

import pandas as pd


def list_excel_sheets(file_bytes: bytes) -> List[str]:
    """Return sheet names for an Excel file."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names


def read_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
    header_row: int = 0,
) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame.

    For Excel, caller may pass sheet_name and header_row.
    For CSV, we try a few common encodings/separators.
    """
    name = filename.lower()
    buf = io.BytesIO(file_bytes)

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(buf, sheet_name=sheet_name, header=header_row)

    if name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else None  # sniff for csv
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, sep=sep, engine="python", encoding=enc,
                                   header=header_row)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError(f"Could not parse CSV {filename} with common encodings.")

    raise ValueError(
        f"Unsupported file type: {filename}. Use .csv, .tsv, .xlsx, .xls, or .xlsm."
    )
