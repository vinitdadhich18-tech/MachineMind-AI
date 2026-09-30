"""
test_csv_validation.py - Comprehensive unit tests for client-side CSV pre-validation logic.
"""

import pytest
import numpy as np

from frontend.utils.csv_validation import validate_snapshot_csv

# DEMO / SYNTHETIC test data generators for unit testing validation bounds


def generate_synthetic_csv(rows=20480, cols=4, sep=",", with_header=False, add_nan=False, add_text=False):
    """DEMO / SYNTHETIC CSV generator helper for unit tests."""
    arr = np.random.randn(rows, cols).astype(np.float64)

    if add_nan:
        arr[10, 1] = np.nan

    lines = []
    if with_header:
        lines.append("Channel_1,Channel_2,Channel_3,Channel_4")

    for r in arr:
        if sep == r"\s+" or sep == " ":
            lines.append("  ".join(str(val) for val in r))
        elif sep == "\t":
            lines.append("\t".join(str(val) for val in r))
        else:
            lines.append(",".join(str(val) for val in r))

    if add_text:
        lines.append("invalid_text_line,1,2,3")

    return "\n".join(lines).encode("utf-8")


def test_valid_csv_file():
    """Verifies valid 20480x4 CSV file passes pre-validation."""
    content = generate_synthetic_csv(rows=20480, cols=4)
    valid, err, details, df = validate_snapshot_csv(content, "test.csv")
    assert valid is True
    assert err is None
    assert details["rows"] == 20480
    assert details["cols"] == 4
    assert df is not None


def test_header_row_tolerated():
    """Verifies that an optional single header row is tolerated and dropped cleanly."""
    content = generate_synthetic_csv(rows=20480, cols=4, with_header=True)
    valid, err, details, df = validate_snapshot_csv(content, "test.csv")
    assert valid is True
    assert details["rows"] == 20480


def test_tab_delimited_file():
    """Verifies tab-delimited files (.tsv / .txt) pass pre-validation."""
    content = generate_synthetic_csv(rows=20480, cols=4, sep="\t")
    valid, err, details, df = validate_snapshot_csv(content, "snapshot.tsv")
    assert valid is True
    assert details["rows"] == 20480


def test_invalid_row_count():
    """Verifies wrong row count (e.g. 1000 rows) fails validation cleanly."""
    content = generate_synthetic_csv(rows=1000, cols=4)
    valid, err, details, df = validate_snapshot_csv(content, "short.csv")
    assert valid is False
    assert "Expected exactly 20480" in err
    assert details["actual_rows"] == 1000


def test_invalid_column_count():
    """Verifies wrong column count (e.g. 2 columns) fails validation cleanly."""
    content = generate_synthetic_csv(rows=20480, cols=2)
    valid, err, details, df = validate_snapshot_csv(content, "wrong_cols.csv")
    assert valid is False
    assert "4 vibration data columns" in err


def test_nan_or_non_finite_values():
    """Verifies NaN values trigger clean validation error."""
    content = generate_synthetic_csv(rows=20480, cols=4, add_nan=True)
    valid, err, details, df = validate_snapshot_csv(content, "nan.csv")
    assert valid is False
    assert "non-finite values" in err
    assert details["nan_count"] > 0


def test_unsupported_file_extension():
    """Verifies unsupported file extensions (.pdf, .exe) fail validation cleanly."""
    content = generate_synthetic_csv(rows=20480, cols=4)
    valid, err, details, df = validate_snapshot_csv(content, "document.pdf")
    assert valid is False
    assert "Unsupported file extension" in err
