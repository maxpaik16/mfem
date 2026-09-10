#!/usr/bin/env python3
import sys
import argparse
from typing import TextIO


def iter_clean_lines(stream: TextIO):
    for lineno, raw in enumerate(stream, 1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        yield lineno, line


def parse_header(parts, lineno):
    """
    Accept either:
      rows cols nnz
    or:
      row_start rows col_start nnz
    based on the user's example: '0 180 0 7901'

    Returns inferred number of rows.
    """
    if len(parts) == 3:
        rows, cols, nnz = map(int, parts)
        return rows
    if len(parts) == 4:
        row_start, rows, col_start, nnz = map(int, parts)
        return rows
    raise ValueError(f"line {lineno}: invalid header format: expected 3 or 4 fields, got {len(parts)}")


def compute_row_sums(stream: TextIO):
    row_sums = {}
    nrows = None
    header_seen = False

    for lineno, line in iter_clean_lines(stream):
        parts = line.split()

        if not header_seen:
            nrows = parse_header(parts, lineno)
            header_seen = True
            continue

        if len(parts) != 3:
            raise ValueError(f"line {lineno}: expected data line with 3 fields 'row col value', got {len(parts)}")

        try:
            row = int(parts[0])
            _col = int(parts[1])
            value = float(parts[2])
        except ValueError as e:
            raise ValueError(f"line {lineno}: failed to parse data line '{line}': {e}") from e

        row_sums[row] = row_sums.get(row, 0.0) + value

    if not header_seen:
        raise ValueError("no header line found")

    return nrows, row_sums


def main():
    parser = argparse.ArgumentParser(description="Compute row sums from a sparse matrix text file.")
    parser.add_argument(
        "file",
        nargs="?",
        help="Input file path, reads stdin if omitted"
    )
    args = parser.parse_args()

    try:
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                nrows, sums = compute_row_sums(f)
        else:
            nrows, sums = compute_row_sums(sys.stdin)

        for row in range(nrows):
            print(sums.get(row, 0.0))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()