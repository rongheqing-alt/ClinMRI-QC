"""Append one QC record to a growing CSV file.

Creates the file with the canonical header on first call.
Subsequent calls append rows without touching the header.

Usage
-----
    from quickbrain.append_csv import append_csv_record

    append_csv_record(record, 'qc_results.csv')
"""

import csv
from pathlib import Path

from quickbrain.schema import ALL_COLUMNS


def append_csv_record(record: dict, csv_path: str) -> None:
    """Append one QC record to a growing CSV file.

    Parameters
    ----------
    record   : dict returned by build_qc_record().
    csv_path : path to the target CSV file (created if absent).

    Notes
    -----
    Extra keys in `record` beyond ALL_COLUMNS are silently ignored.
    Missing keys are written as empty strings.
    The header is written only once — on the first call when the file is empty.
    """
    csv_path     = Path(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS,
                                extrasaction='ignore', restval='')
        if write_header:
            writer.writeheader()
        writer.writerow(record)
