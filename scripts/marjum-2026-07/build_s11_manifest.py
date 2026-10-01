"""Write marjum-2026-07/data/s11/manifest.csv from the raw field S11 sweeps.

The ``ants11_*`` / ``recs11_*.h5`` files were written by ``vna_writer``
and copied off the panda in one batch, so their filename stamps are the
copy time (mostly 07-17 19:08 UTC), not the measurement time. This reads
the measurement time from ``header.attrs["metadata_snapshot_unix"]`` and
records, per file, what a consumer needs to pick sweeps without opening
them: mode, DUT traces present, zero-filled traces (cmt_vna issue #54),
and a SHA-256 so the raw file can be re-verified.

One row per file, sorted by measurement time. Columns:

- ``file``: basename under ``data/s11/``.
- ``mode``: ``ant`` or ``rec`` (``header.attrs["mode"]``).
- ``measured_utc``: ISO 8601 UTC, from ``metadata_snapshot_unix``.
- ``measured_unix``: float seconds, same source.
- ``run_started_unix``: ``run_started_at_unix``; empty when the writer
  left it at 0 (unset).
- ``duts``: ``;``-joined DUT keys under ``data/``, excluding the
  internal ``cal:VNA{O,S,L}`` traces.
- ``zero_filled``: ``;``-joined keys (DUTs or ``cal:VNA*``) containing an
  exact zero; empty when none.
- ``pre_campaign``: ``true`` when measured before the first correlator
  file (2026-07-12 13:31 UTC).
- ``bytes``, ``sha256``.

Usage: ``python build_s11_manifest.py`` (no arguments).
"""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
S11_DIR = ROOT / 'marjum-2026-07/data/s11'
CAMPAIGN_START = datetime(2026, 7, 12, 13, 31, 46, tzinfo=timezone.utc)
COLUMNS = ['file', 'mode', 'measured_utc', 'measured_unix', 'run_started_unix',
           'duts', 'zero_filled', 'pre_campaign', 'bytes', 'sha256']


def row(path):
    with h5py.File(path, 'r') as h:
        attrs = h['header'].attrs
        t = float(attrs['metadata_snapshot_unix'])
        started = float(attrs['run_started_at_unix'])
        keys = sorted(h['data'])
        zeros = [k for k in keys if np.any(h['data'][k][()] == 0)]
        mode = str(attrs['mode'])
    when = datetime.fromtimestamp(t, tz=timezone.utc)
    return {
        'file': path.name,
        'mode': mode,
        'measured_utc': when.isoformat(timespec='seconds'),
        'measured_unix': f'{t:.3f}',
        'run_started_unix': f'{started:.3f}' if started > 0 else '',
        'duts': ';'.join(k for k in keys if not k.startswith('cal:')),
        'zero_filled': ';'.join(zeros),
        'pre_campaign': str(when < CAMPAIGN_START).lower(),
        'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    rows = sorted((row(p) for p in S11_DIR.glob('*s11_*.h5')),
                  key=lambda r: float(r['measured_unix']))
    out = S11_DIR / 'manifest.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} files -> {out}')


if __name__ == '__main__':
    main()
