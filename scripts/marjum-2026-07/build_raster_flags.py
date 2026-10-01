"""Build derived/beam/raster_flags_v0001.npz: bad beam-scan measurements.

Exploratory, single-study code. It flags measurements in the 07-17 beam-scan
raster that are discontinuous in time and strongly discrepant from a beam
model, so beam fits can exclude them. The flags add to the existing masks (the
gap-ratio dropout mask, reversal glitches, and the opening el~0 slew), which
are unchanged.

Reference model: the v0006 saved empirical PCA beam (commanded pointing). It is
used only through quantities a smooth beam cannot absorb: a broadband level
shared by all teeth at one instant, and single-sample jumps.

Sample flags (all teeth at that instant):
  bit 1 LEVEL: at least 21 of the 42 fitting teeth usable, and
        |ln k| > 0.35, where k is the least-squares level of data to model
        across those teeth. The MAD of ln k is 0.062, so 0.35 is ~5.7 sigma.
        These are broadband dropouts and spikes.
  bit 2 MOSTLY_MASKED: fewer than 21 fitting teeth pass the gap-ratio mask;
        the survivors share the same dropout.
  bit 4 PAD: time-adjacent (< 1.2 s) to a LEVEL/MOSTLY_MASKED sample and
        itself has |ln k| > 0.2 or fewer than 21 usable teeth.
Per-tooth flags (after sample flags): bit 8 SPIKE. The data jump from the
mean of both time-neighbours by more than 50% of the local level beyond the
model's own jump, and |data - model| > 6 MAD for that tooth.

The opening slew is never flagged here; it stays excluded by its own rule.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / 'marjum-2026-07/data').is_dir())
CAMPAIGN = ROOT / 'marjum-2026-07'
REFERENCE = CAMPAIGN / 'derived/beam/empirical_raster_v0006/pca/diagnostics.npz'
MIN_TEETH, LEVEL, PAD_LEVEL, CONTIGUOUS_S = 21, 0.35, 0.2, 1.2
SPIKE_JUMP, SPIKE_MAD, FLOOR_FRACTION = 0.5, 6.0, 0.02
BITS = {'LEVEL': 1, 'MOSTLY_MASKED': 2, 'PAD': 4, 'SPIKE': 8}


def main(output):
    z = np.load(REFERENCE)
    D, G, T, M = z['data'], z['good'].astype(bool), z['t'], z['empirical_exported']
    fit = z['fit_channels'].astype(bool)
    slew = z['el0_slew'].astype(bool)
    nf, n = D.shape

    usable = G[fit] & (M[fit] > 0)
    k = np.sum(np.where(usable, D[fit]*M[fit], 0), 0) / np.maximum(np.sum(np.where(usable, M[fit]**2, 0), 0), 1e-30)
    teeth = usable.sum(0)
    ln_k = np.log(np.clip(k, 1e-6, None))
    level = ~slew & (teeth >= MIN_TEETH) & (np.abs(ln_k) > LEVEL)
    masked = ~slew & (teeth < MIN_TEETH)
    core = level | masked
    pad = np.zeros(n, bool)
    for i in np.flatnonzero(core):
        for j in (i-1, i+1):
            if 0 <= j < n and not core[j] and not slew[j] and abs(T[j]-T[i]) < CONTIGUOUS_S \
                    and (teeth[j] < MIN_TEETH or abs(ln_k[j]) > PAD_LEVEL):
                pad[j] = True
    sample_bits = (BITS['LEVEL']*level + BITS['MOSTLY_MASKED']*masked + BITS['PAD']*pad).astype(np.uint8)
    sample_flag = sample_bits > 0

    G2 = G & ~sample_flag[None, :]
    local = np.full_like(D, np.nan)
    local[:, 1:-1] = 0.5*(np.abs(D[:, :-2]) + np.abs(D[:, 2:]))
    floor = FLOOR_FRACTION*np.nanpercentile(np.where(G2, D, np.nan), 99, axis=1)[:, None]

    def jump(x):
        j = np.full_like(x, np.nan)
        j[:, 1:-1] = x[:, 1:-1] - 0.5*(x[:, :-2] + x[:, 2:])
        return j/np.maximum(local, floor)

    neighbours = np.zeros_like(G2)
    step = np.diff(T) < CONTIGUOUS_S
    neighbours[:, 1:-1] = G2[:, :-2] & G2[:, 2:] & step[:-1][None, :] & step[1:][None, :]
    R = D - M
    mad = np.array([1.4826*np.median(np.abs(R[f][G2[f]])) for f in range(nf)])
    with np.errstate(invalid='ignore'):
        spike = G2 & neighbours & (np.abs(jump(D) - jump(M)) > SPIKE_JUMP) & (np.abs(R) > SPIKE_MAD*mad[:, None])
    tooth_bits = (BITS['SPIKE']*spike).astype(np.uint8)

    tt = pd.to_datetime(T, unit='s', utc=True)
    commit = subprocess.run(['git', '-C', str(SOURCE.parent), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(SOURCE.parent), 'status', '--porcelain', '--', str(SOURCE.parent)],
                                capture_output=True, text=True).stdout.strip())
    provenance = dict(
        product='raster_flags', version='v0001', campaign='marjum-2026-07',
        generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        generator=str(SOURCE.relative_to(ROOT)), generator_commit=commit, generator_dirty=dirty,
        reference=str(REFERENCE.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        keys='t: correlator time_best (s since 1970 UTC), as in the beam diagnostics; channels: correlator '
             'channel numbers (freq = channel*250/1024 MHz)',
        bits=BITS, parameters=dict(min_teeth=MIN_TEETH, level_abs_ln=LEVEL, pad_abs_ln=PAD_LEVEL,
                                   contiguous_s=CONTIGUOUS_S, spike_jump=SPIKE_JUMP, spike_mad=SPIKE_MAD,
                                   floor_fraction_of_p99=FLOOR_FRACTION),
        ln_k_mad=float(1.4826*np.median(np.abs(ln_k[(teeth >= MIN_TEETH) & ~slew] - np.median(ln_k[(teeth >= MIN_TEETH) & ~slew])))),
        summary=dict(samples=int(n), sample_flags=int(sample_flag.sum()), level=int(level.sum()),
                     mostly_masked=int(masked.sum()), pad=int(pad.sum()),
                     tooth_spikes=int(spike.sum()), tooth_spikes_on_fitting_teeth=int(spike[fit].sum()),
                     sample_flags_by_minute=pd.Series(tt[sample_flag].strftime('%H:%M')).value_counts().sort_index().to_dict()),
        use='exclude: good &= ~sample_flag[None, :] & ~tooth_flag, on top of the gap-ratio, glitch and slew masks',
        caveats=['The reference is a fitted beam; flags use only a broadband level and single-sample jumps, '
                 'which a smooth beam cannot absorb.',
                 'Near model nulls a sharp model feature can mimic a data jump; the spike rule compares the data '
                 'jump with the model jump for that reason.',
                 'Defined on the 07-17 raster samples of the v0006 diagnostics (6901 samples, 46 teeth).'])
    np.savez_compressed(output, t=T, channels=z['channels'], freqs=z['freqs'],
                        sample_flag=sample_flag, sample_bits=sample_bits, tooth_flag=spike, tooth_bits=tooth_bits,
                        level_k=k, usable_teeth=teeth, provenance=np.array(json.dumps(provenance, indent=1)))
    print(json.dumps(provenance['summary'], indent=1))
    print('wrote', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'derived/beam/raster_flags_v0001.npz')
    main(parser.parse_args().output)
