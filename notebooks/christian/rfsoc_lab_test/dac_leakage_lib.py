"""Analysis helpers for the RFSoC DAC spectral-leakage measurement.

The RFSoC emits a comb of tones (every 16 channels). Combined with
injected noise it is recorded as an auto-correlation in ``data/3``.
These helpers locate the comb, characterize the noise floor / dynamic
range, and stack the tone wings to measure spillover into neighbouring
channels.
"""

import h5py
import numpy as np

N_CHAN = 1024
COMB_START = 128
COMB_STEP = 16
COMB_STOP = 944  # inclusive
BAND = (120, 950)  # in-band channels used for floor statistics
DF_MHZ = 250.0 / N_CHAN  # channel width, ~0.2441 MHz


def comb_channels(start=COMB_START, step=COMB_STEP, stop=COMB_STOP):
    """Channel indices of the comb tones (``stop`` inclusive)."""
    return np.arange(start, stop + 1, step)


def load_auto(path, key="3"):
    """Time-averaged auto-correlation from one HDF5 file.

    Returns ``(spec, freqs, n_acc)`` with ``spec`` shape ``(1024,)``.
    """
    with h5py.File(path, "r") as f:
        data = f[f"data/{key}"][:]  # (n_acc, 1024)
        freqs = f["header/freqs"][:]
    return data.mean(axis=0), freqs, data.shape[0]


def in_band_mask(n_chan=N_CHAN, band=BAND):
    """Boolean mask, True for channels inside ``band``."""
    lo, hi = band
    mask = np.zeros(n_chan, dtype=bool)
    mask[lo:hi] = True
    return mask


def detect_comb(spec, rel_threshold=100.0, band=BAND):
    """Channels exceeding ``rel_threshold`` x the in-band median.

    Verified to return exactly ``comb_channels()`` on all four files.
    """
    band_mask = in_band_mask(spec.size, band)
    med = np.median(spec[band_mask])
    return np.where((spec > rel_threshold * med) & band_mask)[0]


def off_tone_mask(spec, tones, guard=2, band=BAND):
    """In-band channels not within ``guard`` of any tone."""
    mask = in_band_mask(spec.size, band)
    for t in tones:
        mask[max(0, t - guard) : t + guard + 1] = False
    return mask


def noise_floor(spec, tones, guard=2, band=BAND):
    """Median power of the off-tone, in-band channels."""
    return np.median(spec[off_tone_mask(spec, tones, guard, band)])


def tone_amplitude(spec, tones):
    """Mean power in the tone channels."""
    return spec[tones].mean()


def dynamic_range(spec, tones, guard=2, band=BAND):
    """Mean tone amplitude divided by the noise floor."""
    floor = noise_floor(spec, tones, guard, band)
    return tone_amplitude(spec, tones) / floor


def stacked_profile(spec, tones, half_width=8):
    """Stack the wings of all comb tones, normalized per tone.

    Each tone's window ``spec[t-hw : t+hw+1]`` is divided by its own
    tone-channel value ``spec[t]`` and averaged across tones.

    Returns ``(offsets, mean, err)``:
      offsets : integer channel offsets ``-hw .. hw``
      mean    : mean normalized power at each offset
      err     : standard error of the mean across tones
    """
    hw = half_width
    offsets = np.arange(-hw, hw + 1)
    windows = np.array(
        [spec[t - hw : t + hw + 1] / spec[t] for t in tones]
    )  # (n_tones, 2*hw+1)
    mean = windows.mean(axis=0)
    err = windows.std(axis=0, ddof=1) / np.sqrt(windows.shape[0])
    return offsets, mean, err


def adjacent_excess(spec, tones, half_width=8, near=1):
    """Excess power in the +/-``near`` channels over the far wing.

    Compares the immediately adjacent (+/-``near``) channels to the
    in-gap far wing (offsets ``near+1 .. half_width`` on both sides),
    both measured from the stacked, per-tone-normalized profile with the
    SAME mean estimator. Using a common estimator cancels the
    mean-vs-median bias of the noise, so a genuine adjacent-channel
    leakage shows up as a positive ``diff``.

    Returns ``(near_mean, far_mean, diff, sem)``:
      near_mean : mean normalized power at offset +/-``near``
      far_mean  : mean normalized power over the far-wing offsets
      diff      : ``near_mean - far_mean`` (the adjacent-channel excess)
      sem       : standard error of ``diff``
    """
    offsets, mean, err = stacked_profile(spec, tones, half_width)
    c = int(np.where(offsets == 0)[0][0])
    near_idx = [c - near, c + near]
    far_idx = [c + d for d in range(near + 1, half_width + 1)]
    far_idx += [c - d for d in range(near + 1, half_width + 1)]
    near_mean = float(np.mean(mean[near_idx]))
    far_mean = float(np.mean(mean[far_idx]))
    near_sem = 0.5 * float(np.hypot(err[c - near], err[c + near]))
    far_sem = float(np.std(mean[far_idx], ddof=1)) / np.sqrt(len(far_idx))
    sem = float(np.hypot(near_sem, far_sem))
    return near_mean, far_mean, near_mean - far_mean, sem
