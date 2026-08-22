from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import curve_fit

from eigsep_observing import io


def to_unix_time(value):
    """
    Convert a datetime string, datetime object, or Unix timestamp to Unix
    seconds (float).

    Strings without a timezone are interpreted as UTC.

    Accepted formats:
        "2026-07-17 06:00:00"
        "2026-7-17 6:00:00"
        "2026-07-17T06:00:00Z"
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Could not interpret time {value!r}. "
                    "Use a format such as '2026-07-17 06:00:00'."
                )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_time_from_name(fname: str) -> datetime:
    """
    Parse datetime from filename of form 'corr_YYYYMMDD_HHMMSS.h5'
    """
    stem = Path(fname).stem  # 'corr_20250922_160500'
    _, datestr, timestr = stem.split("_")  # ['corr', '20250922', '160500']
    return datetime.strptime(datestr + timestr, "%Y%m%d%H%M%S")


@dataclass
class EigsepData:

    data: dict[str, np.ndarray] = None
    acc_cnt: np.ndarray = None
    times: np.ndarray = None
    freq: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 250, num=1024, endpoint=False)
    )

    @classmethod
    def from_path(
        cls,
        path: Path,
        start_time: str = None,
        end_time: str = None,
        pacific_to_mountain: bool = True,
    ):
        """
        Create an EigsepData instance from a directory or a file.

        Parameters
        ----------
        path : Path
            The path to the directory or file.
        start_time : str
            The start time in the format "YYYYMMDD_HHMMSS" for filtering data.
            Only used if reading from a directory.
        end_time : str
            The end time in the format "YYYYMMDD_HHMMSS" for filtering data.
            Only used if reading from a directory.
        pacific_to_mountain : bool
            If True, convert times from Pacific to Mountain time by adding
            3600 seconds.

        Returns
        -------
        EigsepData

        """
        if path.is_dir():
            files = sorted(path.glob("corr*.h5"))
            times = [_parse_time_from_name(f.name) for f in files]
            if start_time:
                start_dt = datetime.strptime(start_time, "%Y%m%d_%H%M%S")
                files = [f for f, t in zip(files, times) if t >= start_dt]
                times = [_parse_time_from_name(f.name) for f in files]
            if end_time:
                end_dt = datetime.strptime(end_time, "%Y%m%d_%H%M%S")
                files = [f for f, t in zip(files, times) if t <= end_dt]
        elif path.is_file():
            files = [path]
        if not files:
            raise ValueError(f"No data files found in {path}.")

        data = {}
        acc_cnt = []
        times = []
        freq = None
        for f in files:
            try:
                d, hdr, metadata = io.read_hdf5(f)
            except Exception as e:
                warnings.warn(f"Failed to read {f}: {e}. Skipping this file.")
                continue
            for k, v in d.items():
                data[k] = data.get(k, []) + [v]
            acc_cnt.append(hdr["acc_cnt"])
            times.append(hdr["times"])
            if freq is None:
                freq = hdr["freqs"]
            elif not np.array_equal(freq, hdr["freqs"]):
                warnings.warn(
                    f"Frequency mismatch in {f}. Using first file's "
                    "frequency array. "
                )
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
        acc_cnt = np.concatenate(acc_cnt, axis=0)
        times = np.concatenate(times, axis=0)
        if pacific_to_mountain:
            times += 3600
        return cls(data=data, acc_cnt=acc_cnt, times=times, freq=freq)

    def slice(self, min_index, max_index):
        """
        Slice the data along the time axis.

        Parameters
        ----------
        min_index : int
            The minimum index (inclusive).
        max_index : int
            The maximum index (exclusive).

        Returns
        -------
        EigsepData
            A new EigsepData instance with the sliced data.

        """
        sliced_data = {k: v[min_index:max_index] for k, v in self.data.items()}
        sliced_acc_cnt = self.acc_cnt[min_index:max_index]
        sliced_times = self.times[min_index:max_index]
        return EigsepData(
            data=sliced_data,
            acc_cnt=sliced_acc_cnt,
            times=sliced_times,
            freq=self.freq,
        )


def extract_clean_pot_data_v2(az_pot, az_step, min_stable_samples=10, settle_samples=3):
    """
    Clean noisy potentiometer data by isolating stable plateaus, computing
    their medians, and linearly interpolating across motor transitions.

    Parameters
    ----------
    az_pot : np.ndarray
        Raw potentiometer azimuth readings, shape (nsamples,).
    az_step : np.ndarray
        Stepper-motor step count at each sample, shape (nsamples,).
    min_stable_samples : int
        Minimum consecutive samples with the same step value to count as a
        stable plateau.
    settle_samples : int
        Samples to discard at the start of each plateau to allow the motor
        to physically settle before taking the median.

    Returns
    -------
    clean_az_pot : np.ndarray
        Cleaned azimuth array with transitions replaced by linear ramps and
        plateaus replaced by their median values.
    """
    clean_az_pot = np.copy(az_pot)

    change_indices = np.where(np.diff(az_step) != 0)[0] + 1
    boundaries = np.concatenate(([0], change_indices, [len(az_step)]))

    plateaus = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if (end - start) >= min_stable_samples:
            safe_start = min(start + settle_samples, end - 1)
            plateau_median = np.median(az_pot[safe_start:end])
            clean_az_pot[start:end] = plateau_median
            plateaus.append((start, end, plateau_median))

    for i in range(len(plateaus) - 1):
        _, curr_end, curr_val = plateaus[i]
        next_start, _, next_val = plateaus[i + 1]
        if next_start > curr_end:
            ramp = np.linspace(curr_val, next_val, next_start - curr_end + 2)
            clean_az_pot[curr_end:next_start] = ramp[1:-1]

    if len(plateaus) > 0:
        if plateaus[0][0] > 0:
            clean_az_pot[: plateaus[0][0]] = plateaus[0][2]
        if plateaus[-1][1] < len(clean_az_pot):
            clean_az_pot[plateaus[-1][1] :] = plateaus[-1][2]

    return clean_az_pot


def calibrate_weak_arm(el_deg, az_deg, dpss_red):
    """
    Compute per-frequency scale factors to normalize the weak transmitter arm
    to the strong arm.

    This function is a dataset-specific workaround for the deployment-4 data
    where one polarization arm has lower coupling. It fits a Malus's-law
    (cos²) curve to the power measured at el=0 crossings to find the expected
    peak power for each arm, then returns the ratio strong/weak so the caller
    can rescale the weak-arm channels.

    Ideally this correction should not be needed for future datasets.

    Parameters
    ----------
    el_deg : np.ndarray
        Elevation angles in degrees, shape (nsamples,).
    az_deg : np.ndarray
        Azimuth angles in degrees, shape (nsamples,).
    dpss_red : np.ndarray
        DPSS-reduced power, shape (nsamples, nfreq). Even frequency indices
        are the strong arm; odd indices are the weak arm.

    Returns
    -------
    scale_factors : np.ndarray
        Multiplicative scale factors for weak-arm channels, shape (nfreq//2,).
    peak_powers : np.ndarray
        Fitted peak power at each frequency, shape (nfreq,). NaN where the
        fit failed.
    """
    crossings = np.where(np.diff(np.sign(el_deg)))[0]
    valid_crossings = [i for i in crossings if abs(el_deg[i]) < 10]

    az_at_0 = []
    power_at_0 = []
    for i in valid_crossings:
        el0, el1 = el_deg[i], el_deg[i + 1]
        az0, az1 = az_deg[i], az_deg[i + 1]
        p0, p1 = dpss_red[i], dpss_red[i + 1]
        t = (0.0 - el0) / (el1 - el0)
        az_at_0.append(az0 + t * (az1 - az0))
        power_at_0.append(p0 + t * (p1 - p0))

    az_at_0 = np.array(az_at_0)
    power_at_0 = np.array(power_at_0)

    def _malus_law(az, peak_power, min_power, phase_offset):
        az_rad = np.deg2rad(az)
        phase_rad = np.deg2rad(phase_offset)
        return min_power + (peak_power - min_power) * np.cos(az_rad - phase_rad) ** 2

    num_freqs = dpss_red.shape[1]
    peak_powers = np.full(num_freqs, np.nan)
    for f_idx in range(num_freqs):
        p_freq = power_at_0[:, f_idx]
        valid = np.isfinite(p_freq) & np.isfinite(az_at_0)
        if np.sum(valid) < 4:
            continue
        p_v, az_v = p_freq[valid], az_at_0[valid]
        try:
            popt, _ = curve_fit(
                _malus_law, az_v, p_v,
                p0=[np.max(p_v), np.min(p_v), az_v[np.argmax(p_v)]],
            )
            peak_powers[f_idx] = popt[0]
        except RuntimeError:
            peak_powers[f_idx] = np.max(p_v)

    idx = np.arange(num_freqs)
    idx_strong, peaks_strong = idx[0::2], peak_powers[0::2]
    idx_weak, peaks_weak = idx[1::2], peak_powers[1::2]
    valid_strong = np.isfinite(peaks_strong)

    expected_strong_at_weak = np.full_like(peaks_weak, np.nan)
    if np.any(valid_strong):
        expected_strong_at_weak = np.interp(
            idx_weak, idx_strong[valid_strong], peaks_strong[valid_strong]
        )

    scale_factors = expected_strong_at_weak / peaks_weak
    return scale_factors, peak_powers
