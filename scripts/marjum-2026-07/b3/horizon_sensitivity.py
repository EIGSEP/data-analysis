"""B3 addendum — sensitivity of T_ant to horizon errors.

Sets geometer's accuracy requirement. Perturbs the *measured* Marjum horizon
profiles and measures the induced antenna-temperature error:

  (a) rigid bearing rotation of the horizon profile   -> dT_ant per degree az
  (b) horizon elevation error, all-azimuth and sector -> dT_ant per degree el
  (c) the real 3.42 m horizontal-position uncertainty, using geometer's own
      eight displaced profiles rather than any approximation here

Two tolerances are reported and they differ by orders of magnitude:

  RAW      -- dT_ant against the per-channel thermal noise of the 87.5 m stare.
              This is the requirement if T_ant must be modelled directly.
  FILTERED -- the part of dT_ant surviving projection of the instrument-weighted
              foreground modes, pushed through a 21-cm matched filter into an
              equivalent bias on the recovered signal amplitude. This is the
              requirement that actually threatens the measurement, because a
              smooth horizon error is largely absorbed by the foreground filter.
              Reported together with alpha-hat, the surviving signal fraction --
              a filter without a signal-loss number is not a result.

Inputs
------
Horizon: ``marjum-2026-07/curation/horizon_profiles.npz``
(``horizon_profiles@v1+360fe99``), 3600 bearings at 0.1 deg, three eras, plus
eight profiles displaced by the 3.42 m horizontal position uncertainty.
Deliberately NOT the packaged ``horizon_models_v000.npz``: that is a HEALPix
nside-64 product, too coarse in azimuth to extract a profile reliably, and it is
under review for a skyline sign bug.

Beam: HFSS bowtie ``eigsep_bowtie_v000.npz``. Sky: GSM16.

Caveats carried
---------------
* Absolute T_ant carries an unresolved factor ~2.4 (rf-calibrator). Every
  tolerance here is a ratio at fixed model, so it is unaffected; only the
  absolute noise reference would scale.
* Antenna azimuth is not registered to true north (pointing-analyst), so the
  beam's orientation relative to the horizon profile is unknown. The bowtie is
  not azimuthally symmetric, so headline cases are evaluated at four beam
  azimuths and the spread is reported as a systematic.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import healpy as hp
from scipy.interpolate import CubicSpline

# data-analysis/scripts/marjum-2026-07/b3/ -> the workspace root.
# Was parents[3] when this lived at marjum-2026-07/analysis/b3/.
REPO = Path(__file__).resolve().parents[4]
HORIZON_NPZ = REPO / "marjum-2026-07/curation/horizon_profiles.npz"
BEAM_NPZ = REPO / "eigsep_sim/src/eigsep_sim/data/eigsep_bowtie_v000.npz"
T21_NPZ = REPO / "eigsep_sim/src/eigsep_sim/data/models_21cm.npz"

SITE_LAT, SITE_LON = 39.2464053, -113.4036356   # terrain/draw_eigsep.py
T_TERRAIN = 300.0
NSIDE = 64
NSIDE_SS = 1024                # sub-pixel horizon rasterization
CHAN_MHZ = 250.0 / 1024
STARE_HOURS = 9.97             # 87.5 m stare, usable (pointing_table v1)
N_LST = 8                      # LSTs for perturbation evaluation
N_LST_ENS = 48                 # dense ensemble spanning the foreground space

BANDS = {"cosmology_50_110": (50.0, 110.0), "trough_60_100": (60.0, 100.0)}
DF = 2.5                       # GSM anchor spacing, MHz
PROBE_FREQS = (60.0, 90.0, 150.0)
N_FG_MODES = 20                # B3: N_ant=4 x N_fg=5 in the cosmology band
BEAM_AZIMUTHS = (0.0, 45.0, 90.0, 135.0)
T21_FIDUCIAL_MK = 200.0        # assumed trough depth for the tolerance statement

DEG = np.pi / 180


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class SuperSampler:
    """Sub-pixel horizon rasterizer.

    A boolean blocked/open mask at the beam nside (64, 0.92 deg pixels) cannot
    resolve horizon perturbations below ~1 deg: sub-pixel shifts either do
    nothing or flip a whole pixel. Rasterize instead at a much finer nside and
    average down to a *fractional* terrain coverage per beam pixel, which
    responds smoothly to arbitrarily small shifts.
    """

    def __init__(self, bearings_deg, nside_out=NSIDE, nside_ss=NSIDE_SS):
        self.nsub = (nside_ss // nside_out) ** 2
        th, ph = hp.pix2ang(nside_ss, np.arange(hp.nside2npix(nside_ss)),
                            nest=True)
        self.el = (np.pi / 2 - th).astype(np.float32)
        self.az = ph.astype(np.float64)
        self.bearings = np.asarray(bearings_deg, float) * DEG
        self.nest2ring = hp.nest2ring(nside_out,
                                      np.arange(hp.nside2npix(nside_out)))

    def fraction(self, prof_rad, az_shift_deg=0.0):
        """Fractional terrain coverage per output pixel, RING ordered.

        The profile is evaluated as a continuous periodic function of azimuth.
        A bearing rotation shifts the evaluation argument rather than resampling
        the profile array, which would smooth it and swamp small shifts.
        """
        a = (self.az - az_shift_deg * DEG) % (2 * np.pi)
        pe = np.interp(a, self.bearings, prof_rad,
                       period=2 * np.pi).astype(np.float32)
        frac = (self.el < pe).reshape(-1, self.nsub).mean(axis=1)
        out = np.empty_like(frac)
        out[self.nest2ring] = frac
        return out


# -------------------------------------------------------------- sky & beam --
def sky_maps(freqs_mhz, lsts_hr, nside=NSIDE):
    """GSM16 rotated into topocentric coords. (n_lst, nfreq, npix)."""
    cache = Path(__file__).parent / f"_sky_cache_GSM16_n{nside}.npz"
    store = dict(np.load(cache)) if cache.exists() else {}
    need = [f for f in freqs_mhz if f"{f:.3f}" not in store]
    if need:
        from pygdsm import GlobalSkyModel16
        g = GlobalSkyModel16(freq_unit="MHz")
        for f in need:
            store[f"{f:.3f}"] = hp.ud_grade(g.generate(f), nside)
        np.savez_compressed(cache, **store)
    gal = np.array([store[f"{f:.3f}"] for f in freqs_mhz])
    out = []
    for lst in lsts_hr:
        rot = hp.Rotator(rot=[lst * 15.0, SITE_LAT - 90.0, 0.0],
                         deg=True, eulertype="ZYX", coord=["G", "C"])
        out.append(np.array([rot.rotate_map_pixel(m) for m in gal]))
    return np.array(out)


def sky_band(anchors, band_freqs, lsts, nside=NSIDE):
    """GSM16 on the correlator channel grid, splined in log-log from anchors."""
    sa = sky_maps(anchors, lsts, nside)
    n_nonpos = int((sa <= 0).sum())          # GSM16 ships a few negative pixels
    sa = np.maximum(sa, 1.0)
    la, lb = np.log(anchors), np.log(band_freqs)
    out = np.array([np.exp(CubicSpline(la, np.log(sa[i]), axis=0)(lb))
                    for i in range(len(lsts))])
    return out, n_nonpos


def beam_maps(freqs_mhz, az_rot_deg=0.0, nside=NSIDE):
    """HFSS bowtie power beam on topocentric pixels, main lobe at zenith."""
    d = np.load(BEAM_NPZ, allow_pickle=True)
    bf, bm = d["freqs"] / 1e6, d["bm"].astype(np.float64)
    x, y, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    a = az_rot_deg * DEG
    xr, yr = x * np.cos(a) + y * np.sin(a), -x * np.sin(a) + y * np.cos(a)
    # stored frame has the main lobe at theta=180; flip z to put it at zenith
    src = hp.vec2pix(int(d["nside"]), xr, yr, -z)
    return np.array([bm[int(np.argmin(np.abs(bf - f)))][src] for f in freqs_mhz])


def t_ant(beam, sky, frac, t_terrain=T_TERRAIN):
    """T_ant(nu) for one LST, with fractional terrain coverage per pixel."""
    eff = frac[None, :] * t_terrain + (1.0 - frac[None, :]) * sky
    return np.sum(beam * eff, axis=1) / np.sum(beam, axis=1)


# ------------------------------------------------------------ signal loss ---
def poly_basis(freqs_mhz, k):
    """Orthonormal k-dim basis of polynomials in log nu (worst-case filter)."""
    x = np.log(np.asarray(freqs_mhz, float) / np.mean(freqs_mhz))
    q, _ = np.linalg.qr(np.vander(x, k, increasing=True))
    return q.T


def fg_eigen_basis(ensemble, k):
    """Leading k eigenmodes of the instrument-weighted foreground."""
    m = np.asarray(ensemble, float).reshape(-1, np.shape(ensemble)[-1])
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    return vt[:min(k, vt.shape[0])]


def project_out(v, basis):
    return v - basis.T @ (basis @ v)


def t21_template(freqs_mhz):
    """Deepest 21-cm model in the library, in K, on freqs_mhz.

    NOTE: models_21cm.npz stores frequency in GHz (0.05-0.249), not MHz.
    """
    d = np.load(T21_NPZ, allow_pickle=True)
    f_mhz = np.asarray(d["freqs"], float) * 1e3
    models = np.asarray(d["models"], float)
    m = models[int(np.argmin(models.min(axis=1)))]      # deepest
    t_mk = np.interp(freqs_mhz, f_mhz, m)
    return t_mk * 1e-3                                   # mK -> K


def main():
    hz = np.load(HORIZON_NPZ, allow_pickle=False)
    bearings = np.asarray(hz["bearings_deg"], float)
    ss = SuperSampler(bearings)
    eras = {"30m": "elev_rad_30m", "87.5m": "elev_rad_87.5m",
            "91m": "elev_rad_91m"}
    pert = np.asarray(hz["horizontal_perturbation_elev_rad"], float)

    results = {"config": {
        "nside": NSIDE, "nside_supersample": NSIDE_SS,
        "site": [SITE_LAT, SITE_LON], "T_terrain": T_TERRAIN,
        "probe_freqs_mhz": list(PROBE_FREQS), "n_lst": N_LST,
        "n_lst_ensemble": N_LST_ENS, "n_fg_modes_removed": N_FG_MODES,
        "stare_hours": STARE_HOURS, "chan_mhz": CHAN_MHZ,
        "beam_azimuths_deg": list(BEAM_AZIMUTHS),
        "t21_fiducial_depth_mK": T21_FIDUCIAL_MK,
        "horizon_product": "marjum-2026-07/horizon_profiles@v1+360fe99",
    }, "bands": {}}

    lsts = np.linspace(0, 24, N_LST, endpoint=False)
    lst_ens = np.linspace(0, 24, N_LST_ENS, endpoint=False)
    beam_p = {a: beam_maps(np.array(PROBE_FREQS), a) for a in BEAM_AZIMUTHS}
    sky_p = sky_maps(np.array(PROBE_FREQS), lsts)

    for bname, (lo, hi) in BANDS.items():
        band_freqs = np.arange(lo, hi + 1e-9, CHAN_MHZ)
        anchors = np.arange(lo, hi + 1e-6, DF)
        sky_b, n_nonpos = sky_band(anchors, band_freqs, lsts)
        sky_e, _ = sky_band(anchors, band_freqs, lst_ens)
        beam_b = {a: beam_maps(band_freqs, a) for a in BEAM_AZIMUTHS}

        t21 = t21_template(band_freqs)
        t21_t2 = float(np.dot(t21, t21))
        band_out = {"range_mhz": [lo, hi], "n_channels": int(band_freqs.size),
                    "gsm16_nonpositive_clamped": n_nonpos, "eras": {}}

        for ename, key in eras.items():
            prof = np.asarray(hz[key], float)
            ent = {"horizon_mean_deg": float(np.degrees(prof).mean()),
                   "horizon_min_deg": float(np.degrees(prof).min()),
                   "horizon_max_deg": float(np.degrees(prof).max()),
                   "by_beam_azimuth": {}}

            for baz in BEAM_AZIMUTHS:
                bb, bp = beam_b[baz], beam_p[baz]
                f0 = ss.fraction(prof)
                base_b = np.array([t_ant(bb, sky_b[i], f0)
                                   for i in range(N_LST)])
                base_p = np.array([t_ant(bp, sky_p[i], f0)
                                   for i in range(N_LST)])

                tsec = STARE_HOURS * 3600.0
                sigma_p = base_p.mean(axis=0) / np.sqrt(CHAN_MHZ * 1e6 * tsec)
                sigma_chan = float(np.mean(base_b.mean(axis=0))
                                   / np.sqrt(CHAN_MHZ * 1e6 * tsec))

                ens = np.array([t_ant(bb, s, f0) for s in sky_e])
                filters = {"fg_eigen": fg_eigen_basis(ens, N_FG_MODES),
                           "poly_worstcase": poly_basis(band_freqs, N_FG_MODES)}
                fmeta, alpha, sig_amp = {}, {}, {}
                for fk, fb in filters.items():
                    tf = project_out(t21, fb)
                    tn = float(np.dot(tf, tf))
                    fmeta[fk] = (fb, tf, tn)
                    alpha[fk] = tn / t21_t2
                    # matched-filter amplitude uncertainty, dimensionless units
                    sig_amp[fk] = (sigma_chan / np.sqrt(tn)) if tn > 0 else None

                sub = {"T_ant_baseline_K": dict(zip(map(str, PROBE_FREQS),
                                                    base_p.mean(axis=0).tolist())),
                       "sigma_stare_K": dict(zip(map(str, PROBE_FREQS),
                                                 sigma_p.tolist())),
                       "alpha_hat": alpha,
                       "sigma_alpha": sig_amp,
                       "perturbations": {}}

                def record(name, prof_new, az_shift=0.0):
                    fn = ss.fraction(prof_new, az_shift)
                    dp = np.array([t_ant(bp, sky_p[i], fn)
                                   for i in range(N_LST)]) - base_p
                    db = np.array([t_ant(bb, sky_b[i], fn)
                                   for i in range(N_LST)]) - base_b
                    raw = np.abs(dp).mean(axis=0)
                    rec = {"dT_raw_K": dict(zip(map(str, PROBE_FREQS),
                                                raw.tolist())),
                           "dT_raw_over_sigma": dict(zip(map(str, PROBE_FREQS),
                                                         (raw / sigma_p).tolist()))}
                    for fk, (fb, tf, tn) in fmeta.items():
                        res = np.array([project_out(d, fb) for d in db])
                        rms = float(np.sqrt(np.mean(res ** 2)))
                        amp = np.array([float(np.dot(r_, tf) / tn)
                                        for r_ in res]) if tn > 0 else None
                        rec[fk] = {
                            "post_filter_rms_K": rms,
                            "suppression": rms / max(
                                float(np.sqrt(np.mean(db ** 2))), 1e-30),
                            "t21_bias_mK": (float(np.max(np.abs(amp))
                                                  * T21_FIDUCIAL_MK)
                                            if amp is not None else None),
                            "t21_bias_over_sigma_alpha": (
                                float(np.max(np.abs(amp)) / sig_amp[fk])
                                if amp is not None and sig_amp[fk] else None),
                        }
                    sub["perturbations"][name] = rec
                    return rec

                deltas = (0.1, 0.25, 0.5, 1.0, 2.0)
                sec = np.abs(((bearings - bearings[int(np.argmax(prof))] + 180)
                              % 360) - 180) < 30.0
                for d in deltas:
                    record(f"az_rot_{d}deg", prof, az_shift=d)
                    record(f"el_raise_{d}deg", prof + d * DEG)
                    record(f"el_sector60_{d}deg", prof + d * DEG * sec)

                # (c) the real 3.42 m position uncertainty (91 m era profiles)
                if ename == "91m":
                    for j in range(pert.shape[0]):
                        record(f"pos_3.42m_dir{j}", pert[j])

                # linearity: a per-degree slope is only quotable where dT is
                # proportional to the perturbation
                lin = {}
                for kind in ("az_rot", "el_raise", "el_sector60"):
                    for f in PROBE_FREQS:
                        v = [sub["perturbations"][f"{kind}_{d}deg"]["dT_raw_K"][str(f)]
                             for d in deltas]
                        sl = [x / d for x, d in zip(v, deltas)]
                        lin[f"{kind}@{f:.0f}MHz"] = {
                            "deltas_deg": list(deltas), "dT_K": v,
                            "slope_K_per_deg": sl,
                            "slope_ratio_max_min": (float(max(sl) / min(sl))
                                                    if min(sl) > 0 else None)}
                sub["linearity"] = lin
                ent["by_beam_azimuth"][f"{baz:.0f}"] = sub

            band_out["eras"][ename] = ent
        results["bands"][bname] = band_out

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO / "eigsep_sim"), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"
    results["provenance"] = {
        "product": "horizon_sensitivity", "campaign": "marjum-2026-07",
        "version": "v0",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eigsep_sim/analysis/b3/horizon_sensitivity.py",
        "generator_commit": commit,
        "inputs": [
            {"path": "marjum-2026-07/curation/horizon_profiles.npz",
             "sha256": sha256(HORIZON_NPZ),
             "stamp": "marjum-2026-07/horizon_profiles@v1+360fe99"},
            {"path": "eigsep_sim/src/eigsep_sim/data/eigsep_bowtie_v000.npz",
             "sha256": sha256(BEAM_NPZ)},
        ],
        "params": {"n_fg_modes": N_FG_MODES, "stare_hours": STARE_HOURS,
                   "t21_fiducial_depth_mK": T21_FIDUCIAL_MK},
        "notes": "Uses geometer's measured horizon profiles, NOT the packaged "
                 "horizon_models_v000.npz (nside-64, too coarse in azimuth for "
                 "profile extraction, and under review for a skyline sign bug).",
    }

    def _j(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    out = Path(__file__).parent / "b3_horizon_sensitivity.json"
    out.write_text(json.dumps(results, indent=2, default=_j))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
