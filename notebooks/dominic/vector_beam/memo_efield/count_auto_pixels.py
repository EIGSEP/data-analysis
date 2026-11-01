"""
Sky pixels the auto-correlation power method covers at the map teeth, on the same
integrations and pixels (Phase 2 pointing selection, v0013 geometry) as the
E-field inputs, against the cross-correlation inputs and the E-field solve
(mount L). Auto recipe: DPSS-reduced sky tones, weak arm scaled, time- and
frequency-axis RFI flags left NaN (no baseline replacement),
fit_dpss_interleaved_fix with the trust mask. Run after compute_beams.py; needs
VECTOR_BEAM_CACHE as pipeline.py does.

Output: output/auto_pixels.json.
"""

import json

import numpy as np
from scipy.ndimage import binary_dilation

from eigsep_data.data import calibrate_weak_arm, extract_beam_mapping_data, extract_clean_pot_data_v2
from eigsep_data.rfi import (build_frequency_trust_mask, fit_dpss_interleaved_fix,
                             fit_dpss_model_per_time, flag_rfi_time, freq_axis_flag_reduced)
from mounts import INPUTS, MAP_D, OUT, tooth_freq  # also puts ../ on sys.path
import pipeline as P  # noqa: E402

z = np.load(INPUTS)
cache = np.load(P.CACHE, allow_pickle=True)
ext = extract_beam_mapping_data(P.Path(str(cache["data_dir"])), str(cache["start_time"]),
                                str(cache["end_time"]), file_patterns=("*.h5", "*.hdf5"),
                                sweep_slice=slice(int(cache["sweep_slice_start"]), None))
assert np.allclose(np.asarray(ext["times"], float), z["times"]), "integration axes differ"
sky, freqs = ext["sky"], ext["freqs"]
az = extract_clean_pot_data_v2(np.copy(ext["pot_az_angle"]), ext["az_pos"] / (113 / 1.8))
rot = fit_dpss_model_per_time(raw=sky, chs=P.CHS, freqs=freqs, nw=4.0, nterms=32, include_poly=True,
                              poly_order=2, guard_bins=1, fit_min_chan=min(P.CHS) - 8,
                              fit_max_chan=sky.shape[1], min_freq=87.0, use_log=True, ridge=1e-5,
                              verbose=False)["reduced"]
scale, _ = calibrate_weak_arm(ext["imu_el_deg"].copy(), az, rot)
Pm = rot.copy()
Pm[:, 1::2] *= scale
Pm = Pm[:, P.SHIFT:]
fm = freqs[::8][P.SHIFT + 1:]
m, _, _ = flag_rfi_time(Pm, window=9)
m = binary_dilation(m | freq_axis_flag_reduced(Pm, window=9), np.ones((3, 1), bool))
P1, P6, G, bad, _, M1, M6 = fit_dpss_interleaved_fix(Pm, fm, rfi_mask=m, nmodes=P.NMODES, NW=P.NW,
                                                     return_measured_mask=True,
                                                     trust_mask=build_frequency_trust_mask(fm, **P.TRUST))
S = {r["d"]: r for r in json.load(open(OUT / "scores.json"))}
npix = lambda use: int(len(np.unique(z["pxs"][use])))
out = {}
for d in MAP_D:
    own, meas = (P1, M1) if d % 2 == 0 else (P6, M6)
    base = z["sel"] & ~bad
    v_meas = base & meas[:, d] & np.isfinite(own[:, d]) & (own[:, d] > 0)
    v_G = base & np.isfinite(G[:, d]) & (G[:, d] > 0)
    v_x = z["sel"] & ~z["bad_time"] & np.isfinite(z["Ex_all"][:, d]) & np.isfinite(z["Ey_all"][:, d])
    out[d] = dict(freq=tooth_freq(d), auto_measured_arm_samples=int(v_meas.sum()),
                  auto_measured_arm_pix=npix(v_meas), auto_total_gain_samples=int(v_G.sum()),
                  auto_total_gain_pix=npix(v_G), cross_samples=int(v_x.sum()), cross_input_pix=npix(v_x),
                  efield_pix=S[d]["L_npix"], pointing_selected_pix=npix(z["sel"]))
    print(d, out[d], flush=True)
json.dump(out, open(OUT / "auto_pixels.json", "w"), indent=1)
