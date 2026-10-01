# abscal — absolute calibration of the Marjum 2026-07 switched-load data

Campaign-specific analysis code for B7. Produces
$T_{\rm ant}^{\rm (sw)}(\nu,t)$, $g_{\rm rx}(\nu,t)$ and $T_{\rm rx}(\nu,t)$
from the in-band Dicke-switching data, with a propagated uncertainty budget.

Dependencies: stdlib + numpy + h5py, plus `eigsep_data.paths` (campaign
location) and `eigsep_data.select_files` (file selection). The calibration
math itself (`abscal.py`'s Y-factor solve and error propagation) still takes
no `eigsep_*` import, so it stays verifiable independently of the packages
whose outputs it will eventually be compared against — only *finding the
campaign and its curated file list* now goes through `eigsep_data`, which is
also where `select_files.py` itself now lives (moved from
`marjum-2026-07/curation/` on 2026-09-19).

Moved here from `~/projects/eigsep/abscal/` on 2026-09-19 (AGENTS.md's
five-destination table). Point at a campaign before running anything:

```sh
export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
```

or in Python, `eigsep_data.set_campaign_root(...)`.

Python: `/home/aparsons/.local/share/mamba/envs/arp/bin/python3`.

## Read this before using any number from here

**The reference plane is the RF switch common port**, inside the receiver box:
the point at which the RFAMB ambient load and the RFNON noise source are
presented. The balun, feed cable, connectors, antenna reflection and radiation
efficiency are all **absorbed into** the reported $T_{\rm ant}$, not corrected
for. The product is the input temperature at that port when the switch selects
the antenna. **It is on an absolute Kelvin scale. It is not a sky brightness
temperature.**

Measured evidence that this distinction is load-bearing: the calibrated
spectrum has $\beta = +0.42$ over 55–85 MHz, where Galactic synchrotron
requires $\beta \approx -2.5$. The entire difference is uncorrected instrument
response.

## Files

| File | Role |
|---|---|
| `abscal.py` | library: file reading, state extraction, Y-factor solve, exact error propagation |
| `run_window.py` | driver: one window → calibrated `.npz` |
| `validate.py` | three checks that are *not* the fit's own training residual |
| `budget.py` | uncertainty budget + cost of the missing lab measurements |
| `scan_switch.py` | targeted `metadata/rfswitch` scan over a file glob |
| `find_windows.py` | groups scanned states into contiguous cal windows |
| `extend_trx.py` | T_rx(nu,t)/g_rx(nu,t) across Phase C, consuming `curation/cal_windows.jsonl` |
| `regimes.py` | segments the Phase-C history into receiver regimes; control-vs-injection decomposition |
| `enr_inband.py` | in-band ENR from the RFNOFF/RFAMB/RFNON triplet route |
| `audit_wrap.py` | int32 accumulator-wrap audit of the solved cycles |
| `make_figures.py` | regenerates the memo figures from the `.npz` products |
| `bracket_gap.py` | quantifies the P_amb step across a cal gap vs. pre/post window size (added 2026-09-19, see MEMORY note on the CAMPAIGN.md 4.5%→4-12% correction) |

## Regenerating the derived products

Everything not tracked in git is reproducible from the raw campaign data plus
this code. Nothing here is hand-curated.

```sh
PY=/home/aparsons/.local/share/mamba/envs/arp/bin/python3

# 1. switch-state scan over 07-17/07-18  -> scan_1718.jsonl  (~4.7 MB, ~1 min)
$PY scan_switch.py --glob 'corr_2026071[78]*.h5' --out scan_1718.jsonl --summary-only

# 2. contiguous cal windows from that scan (prints; no file written)
$PY find_windows.py scan_1718.jsonl

# 3. calibrate one window -> window_0717.npz  (~5.0 MB)
$PY run_window.py --phase C \
    --start 2026-07-17T04:00:00Z --end 2026-07-17T12:00:00Z \
    --out window_0717.npz

# 4. validation + budget
$PY validate.py window_0717.npz
$PY budget.py  window_0717.npz
```

File selection always goes through `eigsep_data.select_files.select()`
(`run_window.py`, `extend_trx.py` and `audit_wrap.py` all call it directly) —
never glob the data directory directly for science selections. `scan_switch.py`
globs, but only to survey the switch axis, never to choose science data.

## Key campaign facts this code depends on

- The switched receiver is **input `4`** (`box-air`). Established empirically:
  across an RFNON/RFAMB dwell input 4 moves ×2.3 while `box-gnd` moves < 2 %.
- Reference temperatures come from the campaign's own
  `header/obs_config.calibration` block — `noise_diode_enr_db: 35.0`,
  `noise_source_atten_db: 30.0`, giving
  $T_{\rm ENR} = 290\cdot10^{(35-30)/10} = 917.1$ K. **This is a nameplate
  value, never measured.** It carries a 354 K (19.2 %) systematic and dominates
  the budget by 225:1 over statistical error.
- `UNKNOWN` switch states are the transition window and are never calibrated on.
- The formalism reuses
  `eigsep_observing/src/eigsep_observing/live_status/calibration.py`, the code
  that ran in the field, so the live dashboard and this pipeline cannot
  silently disagree. That module is explicitly first-order: no noise-wave
  parameters, no reflection correction, no bandpass polynomial.
