# Bench run 2026-10-20 (4 passes): noise-source temperature through the switch

The as-built record of the four-pass bench run of 2026-10-20/21. The
requirements it is measured against are in
[`../../receiver_calibration_lab_protocol.md`](../../receiver_calibration_lab_protocol.md);
each section below follows that protocol's numbering and says how this run
does it. [Gaps](#gaps-against-the-requirements) lists where it falls short,
with the effect and what would close it. The bench is the one of
[`2026-10-08_interim`](../2026-10-08_interim/README.md);
[Changes](#changes-from-2026-10-08_interim) lists what differs.

**Status:** run 2026-10-20 10:24 → 2026-10-21 10:26 PDT; the heater
program (four passes) ran 10:24 → 08:24 PDT and completed with no faults
(see [Run log](#run-log-utc)). Do not edit except to fill in the
to-be-recorded items; a later run gets its own folder.

**Automation:** `eigsep_observing`, branch `calibration_scripts` (commit used:
_to record_; at or after `eabcc3d`, since the as-used config has the
corrected switch-board thermistor mapping). Config `src/eigsep_observing/config/obs_config_switch_bench.yaml`
(its `bench_setup` hardware record is embedded in every corr and VNA file
header); heater sequencer `scripts/hot_load_plateaus.py`.

## Summary

T_NS comes from the hot load on the `RFAMB` port against an
ambient 50 Ω reference on the `RFANT` port (§ 2), now over **four passes** of
the full temperature program with 60-minute plateaus (§ 5). Board temperature
is not stepped. Γ_rec is measured on a lab LNA (§ 1). The SP1 cable
temperature is logged.

## Changes from 2026-10-08_interim

| What | 2026-10-08_interim | This run |
|---|---|---|
| Temperature program (§ 5) | 1 pass: ambient 10 min, plateaus 15 min, cool-down 90 min | **4 passes: ambient 60 min, plateaus 60 min, cool-down 90 min** (~22 h) |
| Switch-board thermistor mapping (§ 2) | Config labelled `temp_therm1` as the noise diode (wrong) | Config corrected (`eabcc3d`): `temp_therm2` (GP28) is the noise diode |
| LNA thermistor channels (§ 2–3) | Data correct; that README lists `tempctrl_lna1`/`lna2` the wrong way round | `tempctrl_lna1` (GP27) = ambient reference load, `tempctrl_lna2` (GP28) = LNA 3, as in the config and firmware |
| Correlator, switch and VNA schedule | — | Unchanged: `corr_acc_len` 2²⁸ (SNAP log), 10 s dwells and 20-min VNA interval (as-used config) |
| Hardware and supplies | — | Unchanged (_to confirm_) |

## § 1. Why most of the chain need not be the field hardware

P_load is the ambient 50 Ω reference on `RFANT` and P_NS is
`RFNON` (§ 2). There is no record of which LNA modules flew, so the chain is
lab LNA 14 → LNA 3 → LNA 10. T_NS (the main result) does not depend on this,
but the bench Γ_rec and noise waves describe LNA 14 and **may not transfer to
the field receiver**. Accepted for the interim runs.

## § 2. Hardware configuration

1. Switch board with the rfswitch pico and the field state names. Whether it
   is the field unit is to be recorded (§ 8).
2. Lab LNA 14 on the common port; cable or adapter to be recorded (§ 8).
3. LNA 14 → LNA 3 → LNA 10 → SNAP C000122, no filters or attenuators.
4. **Hot load on the "Amb/Hot Load" port (`RFAMB` / `VNAAMB`)**, not
   `RFANT`: a 50 Ω load on the FET heater, with the `tempctrl_load`
   thermistor; load and thermistor are encased together in an aluminium
   block.
5. **Ambient 50 Ω reference on the Feed port (`RFANT` / `VNAANT`)**, with
   its own thermistor (`tempctrl_lna1`, GP27, moved from LNA 10). The two
   load roles are therefore **swapped** relative to items 4–5 of the
   protocol's § 2; every formula for this run uses the ports as wired.
6. SP1 cable fitted, open/short driven by the SP1 control pico (GP27;
   low/unpowered = SHORT), with a thermistor between two turns of the cable
   coil (SP1 control pico GP26).
7. VNA as in the field: 1–250 MHz, 1000 points, 100 Hz IFBW, 0 dBm
   (ant-side paths) / −40 dBm (`VNARF`).

Correlator: the field `corr_config.yaml` except `corr_acc_len` 2²⁸ ≈ 1.074 s
(field: 2²⁶ ≈ 0.268 s), as in 2026-10-08; `fft_shift` 0x015F, ADC gain 4,
`corr_scalar` 512. The as-used copy in this folder,
[`corr_config.yaml`](corr_config.yaml), has 2²⁸ active, matching the SNAP log
(`Setting CORR_ACC_LEN: 268435456` at the 10:17:09 PDT sync).

**Picos** (three, DIP code = app id, all under `pico-manager`):

| Pico | Job | Streams |
|---|---|---|
| tempctrl (app 1) | Hot-load heater FET + thermistor (GP26); two read-only thermistors (GP27, GP28) | `tempctrl_load`, `tempctrl_lna1` (GP27, ambient reference load), `tempctrl_lna2` (GP28, LNA 3) |
| potmon (app 2) | SP1 open/short (GP27); SP1 cable thermistor (GP26) | `potmon` (`sp1_term_name`, `pot_az_voltage`) |
| rfswitch (app 5) | Switch matrix; three board thermistors | `rfswitch`, `rfswitch_therm` |

Switch-board thermistors (`rfswitch_therm`, thermistor i on GP(26+i)); on
the bench board GP26 and GP27 are swapped relative to the field board:

| Field | Pin | Bench board | Field board |
|---|---|---|---|
| `temp_therm0` | GP26 | internal VNA calibration load | switch |
| `temp_therm1` | GP27 | common/crossbar switch | internal VNA calibration load |
| `temp_therm2` | GP28 | **noise diode** | **noise diode** |

## § 3. Logging

All of the streams listed in the protocol's § 3 are logged, plus:

- `tempctrl_load` also carries the heater setpoint in force (`T_target`,
  `enabled`, `hysteresis`) per integration;
- `rfswitch_therm.temp_therm2` (noise diode) is the noise-source/pad proxy;
- `tempctrl_lna1` is the ambient reference load temperature;
- `potmon`: the SP1 termination (`sp1_term_name`) and the SP1 cable
  thermistor as a raw voltage (`pot_az_voltage`; ignore the derived
  `pot_az_*` angle fields);
- the full observing config, including the `bench_setup` hardware record,
  is embedded in every corr and VNA file header.

Supply voltages are **not** logged automatically; written log (2026-10-08
values, assumed unchanged, _to confirm_):

| Supply | Voltage |
|---|---|
| LNAs (14, 3, 10) | 4.96 V |
| Switch board, SNAP and SP1 cable switch | 12.00 V |
| Hot-load heater | 16.330 V |

Heater actions are logged automatically with UTC times by
`hot_load_plateaus.py`.

**Thermistors.** All four bench thermistors are YSI 44909 (30 kΩ at 25 °C,
YSI curve H, rated −55 to +90 °C, ±0.2 °C interchangeable over 0–70 °C),
wired `3V3 –[pull-up]– ADC pin –[NTC]– GND`, with no filter capacitor.
Measured pull-ups (also in `bench_setup.thermistors`):

| Channel | Location | Pull-up |
|---|---|---|
| `tempctrl_load` (GP26) | Hot load body (encased together in aluminium) | 9.98 kΩ |
| `tempctrl_lna1` (GP27) | Ambient 50 Ω reference load | 9.97 kΩ |
| `tempctrl_lna2` (GP28) | LNA 3 | 9.91 kΩ |
| `potmon.pot_az_voltage` (SP1 pico GP26) | SP1 cable, between two turns of the coil | 9.96 kΩ |

The firmware assumes exactly 10 000 Ω (its `T_now` is for heater control
only). Offline:

- tempctrl channels: R = `resistance` × R_pullup / 10 000 (exact);
- potmon: R = R_pullup · V / (3.3 − V), with V = `pot_az_voltage`;
- then Steinhart–Hart, 1/T = A + B ln R + C (ln R)³ (T in K, R in Ω), with
  A = 9.362744897e-4, B = 2.209138866e-4, C = 1.280636863e-7 (fit to YSI's
  30 kΩ table over 0–110 °C, residual ≤ 0.019 K; `LOAD_THERM_SH_*` in
  pico-firmware `tempctrl.h`).

**Interim accuracy.** The pull-ups were measured 2-wire with a handheld
meter (~0.5–1 %), which leaves ~0.1–0.3 K of uncertainty per channel on top
of the ±0.2 °C part tolerance. The thermistors were also noisy in
2026-10-08; see [Gaps](#gaps-against-the-requirements).

## § 4. Switch schedule

10 s per state, a **1-minute cycle**:

    RFANT (ambient ref) → RFAMB (hot) → RFNON → RFNOFF → RFSP1 short → RFSP1 open → …

10 s is ~9 integrations, minus ~0.5 s flagged after each switch change.
SP1 is visited every cycle for 10 s per termination (not ≥ 1 min every few
cycles), so average SP1 across consecutive cycles.

## § 5. Temperature program

Run by `scripts/hot_load_plateaus.py` alongside the observer:

    python scripts/hot_load_plateaus.py --targets 47 67 87 \
        --ambient-min 60 --hold-min 60 --cooldown-min 90 --passes 4 \
        --hysteresis 0.05

Each pass is 60 + 3 × 60 + 90 = 330 min (5 h 30 min); four passes take
**about 22 h**.

1. Ambient (heater off) for 60 min (requirement: ≥ 45 min). It is also the
   lowest point on the hot load's own Q line (295→360 K instead of
   320→360 K, about 1.7× smaller slope error), and with the heater off
   Q ≈ 0 and the hot-load and reference thermistors should agree.
2. Plateaus at **47, 67 and 87 °C targets (~320, 340, 360 K)**, not
   320/345/370 K: the YSI 44909 is rated to 90 °C, so the script refuses
   targets above 88 °C. Each plateau is held 60 min including the ramp, on
   a timer rather than until steady; use only the steady part (the script
   logs a 10-min peak-to-peak every minute). The heater is on/off with
   0.05 °C hysteresis, so a plateau's mean sits ~0.025 °C below target; use
   the logged temperature, not the setpoint.
3. Cool-down, heater off, 90 min.
4. **Four passes** (requirement: repeat if time allows; two passes test the
   drift). The passes cover different times of day, so the room and board
   temperatures vary between them.

**As run** (from the heater log's once-a-minute `T_now`, the firmware's
nominal-pull-up conversion; analysis uses the corrected conversion in § 3).
Ramp = minutes from heater on to first reaching the target; mean and σ over
each phase's last 30 min:

| Pass | Ambient mean | 47 °C: ramp / mean / σ | 67 °C: ramp / mean / σ | 87 °C: ramp / mean / σ | End of cool-down |
|---|---|---|---|---|---|
| 1 | 21.41 °C | 6 min / 47.05 / 0.08 | 6 min / 67.00 / 0.11 | 10 min / 86.84 / 0.16 | 23.08 °C |
| 2 | 21.64 °C | 6 min / 47.04 / 0.07 | 10 min / 66.98 / 0.12 | 10 min / 86.88 / 0.14 | 22.88 °C |
| 3 | 21.45 °C | 6 min / 47.03 / 0.08 | 9 min / 66.99 / 0.10 | 10 min / 86.86 / 0.14 | 23.03 °C |
| 4 | 21.66 °C | 6 min / 47.05 / 0.09 | 6 min / 66.97 / 0.11 | 10 min / 86.91 / 0.11 | 23.03 °C |

Every plateau reached its target within 10 min, leaving ≥ 50 min at
temperature. The σ of 0.07–0.16 °C is mostly readout noise (the
thermistors are noisy; see Gaps). Plateau means repeat between passes to
within ~0.07 °C. The 87 °C plateaus sit ~0.1 °C below target. The 90-min
cool-down ends ~1.5 °C above room temperature, so each following ambient block
starts warm and settles within its hour.

The script sends a keepalive every 10 s, turns the heater off and stops on
any heater trip or sensor error, and turns the heater off on exit.
Restarting `eigsep-panda` resets the heater to off, so restart the script
too. Board temperature is not stepped: the board thermistors will show only
the room's natural range over the day.

## § 6. VNA measurements

The VNA runs automatically every 20 min (and once at
observer start), with the internal standards, then:

| What | Path (as wired) | When |
|---|---|---|
| Hot load, Γ_hot | `VNAAMB` | Every 20 min: ~3 per plateau, including ambient |
| Ambient reference, Γ_amb | `VNAANT` | Every 20 min |
| Noise source on / off | `VNANON` / `VNANOFF` | Every 20 min |
| SP1 open and short | `VNASP1` | Every 20 min, with the cable temperature logged |
| Receiver, Γ_rec | `VNARF` | Every 20 min (covers start and end) |
| Receiver, measured directly | VNA on LNA 14's input, S911T at that connector | **Manual**, start and end |

Sweeps are interleaved into the radiometer cycle (they take over the
switch during an `RFANT` dwell for a few minutes) rather than strictly
between blocks. Their integrations carry `VNA*` states and are excluded
via `rfswitch`. The VNA service starts ~5 s **before** it takes the
switch, so integrations just before each sweep may carry the 1 MHz comb
(§ 7); flag them by time.

## § 7. EMI test (bonus, about 20 minutes)

Manual, outside the automated run (stop `eigsep-panda`, hold
the switch on a load). The automated run gives a weaker version for free:
compare integrations just before each 20-min sweep with the rest.

## § 8. Hardware description (with photos)

Recorded in `bench_setup` in the config (embedded in every
file header). LNA modules: lab labels 14 / 3 / 10, **no record of which, if
any, flew**. Downstream chain: LNA 14 → LNA 3 → LNA 10 → SNAP, no filters
or attenuators. Still to fill in: switch-board serial and whether it flew,
the P → LNA 14 connection, the hot-load cable/adapters, the SP1 cable
length.

## § 9. Separate, when convenient: the balun

Not done yet.

## § 10. Quick checks during the run

Also: with the heater off, Q of the hot load ≈ 0 and the
hot-load and reference thermistors agree to a fraction of a kelvin;
`scripts/pico_preflight.py` shows all three picos reporting. Between passes,
the same plateau should give the same Q (the drift test).

## § 11. What to hand over

Also hand over the `hot_load_plateaus` log and
`obs_config_switch_bench.yaml` as used. Q is computed for the hot load
(`RFAMB`) against the ambient reference (`RFANT`); the bench Γ_rec carries
the lab-LNA caveat of § 1.

## Gaps against the requirements

| Requirement | This run | Effect | To meet the requirement |
|---|---|---|---|
| Field first LNA board (§ 1–2) | Lab LNA 14; flight history unknown | T_NS unaffected; Γ_rec and noise waves may not transfer | Identify and use the field first-stage board |
| Hot load on `RFANT`, ambient on `RFAMB` (§ 2) | Swapped: hot on `RFAMB`, ambient on `RFANT` | Use the matching paths (`VNAAMB` hot, `VNAANT` ambient) | None needed if the analysis follows the wiring; or rewire |
| Plateaus ~320 / 345 / 370 K (§ 5) | ~320 / 340 / 360 K | Span 65 K instead of 75 K | Thermistors rated above ~100 °C (higher dynamic range) |
| Plateau held until steady to ~0.1 K, then ≥ 30 min (§ 5) | Fixed 60 min per plateau including the ramp; ramps took 6–10 min, leaving ≥ 50 min at temperature | Time requirement met; steadiness to ~0.1 K limited by thermistor readout noise (σ 0.07–0.16 °C) | Lower-noise thermistor readout |
| Temperatures good to ~0.1 K (§ 5) | ±0.2 °C parts; 2-wire pull-ups (~0.1–0.3 K); thermistors noisy in 2026-10-08 | Temperature scale uncertainty of a few tenths of a kelvin | Better thermistors, 4-wire pull-ups or an ice-point / reference-thermometer calibration |
| ~30 s dwells, 3–5 min cycle (§ 4) | 10 s dwells, 1 min cycle | Fewer integrations per visit, less drift between visits | Change `switch_schedule` |
| SP1 every few cycles, ≥ 1 min each (§ 4) | Every cycle, 10 s each | Average SP1 across cycles | Change `switch_schedule` |
| VNA strictly between radiometer blocks (§ 6) | Interleaved every 20 min | Sweep integrations `VNA*`-flagged; integrations just before a sweep may carry the comb | Pause switching around sweeps, or schedule sweeps explicitly |
| 2–3 board temperatures (§ 5) | Not stepped | Only the room's natural range over ~22 h | Board enclosure or heater |
| Supply voltages logged (§ 3) | Written log only | Manual | A logged supply monitor |

## Run log (UTC)

**Time zones.** Everything the bench writes with a trailing `Z` (data file
names, heater-log lines) is UTC, as are the Unix times in the file headers
(`sync_time`, `obs_config_owner_uploaded_unix`). The ground computer's and
the panda's logs are in local time (PDT, UTC−7); their distilled versions in
this folder give both. All clocks agreed.

| Event | UTC | PDT | Source |
|---|---|---|---|
| SNAP synced (`sync_time` 1792516629.873) | 10-20 17:17:09 | 10-20 10:17:09 | ground event log |
| `eigsep-observe` started | 10-20 17:24:00 | 10-20 10:24:00 | ground event log |
| `eigsep-panda` started | 10-20 17:24:16 | 10-20 10:24:16 | panda event log |
| Heater program start | 10-20 17:24:31 | 10-20 10:24:31 | heater log |
| Program complete | 10-21 15:24:31 | 10-21 08:24:31 | heater log |
| `eigsep-observe` stopped | 10-21 17:25:53 | 10-21 10:25:53 | ground event log |
| `eigsep-panda` stopped | 10-21 17:26:00 | 10-21 10:26:00 | panda event log |
| Corr files (first → last) | 10-20 17:28:16 → 10-21 17:25:53 | 10-20 10:28 → 10-21 10:25 | file names |

Heater phases (heater log; each pass: ambient 60 min, plateaus 47 / 67 /
87 °C 60 min each, cool-down 90 min), phase start times:

| Pass | Ambient | 47 °C | 67 °C | 87 °C | Cool-down |
|---|---|---|---|---|---|
| 1 | 10-20 17:24:31Z (10:24 PDT) | 18:24:31Z (11:24) | 19:24:31Z (12:24) | 20:24:31Z (13:24) | 21:24:31Z (14:24) |
| 2 | 10-20 22:54:31Z (15:54 PDT) | 23:54:31Z (16:54) | 10-21 00:54:31Z (17:54) | 01:54:31Z (18:54) | 02:54:31Z (19:54) |
| 3 | 10-21 04:24:31Z (21:24 PDT, 10-20) | 05:24:31Z (22:24) | 06:24:31Z (23:24) | 07:24:31Z (00:24, 10-21) | 08:24:31Z (01:24) |
| 4 | 10-21 09:54:31Z (02:54 PDT) | 10:54:31Z (03:54) | 11:54:31Z (04:54) | 12:54:31Z (05:54) | 13:54:31Z (06:54) |

After the program completed (08:24 PDT) the bench kept recording with the
heater off until 10:25 PDT, ~2 h of extra ambient data.

**Manual actions during the run:** none in the logs (no restart of
`eigsep-observe`, `eigsep-panda` or the heater script between 10-20 10:24 and
10-21 10:25 PDT); written log _to confirm_.

Before the run (ground and panda event logs): a test session 09:29–09:38 PDT;
a ground-computer restart at 10:16 PDT followed by SNAP programming and the
10:17:09 sync; brief `eigsep-observe` / `eigsep-panda` restarts until
10:24.

## Problems during the run

None during the run itself: no corr or VNA write failures, no dropped
integrations, no heater trips or sensor errors, no restarts. Noted from the
logs:

| Problem | When | Effect | Status |
|---|---|---|---|
| T7 drive input/output errors on 4 corr writes | 10-20 09:37–09:38 PDT (pre-run test) | Those test files lost; no errors during the run | Watch the drive; check it before the next run |
| 2 corr writes failed ("No such file or directory") at the SNAP re-sync file roll | 10-20 10:21:27 PDT (setup) | Setup data only | Benign |
| Thermistor readout noise σ 0.07–0.16 °C | Whole run | Limits plateau steadiness to ~0.1 K | Open (thermistors being replaced) |
| "No pico devices registered by PicoManager" at each `eigsep-panda` start | Starts | None (legacy check; the picos ran throughout) | Benign; could remove the check |

Open items carried over from 2026-10-08 are in
[its to-do list](../2026-10-08_interim/README.md#to-do).

## To-do

- **This record:** the _to confirm_ items above (hardware, supplies, no
  manual actions); the final location of the data; switch-board serial and
  whether it flew,
  the P → LNA 14 connection, hot-load cable/adapters, SP1 cable length
  (§ 8); photos; the `eigsep_observing` commit used.

## Data and files

- **Correlator and VNA files** (not tracked by git): written to
  `/media/eigsep/T7/data/` on the ground computer (_final location to
  record_). 336 `corr_*.h5` (258 s each, `corr_20261020_172816Z.h5` →
  `corr_20261021_172553Z.h5`) and 129 `ants11_*.h5` / `recs11_*.h5` files
  from ~65 VNA sweeps. The first files before 10:24 PDT on 10-20 are setup
  data.
- **Configs as used** (in this folder): [`obs_config_switch_bench.yaml`](obs_config_switch_bench.yaml)
  and [`corr_config.yaml`](corr_config.yaml) (`corr_acc_len` 2²⁸).
- **Ground-computer event log** (in this folder):
  [`eigsep_events_2026-10-20.log`](eigsep_events_2026-10-20.log), the ground
  computer's two rotated `eigsep.log` files combined and distilled to
  start/stop, configuration and collapsed warning/error events, with a
  key-periods summary.
- **Panda event log** (in this folder):
  [`panda_events_2026-10-20.log`](panda_events_2026-10-20.log), the panda's
  `eigsep.log` for 10-20 and 10-21, distilled the same way.
- **Heater log** (in this folder): [`hot_load_plateaus_20261020T172431Z.log`](hot_load_plateaus_20261020T172431Z.log)
  (UTC; every action and a once-a-minute temperature readout). The full
  ground and panda logs were not kept.
- **Analysis notebook:** [`../../hot_load_testing_data.ipynb`](../../hot_load_testing_data.ipynb).
- **Photos (§ 8):** _to add._
