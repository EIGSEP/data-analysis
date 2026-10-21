# Bench run 2026-10-20 (4 passes): noise-source temperature through the switch

The as-built record of the four-pass bench run started 2026-10-20. The
requirements it is measured against are in
[`../../receiver_calibration_lab_protocol.md`](../../receiver_calibration_lab_protocol.md);
each section below follows that protocol's numbering and says how this run
does it. [Gaps](#gaps-against-the-requirements) lists where it falls short,
with the effect and what would close it. The bench is the one of
[`2026-10-08_interim`](../2026-10-08_interim/README.md);
[Changes](#changes-from-2026-10-08_interim) lists what differs.

**Status:** in progress (started 2026-10-20). Fill in the run log, data,
problems and the _to confirm_ items when it ends; then do not edit except to
fill in the to-be-recorded items. A later run gets its own folder.

**Automation:** `eigsep_observing`, branch `calibration_scripts` (commit used:
_to record_; must include `eabcc3d`, the corrected switch-board thermistor
mapping). Config `src/eigsep_observing/config/obs_config_switch_bench.yaml`
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
| Hardware, supplies, correlator, switch and VNA schedule | — | Unchanged (_to confirm_) |

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
(field: 2²⁶ ≈ 0.268 s), as in 2026-10-08 (_to confirm_); `fft_shift`
0x015F, ADC gain 4, `corr_scalar` 512. _Copy the as-used `corr_config.yaml`
into this folder._

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
values, _to confirm for this run_):

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
   temperatures will vary between them.

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
| Plateau held until steady to ~0.1 K, then ≥ 30 min (§ 5) | Fixed 60 min per plateau, including the ramp; steady part chosen offline | Likely ≥ 30 min steady; check each plateau in the heater log | Hold-until-steady logic in the sequencer |
| Temperatures good to ~0.1 K (§ 5) | ±0.2 °C parts; 2-wire pull-ups (~0.1–0.3 K); thermistors noisy in 2026-10-08 | Temperature scale uncertainty of a few tenths of a kelvin | Better thermistors, 4-wire pull-ups or an ice-point / reference-thermometer calibration |
| ~30 s dwells, 3–5 min cycle (§ 4) | 10 s dwells, 1 min cycle | Fewer integrations per visit, less drift between visits | Change `switch_schedule` |
| SP1 every few cycles, ≥ 1 min each (§ 4) | Every cycle, 10 s each | Average SP1 across cycles | Change `switch_schedule` |
| VNA strictly between radiometer blocks (§ 6) | Interleaved every 20 min | Sweep integrations `VNA*`-flagged; integrations just before a sweep may carry the comb | Pause switching around sweeps, or schedule sweeps explicitly |
| 2–3 board temperatures (§ 5) | Not stepped | Only the room's natural range over ~22 h | Board enclosure or heater |
| Supply voltages logged (§ 3) | Written log only | Manual | A logged supply monitor |

## Run log (UTC)

**Time zones.** Everything the bench writes with a trailing `Z` (data file
names, heater-log lines) is UTC, as are the Unix times in the file headers
(`sync_time`, `obs_config_owner_uploaded_unix`). The ground computer's log is
in local time (PDT, UTC−7).

| Event | UTC | PDT | Source |
|---|---|---|---|
| SNAP synced | _to fill_ | _to fill_ | header `sync_time` |
| `eigsep-panda` started | _to fill_ | _to fill_ | header `obs_config_owner_uploaded_unix` |
| Heater program start (pass 1 ambient) | _to fill_ | _to fill_ | heater log |
| Passes 1–4: ambient / plateaus / cool-down | _to fill_ | _to fill_ | heater log |
| Program complete | _to fill_ | _to fill_ | heater log |
| Data kept (first → last file) | _to fill_ | _to fill_ | file names |

_Manual actions with their times: to fill in._

## Problems during the run

_To fill in._ Open items carried over from 2026-10-08 are in
[its to-do list](../2026-10-08_interim/README.md#to-do).

## To-do

- **This record:** the _to confirm_ items above (hardware, supplies,
  `corr_acc_len`); the run log; switch-board serial and whether it flew,
  the P → LNA 14 connection, hot-load cable/adapters, SP1 cable length
  (§ 8); photos; the `eigsep_observing` commit used.

## Data and files

- **Correlator and VNA files** (not tracked by git): _location to record._
  Expect roughly 300 `corr_*.h5` (~4.3 min each at 2²⁸ and `corr_ntimes`
  240) and ~65 each of `ants11_*.h5` and `recs11_*.h5` (every 20 min), on the
  order of 5 GB.
- **Configs as used** (in this folder): _copy `obs_config_switch_bench.yaml`
  and `corr_config.yaml` here._
- **Ground-computer event log** (in this folder): _to add (distilled
  `eigsep.log`, as for 2026-10-08)._
- **Heater log** (in this folder): _to add
  (`hot_load_plateaus_<UTC>.log`)._
- **Analysis notebook:** [`../../hot_load_testing_data.ipynb`](../../hot_load_testing_data.ipynb).
- **Photos (§ 8):** _to add._
