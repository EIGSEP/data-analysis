# Receiver temperature calibration — lab measurement protocol

For Charlie. The goal is to put the 2026 field receiver on an absolute kelvin
scale. Today the field calibration (`tcal`, memo 004) rests on the noise
source's **nameplate** excess, T_NS = 917 K. The only in-band estimate is 2.4
times lower and cannot be trusted either. This run measures T_NS directly, the
way EDGES does (their constant C₁), and checks the receiver reflection that the
tier-2 noise-wave fit cannot settle on its own.

> **How to read this document (updated 2026-10-07).** The body of each
> section is the **original requirement**, unchanged (git `d3e2572`). It
> remains the target for the final measurement. Below each section, a
> **This run** block describes the interim bench as it is actually being run
> now, and where it falls short. § 12 collects every gap, with what is needed
> to close it. The interim run is automated by `eigsep_observing` on the
> `calibration_scripts` branch: config
> `src/eigsep_observing/config/obs_config_switch_bench.yaml`, heater sequencer
> `scripts/hot_load_plateaus.py`.

## What we need, and why

| Quantity | Why it matters | Comes from |
|---|---|---|
| **T_NS**, the noise-source excess as seen at plane P | Sets the kelvin scale of every calibrated spectrum | Hot and ambient loads on the antenna port, switch cycling (§ 4–5) |
| **T_NS against board temperature** | The field injection drifted 15 % over a night | The same, with the switch-board thermistors logged, at 2–3 board temperatures if possible (§ 5) |
| **Γ_rec**, the receiver reflection at P, measured independently | The tier-2 fit wants it ~0.05 off the field value; field data cannot tell why | VNA, directly at the first LNA board's input and through VNARF (§ 6) |
| **The path from P to the first LNA stage** | The tier-2 fit wants a ~8 ns round-trip delay in the noise waves (~0.8 m of coax) | A description, cable lengths and photos (§ 8) |
| Optional: noise waves | Matter most below 60 MHz | The spare-port (SP1) open/short at a logged temperature (§ 4) |
| Bonus: the 1 MHz comb EMI | Memo 006 cannot identify its source without a physical test | VNA on, idle and off blocks (§ 7) |

**Plane P** is the switch matrix's LNA-side common port: the connector where
you placed the S911T standards to characterize the RF-side switch paths
(`characterizing_system_2026.ipynb`). Every calibrated temperature and every Γ
is referred to it.

> **This run.** T_NS comes from the hot load on the `RFAMB` port against an
> ambient 50 Ω reference on the `RFANT` port (§ 2). Board temperature is not
> stepped. Γ_rec is measured on a lab LNA (§ 1). The SP1 cable temperature is
> logged.

## 1. Why most of the chain need not be the field hardware

The calibration works on the switched ratio

    Q_s = (P_s − P_load) / (P_NS − P_load)

per channel. The gain of everything **after** plane P cancels in Q, and any
noise added after P is common to all switch states and is absorbed into the
load constant. So the RF-over-fibre link, the ground receiver, the filters and
the attenuators do **not** have to be the field units. They only have to be
**linear and stable over one switching cycle**. The lab LNA modules and the SNAP
are fine downstream, provided nothing saturates.

What **must** be the field hardware is whatever sets the quantities at P:

- **the field switch board**, with its noise source and 30 dB pad, because T_NS
  is defined there;
- **the field's first LNA board (the same unit)**, because it sets Γ_rec and the
  noise waves seen from P. The second board matters much less once the first
  board's gain dominates.

> **This run.** P_load is the ambient 50 Ω reference on `RFANT` and P_NS is
> `RFNON` (§ 2). There is no record of which LNA modules flew, so the chain is
> lab LNA 14 → LNA 3 → LNA 10. T_NS (the main result) does not depend on this,
> but the bench Γ_rec and noise waves describe LNA 14 and **may not transfer to
> the field receiver**. Accepted for the interim run.

## 2. Hardware configuration

1. Field switch board, with the rfswitch daemon running exactly as in the field
   (the same state names: `RFANT`, `RFAMB`, `RFNON`, `RFNOFF`, `RFSP1`, and the
   `VNA*` states).
2. Field first LNA board connected to the switch's common port **with the same
   cable or adapter used in the field**. Record what it is and its length (§ 8).
3. Downstream: the field's second LNA board if you have it, then the lab chain to
   the SNAP. Keep the SNAP input level comparable to the field and well below
   clipping.
4. **Hot load on the antenna port (`RFANT`)**: a 50 Ω load on the heater, with the
   `tempctrl_load` thermistor on the load body. Record the cable or adapter
   between the load and the port, and keep it as short as possible: it heats too.
5. **Ambient load on the `RFAMB` port**, as in the field, with its own thermistor
   if one is available.
6. Optional: **SP1 cable** terminated open and short, as in the field, **with a
   temperature sensor taped to the cable**. In the field its temperature was never
   logged, which limits the noise-wave fit.
7. VNA connected exactly as in the field, so that the field `VNA*` paths and
   `eigsep_cal.S11` apply unchanged.

Use the field correlator configuration (`corr_acc_len`, `fft_shift`, ADC gain,
`corr_scalar`), and note any difference.

> **This run.**
>
> 1. Switch board with the rfswitch pico and the field state names. Whether it
>    is the field unit is to be recorded (§ 8).
> 2. Lab LNA 14 on the common port; cable or adapter to be recorded (§ 8).
> 3. LNA 14 → LNA 3 → LNA 10 → SNAP C000122, no filters or attenuators.
> 4. **Hot load on the "Amb/Hot Load" port (`RFAMB` / `VNAAMB`)**, not
>    `RFANT`: a 50 Ω load on the FET heater, with the `tempctrl_load`
>    thermistor; load and thermistor are encased together in an aluminium
>    block.
> 5. **Ambient 50 Ω reference on the Feed port (`RFANT` / `VNAANT`)**, with
>    its own thermistor (`tempctrl_lna2`, moved from LNA 10). The two load
>    roles are therefore **swapped** relative to items 4–5 above; every
>    formula for this run uses the ports as wired.
> 6. SP1 cable fitted, open/short driven by the SP1 control pico (GP27;
>    low/unpowered = SHORT), with a thermistor between two turns of the cable
>    coil (SP1 control pico GP26).
> 7. VNA as in the field: 1–250 MHz, 1000 points, 100 Hz IFBW, 0 dBm
>    (ant-side paths) / −40 dBm (`VNARF`).
>
> Correlator: the field `corr_config.yaml` (`corr_acc_len` 2²⁶ ≈ 0.268 s,
> `fft_shift` 0x015F, ADC gain 4, `corr_scalar` 512).
>
> **Picos** (three, DIP code = app id, all under `pico-manager`):
>
> | Pico | Job | Streams |
> |---|---|---|
> | tempctrl (app 1) | Hot-load heater FET + thermistor (GP26); two read-only thermistors (GP27, GP28) | `tempctrl_load`, `tempctrl_lna1` (LNA 3), `tempctrl_lna2` (ambient reference load) |
> | potmon (app 2) | SP1 open/short (GP27); SP1 cable thermistor (GP26) | `potmon` (`sp1_term_name`, `pot_az_voltage`) |
> | rfswitch (app 5) | Switch matrix; three board thermistors | `rfswitch`, `rfswitch_therm` |
>
> Switch-board thermistors (`rfswitch_therm`, thermistor i on GP(26+i)):
> `temp_therm0` = common/crossbar switch, `temp_therm1` = **noise diode**,
> `temp_therm2` = internal VNA calibration load.

## 3. Logging

Every correlator file must carry these metadata streams. In the 09-25 run,
`rfswitch` and `rfswitch_therm` were missing.

- `rfswitch`: the switch state per integration. **This is essential.**
- `tempctrl_load`: the hot-load temperature.
- `rfswitch_therm`: the three switch-board thermistors, the proxy for the
  noise-source and pad temperature.
- `tempctrl_lna1`, `tempctrl_lna2`, `fpga_temp`, `adc_stats`.
- Supply voltages, if they can be logged.

Keep a **written log with UTC times** of every manual action: heater set points,
reconnections, VNA sweeps, voltage changes, anything moved.

> **This run.** All of the streams above are logged, plus:
>
> - `tempctrl_load` also carries the heater setpoint in force (`T_target`,
>   `enabled`, `hysteresis`) per integration;
> - `rfswitch_therm.temp_therm1` (noise diode) is the noise-source/pad proxy;
> - `tempctrl_lna2` is the ambient reference load temperature;
> - `potmon`: the SP1 termination (`sp1_term_name`) and the SP1 cable
>   thermistor as a raw voltage (`pot_az_voltage`; ignore the derived
>   `pot_az_*` angle fields);
> - the full observing config, including the `bench_setup` hardware record,
>   is embedded in every corr and VNA file header.
>
> Supply voltages are **not** logged automatically; they go in the written
> log. Heater actions are logged automatically with UTC times by
> `hot_load_plateaus.py`.
>
> **Thermistors.** All four bench thermistors are YSI 44909 (30 kΩ at 25 °C,
> YSI curve H, rated −55 to +90 °C, ±0.2 °C interchangeable over 0–70 °C),
> wired `3V3 –[pull-up]– ADC pin –[NTC]– AGND` with 100 nF from pin to AGND.
> Measured pull-ups (also in `bench_setup.thermistors`):
>
> | Channel | Location | Pull-up |
> |---|---|---|
> | `tempctrl_load` | Hot load body (encased together in aluminium) | 9.98 kΩ |
> | `tempctrl_lna1` | LNA 3 | 9.91 kΩ |
> | `tempctrl_lna2` | Ambient 50 Ω reference load | 9.97 kΩ |
> | `potmon.pot_az_voltage` | SP1 cable, between two turns of the coil | 9.96 kΩ |
>
> The firmware assumes exactly 10 000 Ω (its `T_now` is for heater control
> only). Offline:
>
> - tempctrl channels: R = `resistance` × R_pullup / 10 000 (exact);
> - potmon: R = R_pullup · V / (3.3 − V), with V = `pot_az_voltage`;
> - then Steinhart–Hart, 1/T = A + B ln R + C (ln R)³ (T in K, R in Ω), with
>   A = 9.362744897e-4, B = 2.209138866e-4, C = 1.280636863e-7 (fit to YSI's
>   30 kΩ table over 0–110 °C, residual ≤ 0.019 K; `LOAD_THERM_SH_*` in
>   pico-firmware `tempctrl.h`).
>
> **Interim accuracy.** The pull-ups were measured 2-wire with a handheld
> meter (~0.5–1 %), which leaves ~0.1–0.3 K of uncertainty per channel on top
> of the ±0.2 °C part tolerance. Good enough for the interim run; see § 12.

## 4. Switch schedule

Cycle continuously, with dwells about as long as the field's (~110
integrations, ~30 s):

    RFANT (hot load) → RFAMB → RFNON → RFNOFF → RFANT → …

- Make the cycle **shorter than the field's 13 minutes**, ideally 3–5 minutes,
  so that drift between visits is small.
- If the SP1 cable is fitted, add `RFSP1` with the open and the short every
  few cycles, each dwell at least 1 minute, logging the termination name (as
  `potmon` did in the field).

> **This run.** 10 s per state, a **1-minute cycle**:
>
>     RFANT (ambient ref) → RFAMB (hot) → RFNON → RFNOFF → RFSP1 short → RFSP1 open → …
>
> 10 s is ~37 integrations, minus ~0.5 s flagged after each switch change.
> SP1 is visited every cycle for 10 s per termination (not ≥ 1 min every few
> cycles), so average SP1 across consecutive cycles.

## 5. Temperature program

The hot-load temperature is the lever arm for T_NS, so give it **several
plateaus**, not a single hot stretch followed by a cool-down:

1. **Ambient** (heater off) for ≥ 45 min.
2. Plateaus at about **320, 345 and 370 K**, each held until the thermistor is
   steady to ~0.1 K, then ≥ 30 min more.
3. A **slow cool-down** back to ambient. It gives a continuous range, but the
   thermistor can lag the resistor, so the plateaus carry the result.
4. If there is time, **repeat** the sequence. Two passes test the drift
   directly.

**Board temperature.** If you can change the switch board's temperature (an
enclosure, a heater, or just the room over a day), repeat steps 1–2 at 2–3
board temperatures, about 10 K apart. That gives T_NS against board
temperature, which `tcal` needs to follow the noise source's drift.

> **This run.** Run by `scripts/hot_load_plateaus.py` alongside the observer:
>
>     python scripts/hot_load_plateaus.py --targets 47 67 87 \
>         --ambient-min 60 --hold-min 60 --cooldown-min 90 --passes 2
>
> 1. Ambient (heater off) for 60 min. It is also the lowest point on the hot
>    load's own Q line (295→360 K instead of 320→360 K, about 1.7× smaller
>    slope error), and with the heater off Q ≈ 0 and the hot-load and
>    reference thermistors should agree.
> 2. Plateaus at **47, 67 and 87 °C targets (~320, 340, 360 K)**, not
>    320/345/370 K: the YSI 44909 is rated to 90 °C, so the script refuses
>    targets above 88 °C. Each plateau is held 60 min including the ramp, on
>    a timer rather than until steady; use only the steady part (the script
>    logs a 10-min peak-to-peak every minute). The heater is on/off with
>    0.5 °C hysteresis, so a plateau's mean sits ~0.25 °C below target; use
>    the logged temperature, not the setpoint.
> 3. Cool-down, heater off, 90 min.
> 4. Two passes.
>
> The script turns the heater off and stops on any heater trip or sensor
> error. Restarting `eigsep-panda` resets the heater to off, so restart the
> script too. Board temperature is not stepped: the board thermistors will
> show only the room's natural range.

## 6. VNA measurements

Do these **between** radiometer blocks, never during them: the VNA occupies
the `VNA*` switch states, and see § 7. Each sweep records the internal
standards (`VNAO/S/L`) as in the field.

| What | Path | When |
|---|---|---|
| Hot load, Γ_hot at P | `VNAANT` (then `RFANT` embedded, as for the field antenna) | At every temperature plateau, including ambient |
| Receiver, Γ_rec | `VNARF` | At the start and at the end |
| **Receiver, measured directly** | VNA straight on the first LNA board's input, with the S911T calibration at that connector | At the start and at the end. **This is the independent check.** |
| Ambient load, Γ_amb | `VNAAMB` | Once |
| SP1 open and short | `VNASP1` | Once, if fitted, with the cable temperature logged |

For matched loads, Γ to ~0.01 is enough. With |Γ_rec| ≈ 0.3, a hot-load
reflection of 0.05 still changes its mismatch factor by ~3 %, so do measure the
hot load hot.

> **This run.** The VNA runs automatically every 20 min (and once at
> observer start), with the internal standards, then:
>
> | What | Path (as wired) | When |
> |---|---|---|
> | Hot load, Γ_hot | `VNAAMB` | Every 20 min: ≥ 2 per plateau, including ambient |
> | Ambient reference, Γ_amb | `VNAANT` | Every 20 min |
> | Noise source on / off | `VNANON` / `VNANOFF` | Every 20 min |
> | SP1 open and short | `VNASP1` | Every 20 min, with the cable temperature logged |
> | Receiver, Γ_rec | `VNARF` | Every 20 min (covers start and end) |
> | Receiver, measured directly | VNA on LNA 14's input, S911T at that connector | **Manual**, start and end |
>
> Sweeps are interleaved into the radiometer cycle (they take over the
> switch during an `RFANT` dwell for a few minutes) rather than strictly
> between blocks. Their integrations carry `VNA*` states and are excluded
> via `rfswitch`. The VNA service starts ~5 s **before** it takes the
> switch, so integrations just before each sweep may carry the 1 MHz comb
> (§ 7); flag them by time.

## 7. EMI test (bonus, about 20 minutes)

In the field, three calibration visits that shared a window with an S11 sweep
carried a strong **1.000 MHz comb** on box-air (memo 006). Its source is not
established. With the switch on the hot or ambient load, run three blocks of
≥ 5 minutes each:

1. VNA **sweeping** continuously;
2. VNA **powered, idle**;
3. VNA **off**, or disconnected.

Log the times. A comb present only in block 1 (or 1 and 2) identifies the
source.

> **This run.** Manual, outside the automated run (stop `eigsep-panda`, hold
> the switch on a load). The automated run gives a weaker version for free:
> compare integrations just before each 20-min sweep with the rest.

## 8. Hardware description (with photos)

- What connects the switch common port (P) to the **first LNA stage**: cable,
  adapter or trace, its type and length. This tests the ~8 ns delay.
- Serial numbers of the switch board and the LNA boards, and whether they flew.
  **Were any of the three lab LNA modules in the 09-25 run field units?**
- The hot-load cable and adapters, and their lengths.
- The downstream lab chain: modules, attenuators, filters.

> **This run.** Recorded in `bench_setup` in the config (embedded in every
> file header). LNA modules: lab labels 14 / 3 / 10, **no record of which, if
> any, flew**. Downstream chain: LNA 14 → LNA 3 → LNA 10 → SNAP, no filters
> or attenuators. Still to fill in: switch-board serial and whether it flew,
> the P → LNA 14 connection, the hot-load cable/adapters, the SP1 cable
> length.

## 9. Separate, when convenient: the balun

Measure the antenna balun's **two-port S-parameters**: VNA on one side, S911T
standards on the other, the same method as the switch paths. These are needed
to compare calibrated field spectra with any sky simulation, and to treat the
balun's loss properly below ~60 MHz. The balun–switch coax did not survive,
so a matching spare length would help if one exists.

> **This run.** Not done yet.

## 10. Quick checks during the run

- **No clipping or int32 wraps:** watch `adc_stats` and look for negative
  counts in the autos.
- **The noise-source step** (P_NS − P_load) is clean and steady from cycle to
  cycle.
- **Q of the hot load rises linearly with its temperature** at a few
  channels: the slope of Q against T_hot is 1/T_NS. If the plateaus do not line
  up, stop and check the thermistor contact and the hookup.

> **This run.** Also: with the heater off, Q of the hot load ≈ 0 and the
> hot-load and reference thermistors agree to a fraction of a kelvin;
> `scripts/pico_preflight.py` shows all three picos reporting.

## 11. What to hand over

- The correlator files and the S11 files (`vna_writer` output), as in the
  field.
- The written UTC log (§ 3) and the hardware description and photos (§ 8).
- The correlator configuration used.

**What we will do with it:** compute Q for the hot load at each plateau. The
slope against load temperature gives T_NS (C₁), and the intercept gives the load
constant (C₂), after the mismatch correction from § 6, and per board temperature
if § 5 allows. Then refit the field tier-2 noise waves with the bench Γ_rec, and
rescale `tcal`. References: memo 004 (`memos/memo-004-marjum-2026-07-calibration/`),
the tier-2 study and its next steps
(`data-analysis/scripts/marjum-2026-07/noisewave/README.md`), and memo M003 in
`eigsep_cal/docs/memos/` for the equations.

> **This run.** Also hand over the `hot_load_plateaus` log and
> `obs_config_switch_bench.yaml` as used. Q is computed for the hot load
> (`RFAMB`) against the ambient reference (`RFANT`); the bench Γ_rec carries
> the lab-LNA caveat of § 1.

## 12. Gaps between this run and the requirements

| Requirement | This run | Effect | To meet the requirement |
|---|---|---|---|
| Field first LNA board (§ 1–2) | Lab LNA 14; flight history unknown | T_NS unaffected; Γ_rec and noise waves may not transfer | Identify and use the field first-stage board |
| Hot load on `RFANT`, ambient on `RFAMB` (§ 2) | Swapped: hot on `RFAMB`, ambient on `RFANT` | Use the matching paths (`VNAAMB` hot, `VNAANT` ambient) | None needed if the analysis follows the wiring; or rewire |
| Plateaus ~320 / 345 / 370 K (§ 5) | ~320 / 340 / 360 K | Span 65 K instead of 75 K | Thermistors rated above ~100 °C (higher dynamic range) |
| Plateau held until steady to ~0.1 K, then ≥ 30 min (§ 5) | Fixed 60 min per plateau; steady part chosen offline | Possibly shorter steady windows | Hold-until-steady logic in the sequencer, or longer holds |
| Temperatures good to ~0.1 K (§ 5) | ±0.2 °C parts; 2-wire pull-ups (~0.1–0.3 K) | Temperature scale uncertainty of a few tenths of a kelvin | Better thermistors, 4-wire pull-ups or an ice-point / reference-thermometer calibration |
| ~30 s dwells, 3–5 min cycle (§ 4) | 10 s dwells, 1 min cycle | Fewer integrations per visit, less drift between visits | Change `switch_schedule` |
| SP1 every few cycles, ≥ 1 min each (§ 4) | Every cycle, 10 s each | Average SP1 across cycles | Change `switch_schedule` |
| VNA strictly between radiometer blocks (§ 6) | Interleaved every 20 min | Sweep integrations `VNA*`-flagged; integrations just before a sweep may carry the comb | Pause switching around sweeps, or schedule sweeps explicitly |
| 2–3 board temperatures (§ 5) | Not stepped | Only the room's natural range | Board enclosure or heater |
| Supply voltages logged (§ 3) | Written log only | Manual | A logged supply monitor |
