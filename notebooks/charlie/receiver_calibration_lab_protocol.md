# Receiver temperature calibration — lab measurement protocol

For Charlie. The goal is to put the 2026 field receiver on an absolute kelvin
scale. Today the field calibration (`tcal`, memo 004) rests on the noise
source's **nameplate** excess, T_NS = 917 K. The only in-band estimate is 2.4
times lower and cannot be trusted either. This run measures T_NS directly, the
way EDGES does (their constant C₁), and checks the receiver reflection that the
tier-2 noise-wave fit cannot settle on its own.

> **Bench runs.** This protocol states the requirements for the final
> measurement. Each bench session is recorded as built, with where it falls
> short of these requirements, in its own folder under
> [`bench_runs/`](bench_runs/README.md), one per run date. A run record is not
> edited after the run; the next run gets a new folder.

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

## 4. Switch schedule

Cycle continuously, with dwells about as long as the field's (~110
integrations, ~30 s):

    RFANT (hot load) → RFAMB → RFNON → RFNOFF → RFANT → …

- Make the cycle **shorter than the field's 13 minutes**, ideally 3–5 minutes,
  so that drift between visits is small.
- If the SP1 cable is fitted, add `RFSP1` with the open and the short every
  few cycles, each dwell at least 1 minute, logging the termination name (as
  `potmon` did in the field).

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

## 8. Hardware description (with photos)

- What connects the switch common port (P) to the **first LNA stage**: cable,
  adapter or trace, its type and length. This tests the ~8 ns delay.
- Serial numbers of the switch board and the LNA boards, and whether they flew.
  **Were any of the three lab LNA modules in the 09-25 run field units?**
- The hot-load cable and adapters, and their lengths.
- The downstream lab chain: modules, attenuators, filters.

## 9. Separate, when convenient: the balun

Measure the antenna balun's **two-port S-parameters**: VNA on one side, S911T
standards on the other, the same method as the switch paths. These are needed
to compare calibrated field spectra with any sky simulation, and to treat the
balun's loss properly below ~60 MHz. The balun–switch coax did not survive,
so a matching spare length would help if one exists.

## 10. Quick checks during the run

- **No clipping or int32 wraps:** watch `adc_stats` and look for negative
  counts in the autos.
- **The noise-source step** (P_NS − P_load) is clean and steady from cycle to
  cycle.
- **Q of the hot load rises linearly with its temperature** at a few
  channels: the slope of Q against T_hot is 1/T_NS. If the plateaus do not line
  up, stop and check the thermistor contact and the hookup.

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
