# B7 — First absolutely-scaled spectra from Marjum 2026-07, and what the missing lab data costs

_rf-calibrator, 2026-09-12. Status: first calibrated window exists. Memo-ready draft for `scribe`._

---

## Claim

1. The July 2026 campaign contains a **usable in-band absolute-calibration
   dataset**, and a first-order calibrated product
   $T_{\rm ant}^{\rm (sw)}(\nu,t)$ now exists for an 8-hour Phase-C window.
   The receiver is well-behaved: $T_{\rm rx} = 381$ K, stable to 6.92 K rms
   over 8 hours (55–85 MHz), gain drift 2.8 % rms.
2. **The ENR provenance question is answered, in-band.** It was never missing:
   the campaign `obs_config` carries a `calibration` block. It is a *nameplate*
   value, not a measurement.
3. **The product is not yet a sky spectrum, and the data proves it.** The
   calibrated spectrum has $\beta = +0.419$ **over 55–85 MHz** where synchrotron
   requires $\beta \approx -2.5$ (band-critical: the same product gives
   $\beta = -0.961$ over 55–115 MHz and $+0.207$ over 60–80 MHz — every
   quotation of $\beta$ must name its band). The whole discrepancy is uncorrected antenna
   response.
4. **The missing lab measurements cost 354 K (19.2 %) of absolute scale.**
   Lab-measuring the ENR alone would cut that 10× to 35 K (1.9 %).

---

## Data recipe

| | |
|---|---|
| Window | 2026-07-17 04:00:00Z → 11:57:05Z, Phase C |
| Selection | `curation/select_files.py --phase C --start … --end …` → **223 files, 0 dropped** |
| Switched input | **4** (`box-air`, the suspended bowtie) |
| Cal cycles | 38 usable of 38; RFNON/RFAMB pairs on a 12-min cadence |
| RFANT spectra | 215 |
| `corr_acc_len` | uniform — window is entirely post-doubling (pivot 07-15 15:55 UTC) |
| Masks | none triggered; SNAP-flip and outage windows all outside this range |

**How input 4 was identified.** Across one RFNON/RFAMB dwell, input 4 moves by
a factor 2.3 (4.94e5 → 1.15e6 counts) while input 0 (`box-gnd`) moves < 2 %.
Only input 4 is behind the RF switch. This is empirical, not assumed.

---

## Method + provenance

Three-state Y-factor, deliberately reusing the formalism in
`eigsep_observing/src/eigsep_observing/live_status/calibration.py` — the code
that actually ran in the field — so the field dashboard and this pipeline
cannot silently disagree:

$$G(\nu) = \frac{P_{\rm on}-P_{\rm amb}}{T_{\rm hot}-T_{\rm amb}}, \qquad
T_{\rm rx}(\nu) = \frac{P_{\rm amb}}{G} - T_{\rm amb}, \qquad
T_{\rm in}(\nu) = \frac{P}{G} - T_{\rm rx}$$

References, all from the campaign's own `header/obs_config.calibration`:

```json
{"noise_diode_enr_db": 35.0, "noise_source_atten_db": 30.0,
 "t_ns_stream": "rfswitch_therm", "t_ns_field": "temp_therm2",
 "t_amb_stream": "tempctrl_load", "t_amb_field": "T_now"}
```

$T_{\rm ENR} = 290\,{\rm K}\cdot 10^{(35.0-30.0)/10} = 917.1$ K;
$T_{\rm ns} = 304.93$ K; $T_{\rm amb} = 300.22$ K; $T_{\rm hot} = 1222.0$ K.

Code: `abscal/` on branch `abscal-b7-first-calibration`. **Both products were
re-cut at `HEAD` after the int32-wrap guard landed**, so a single hash covers
every number in both memos — see the provenance table at the end.

### Reference plane — read this before using the product

The switched reference plane is the **RF switch common port** inside the
receiver box: the point at which the RFAMB load and RFNON noise source are
presented. Everything between that plane and the antenna terminals — balun,
feed cable, connectors — is **absorbed into the reported $T_{\rm ant}$**, not
corrected. Antenna reflection and radiation efficiency are likewise **not**
removed.

So the product is *the input temperature at the switch common port when the
switch selects the antenna*. It is on an absolute Kelvin scale. It is **not**
a sky brightness temperature, and it must not be handed to `forward-modeler`
as one.

---

## Residuals and validation

Three checks, none of which is the training residual of the fit that produced
the calibration:

**1. Adjacent-cycle closure** — apply cycle $i$'s solution to cycle $i{+}1$'s
ambient load. Not algebraically forced, so it is informative.
→ median $-0.10$ K, **rms 1.96 K**, max 4.93 K, **in the 55–85 MHz band**.
That is the real cost of interpolating the calibration across one 12-min
cadence step, and it is small.

*(Corrected: earlier drafts quoted $-0.15$ / 1.93 / 4.92, which are the
55–115 MHz figures — the third instance of the same band-leak that produced
the 1:225 and 6.6 K errors. All script defaults are now aligned to 55–85 MHz
and `validate.py` prints its band as a header. Root cause fixed, not just the
number.)*

**2. Spectral index** — $\beta = +0.419$ over **55–85 MHz** (FM band excluded).
Band-critical: 55–115 MHz gives $-0.961$, 60–80 MHz gives $+0.207$. IN-SAMPLE
(the power law is fitted to this same median spectrum), so it is a shape
diagnostic, not a validation.
Sky requires $\approx -2.5$. **This is the expected consequence of the stated
reference plane**, not a pipeline failure: the bowtie response peaks near
80 MHz and is entirely uncorrected. It is also the sharpest available evidence
that the reflection/noise-wave chain is not optional.

**3. $T_{\rm rx}$ / $T_{\rm ant}$ leakage** — $T_{\rm rx}$ tracks the physical
thermometry, $r(T_{\rm rx},T_{\rm ns}) = +0.74$, $r(T_{\rm rx},T_{\rm amb}) =
+0.76$, with $T_{\rm rx}$ swinging 31.5 K while the physical references move
only 2.5 K. A ~12 K/K lever. Plausibly a real LNA thermal coefficient, but
**unverified**, and it is the leading candidate for a time-dependent
systematic. Flagged, not resolved. `tempctrl_lna` is **0 % usable** across the
window (all error records), so LNA physical temperature cannot currently be
used to model it.

---

## Uncertainty budget — 55–85 MHz, $T_{\rm ant}^{\rm (sw)} = 1845$ K

$R \equiv (T_{\rm ant}-T_{\rm amb})/(T_{\rm hot}-T_{\rm amb}) = 1.675$.
Partials are exact (see `abscal.propagate`).

| Term | σ [K] | % | Basis |
|---|---:|---:|---|
| Radiometric noise (per 8 s) | 1.21 | 0.07 % | measured from data |
| Cal-cycle interpolation (12 min) | 1.96 | 0.11 % | **measured** (closure test) |
| $T_{\rm amb}$ statistical | 0.01 | 0.00 % | ADC rms 0.28 K averaged per cycle |
| $T_{\rm amb}$ systematic | 1.35 | 0.07 % | assumed ±2 K — **unverified** |
| $T_{\rm ns}$ systematic | 3.35 | 0.18 % | assumed ±2 K — **unverified** |
| **Effective ENR** | **353.8** | **19.2 %** | **nameplate ±1 dB — unverified** |

Statistical : systematic = **1 : 292** *in this table's 55–85 MHz band*
(353.8 / 1.21). This measurement is not noise-limited
and never will be; more integration buys nothing.

### What the lab measurement is worth

| σ(ENR_eff) | σ(T_ENR) | σ(T_ant) | % | Provenance |
|---:|---:|---:|---:|---|
| 0.10 dB | 21 K | **35 K** | 1.9 % | lab-measured diode + pad — achievable |
| 0.25 dB | 53 K | 88 K | 4.8 % | lab diode, datasheet pad |
| 0.50 dB | 106 K | 177 K | 9.6 % | manufacturer certificate |
| **1.00 dB** | **211 K** | **354 K** | **19.2 %** | **nameplate — what we have** |
| 2.00 dB | 422 K | 708 K | 38.4 % | nameplate, aged diode |

**The Christian ask, as a number: lab-measuring the noise diode and its 30 dB
pad converts a 354 K absolute-scale systematic into a 35 K one — a 10×
reduction, for one afternoon on a bench.**

Against the EIGSEP paper's own 1-part-in-$10^4$ standard (0.184 K here) we are
short by **1918×** now, and would still be short by **192×** with a
lab-measured ENR. The remaining gap is the reflection/noise-wave chain.

### The distinction that matters for the program

An ENR error has partial $R$ — smooth in frequency, very nearly a **pure scale
error**. It limits any claim tied to absolute amplitude, but it does *not*
manufacture spectral structure and so does not by itself threaten a
21-cm-shaped residual.

The terms that *do* manufacture spectral structure are exactly the ones this
dataset cannot bound at all:

- $\Gamma_{\rm ant}$, $\Gamma_{\rm rx}$ — the VNA writes to its own files
  (`vna_writer` → `stream:vna`); **none are in the repo**, only the *timing* of
  ~25 SOL sweeps survives in the switch axis.
- The five noise-wave parameters — no four-standard lab measurement.
- Balun loss and radiation efficiency — HFSS only.

$\beta = +0.419$ (55–85 MHz) is the measured evidence that these are large.

---

## Findings for other agents

**For `data-archivist`:**

- **Whole-file clock glitch — already masked, and the mechanism is now clear.**
  `corr_20260717_032454Z.h5` carries `header/times` for
  **2026-07-09T01:08:01Z — all 240 samples** — eight days before its own
  filename and three days before campaign start. Sample spacing within the file
  is normal (0.537 s), so the timing is not corrupted; the epoch base is wrong.

  *Correction to an earlier draft of this memo, which claimed
  `select_files.py` does not drop it. It does.* Verified behaviourally: the
  file falls inside the **`snap-test-flip`** mask (`CAMPAIGN.md:149`,
  03:24:54–03:27:03 UTC 07-17) and `select_files.py --phase C` drops it with
  the reason *"SNAP swapped to scratch board C000069 (not a flight
  configuration)"*.

  That also explains the glitch: the timestamps come from the **scratch SNAP
  board**, which evidently carries its own epoch. So this is not an
  independent defect needing a new mask — it is a *symptom* of the SNAP flips,
  already correctly handled. Useful as corroboration that the eight SNAP-flip
  windows really are board swaps, and as a cheap detector for any further ones:
  a pre-campaign `header/times` is a board-swap fingerprint.
- `tempctrl_lna` is **0 % usable** (4773/4773 error records sampled) across
  07-17 04:00–10:00. LNA physical temperature is unavailable campaign-wide as
  far as I have checked.
- `tempctrl_load.sensor_tripped` is set on 61.4 % of records and
  `enabled` is `false` on 100 % (consistent with `use_tempctrl: false`). The
  latch is *sticky* and gates heater drive, not the reading. I verified the
  reading is still usable: per-sample ADC noise is real (rms 0.28 K, max jump
  1.18 K — the latch was justified) but the per-cycle **mean** is solid and
  correlates $+0.92$ with the independent `rfswitch_therm` thermistors.
  **Conclusion: $T_{\rm amb}$ is usable; the latch is stale.** Suggest
  recording that adjudication in `INDEX.md` so the next consumer doesn't
  re-litigate it.
- Correction to the B7 brief framing: "in-band 1-port SOL cal + antenna S11"
  overstates what exists. The SOL/VNAANT **states** are in the stream; the S11
  **data** is not. I tested for in-band reflectometry directly — cross-product
  coherence during VNA states is 0.001, i.e. none. Suggest `cal_windows.jsonl`
  label these `vna_timing` rather than implying reflection data.

**For `experimental-strategist` / the Christian escalation:** the SP1
reflection standard ran **SHORT only** for the whole campaign
(`sp1_term_name: "SHORT"`, failsafe default). `RFSP1_OPEN` never executed —
the campaign ran the pre-feature schedule (`switch_schedule` has no RFSP1 key
at all by 07-17). So even the one in-band reflection standard gives a single
termination, not the open/short pair the acceptance test wants. That is a
**next-deployment fix**, not something recoverable from this dataset.

---

## What I am not claiming

- Not a sky spectrum. Not a foreground measurement. Not an input to
  foreground separation in this form.
- The ±2 K on the thermistor systematics is an *assumption*, flagged as such,
  not a propagated measurement. It is currently negligible next to the ENR
  term, so it is not worth chasing until the ENR is fixed.
- The 12 K/K $T_{\rm rx}$ thermal lever is unexplained. If it is instrumental
  rather than real it would be a time-dependent systematic, and with
  `tempctrl_lna` dead I cannot currently distinguish the two.

---

# Addendum — $T_{\rm rx}(\nu,t)$ across Phase C, and a falsified assumption

_Added 2026-09-13, after extending to the full cadence._

## Data recipe

66 cal cycles over **22.8 h**, 2026-07-17 04:11 → 2026-07-18 02:57, taken from
`curation/cal_windows.jsonl` ∩ `select_files.py --phase C` (1 of 67 candidate
windows dropped — all its files were masked by curation). Code: `extend_trx.py`,
`regimes.py`.

## Claim 1 — the 12 K/K thermal lever was co-trending, not causal

The first memo flagged an unexplained $\sim$12 K/K coupling between
$T_{\rm rx}$ and the physical thermometry ($r \approx +0.75$ over 8 h). The
longer baseline resolves it. Within the long segment:

| | |
|---|---|
| $r(T_{\rm rx}, T_{\rm ns})$ | $+0.554$ (slope $+15.0$ K/K) |
| $r(T_{\rm rx}, {\rm time})$ | $\mathbf{-0.904}$ |

Time is by far the better predictor. The 8-hour correlation was two quantities
drifting in the same window, not a thermal coefficient. **Withdrawn as a
thermal effect.**

## Claim 2 — the drift is in the noise injection, not the receiver

$T_{\rm rx}$ falls 400 → 206 K across 07-17 while $g_{\rm rx}$ rises
872 → 1167 counts/K, $r = -0.993$. That anticorrelation on its own proves
nothing — $T_{\rm rx} \equiv P_{\rm amb}/g - T_{\rm amb}$, so any error in $g$
maps straight into $T_{\rm rx}$ by construction. (Likewise
$(T_{\rm rx}+T_{\rm amb})\cdot g \equiv P_{\rm amb}$ is an algebraic identity,
not a test — an earlier draft of `regimes.py` mislabelled it as one.)

Decomposing instead into the two genuinely independent measured powers, over
the 59-cycle segment 07-17 04:11 → 16:23:

**Primary statistic — $(Y-1) = (P_{\rm on}-P_{\rm amb})/P_{\rm amb}$,
formed per channel.** This is invariant under a common scaling of every power,
so a receiver-gain explanation cannot survive in it *by construction* — it
retires the common-mode objection without argument. Over the 59-cycle segment:
**rms 3.77 %, ptp 16.12 %, $r(\cdot,t) = +0.904$.** The drift survives intact.

Supporting decomposition:

| Observable | rms | ptp | $r(\cdot,t)$ |
|---|---:|---:|---:|
| $P_{\rm amb}$ — receiver on a **fixed passive load** (the control) | **0.76 %** | 5.91 % | $+0.336$ |
| $P_{\rm on}-P_{\rm amb}$ — the **injected excess** | 3.81 % | **15.24 %** | $\mathbf{+0.929}$ |

with $r(\text{control}, \text{excess}) = +0.136$ — so the drift is **not**
common-mode — and the excess **5.0× less steady** than the control.

> **Band-reduction order matters, and it caught me.** The excess at channel $i$
> is $P_{\rm on}[i]-P_{\rm amb}[i]$. The median is not linear, so reducing each
> state to a band median *before* subtracting compares the median channel of
> $P_{\rm on}$ against a different channel of $P_{\rm amb}$. That mixing inflates
> $r(\text{control},\text{excess})$ from $+0.136$ to $+0.460$ and the excess rms
> from 3.81 % to 3.89 %. **Difference first, always.** An earlier draft — and a
> correction I sent `scribe` insisting on the larger value — used the wrong
> order; `scribe` diagnosed it. Band *mean* is unusable throughout (89 % rms):
> a few RFI channels swamp it. Median everywhere.

The receiver's response to an unchanging passive load is **five times
steadier** than the injected noise excess, which climbs monotonically by 15 %.
(Not "the receiver is stable" — the control carries its own mild trend,
$r(\cdot,t)=+0.336$. The weaker claim is the correct one.) Two alternatives
are excluded independently:

- **Receiver gain drift** — excluded *algebraically*: if every power scales by
  a common $k$, then $g \to kg$ and $T_{\rm rx} = kP_{\rm amb}/(kg) -
  T_{\rm amb}$ is invariant. Gain drift cancels exactly out of $T_{\rm rx}$.
- **Receiver compression** — excluded *empirically*: the drift is
  frequency-independent to 1.7 % (55–85 MHz: ×1.153, FM 88–108: ×1.124,
  120–180: ×1.107 at the last cycle). Compression would bite hardest in the
  brightest channels; the FM band drifts *least*.

**Conclusion: the noise-source output is drifting relative to the ambient
load.** This is the quantity the nameplate ENR asserts is a constant.

## Claim 3 — the constant-nameplate-ENR assumption is falsified

Implied effective excess, relative to the nameplate 917.1 K:

| Segment | $T_{\rm ENR}$ implied | vs nameplate |
|---|---:|---:|
| 07-17 04:11 → 16:23 | 951 K | +3.7 % |
| 07-17 19:18 → 19:42 | 1182 K | **+28.9 %** |
| 07-18 01:24 | 691 K | **−24.7 %** |
| 07-18 02:31 → 02:57 | 653 K | **−28.8 %** |

**This changes the character of the ENR systematic, and I am walking back part
of the first memo.** I argued there that an ENR error is a smooth *scale*
error, and therefore limits absolute amplitude without manufacturing spectral
structure. The frequency-independence result (1.7 %) confirms the *spectral*
half of that claim. But the drift makes the error **time-dependent**, and that
half was wrong: a 15 %/12 h drift in the assumed excess injects
$\sim$200 K of spurious slow time structure into $T_{\rm ant}$ over a
half-day. Anything binned or fitted in LST — which is most of what the program
wants to do — is exposed to it.

Two mitigations, and they are not equivalent:

1. The 12-min cadence **tracks** the drift. Solving per cycle, as this pipeline
   does, keeps each $T_{\rm ant}$ internally consistent. This is why the cadence
   is doing real work and must not be averaged away.
2. Nothing here recovers the **absolute** scale. Per-cycle solving removes the
   time dependence *given* a correct $T_{\rm ENR}$; it cannot supply one.

So the bench measurement is now worth more than the first memo argued — it is
not only an absolute-scale fix but the only way to know whether the drift is in
the diode, the pad, or the switch path.

## Claim 4 — the calibration data independently detects the 07-17 power event

There is a discrete regime change between 07-17 19:42 and 07-18 01:24:
$T_{\rm rx}$ 212 → 539 K, $g_{\rm rx}$ 1150 → 672. The gap brackets two logged
events — the **7U sky battery failure ~22:00 UTC 07-17** and the **box-air
outage ~23:35 UTC 07-17** (`CAMPAIGN.md:166`). The post-event receiver is in a
materially different state, not a continuation of the drift.

This speaks directly to `CAMPAIGN.md:185`, data-archivist's open question:
*"Identify the 7U sky battery failure signature in the Phase-C data around
22:00 UTC 07-17. Cross-check against auto power levels."* **The switched-load
solutions are a sharper probe than the auto power levels**, because they
separate the receiver's response to a fixed load from the injected noise.
I cannot date the transition more precisely than the 5.7-hour observing gap;
the RFANT autos through the gap could narrow it, and that is archivist's call.

**Calibration products from before and after that boundary must not be
interpolated across it.** `cal_windows.jsonl` does not currently carry a
regime marker; recommend one.

## What I am not claiming here

- Not that the diode specifically is the drifting element. The measurement
  localises the drift to the **injected-noise path relative to the ambient
  load** — diode, pad, switch isolation, or a connector are all still live.
  Separating them needs the bench.
- The post-event segments are 1 and 3 cycles. The regime change is unambiguous;
  its *level* is not well determined.
- No RFNOFF dwells exist in Phase C (`switch_schedule` is
  `{RFANT: 600, RFNON: 60, RFAMB: 60}`), so the offline noise-off cross-check
  that would most cleanly isolate the diode is unavailable. **That is a
  next-deployment ask: restore RFNOFF to the schedule.**

---

# Addendum 2 — two negative results, and a cheap fix worth more than either

_Added 2026-09-13._

## Claim 5 — diurnal vs monotonic cannot be answered from this campaign

Extending the selection back to the start of Phase C (2026-07-15 00:32) adds
**zero** cal cycles. The reason is in the cadence history, not the code:

| Phase | Windows with both RFNON and RFAMB |
|---|---:|
| A | 1 (2026-07-13 16:23) |
| B | **0** |
| C | 67 — of which **62 on 07-17**, 4 on 07-18, 1 on 07-16 |

The single 07-16 Phase-C window is dropped by curation. So the entire usable
switched-load record is **22.8 h beginning 07-17 04:12** — which is when
Christian set up the cadence (`CAMPAIGN.md:134`, ~04:00 UTC 07-17). Before
that, switched measurements were sporadic.

**Verdict: the 15 %/12 h injection drift cannot be classified as diurnal or
monotonic.** The record contains one continuous 12.2-hour run, a 3-cycle stub,
and a 4-cycle post-event stub, and the largest excursion is confounded with the
battery failure. Distinguishing a thermal diode coefficient from a degrading
component needs either the bench or a next deployment. I would not spend more
analysis time on it.

*(Housekeeping: `abscal.solve_gain_trx` now refuses to pool dwells across a
`corr_acc_len` change, and reports gain as counts/**s**/K. The doubling at
07-15 15:55 UTC turns out not to bite — every solved cycle is post-doubling,
dt = 0.5369 s uniformly — but the guard should exist before anyone extends into
Phase B.)*

## Claim 6 — there is an in-band route to the ENR, and this campaign just misses it

Where an **RFNOFF** dwell exists alongside RFAMB and RFNON, the dominant budget
term can in principle be measured in situ, with no bench at all. RFAMB and
RFNOFF are both *passive* loads at independently-known physical temperatures,
so they give the gain without involving the diode:

$$g = \frac{P_{\rm noff}-P_{\rm amb}}{T_{\rm ns}-T_{\rm amb}}, \qquad
T_{\rm ENR} = \frac{P_{\rm non}-P_{\rm noff}}{g}$$

and because $P_{\rm non}$ and $P_{\rm noff}$ share a physical path, path loss
and mismatch largely cancel — cleaner than the $P_{\rm on}-P_{\rm amb}$ route
the pipeline currently uses. Such triplets exist: 2026-07-13 16:23 (Phase A)
and 2026-07-16 01:15 (Phase C).

Running it on the Phase-C triplet gives $T_{\rm ENR} = 360$ K against a
nameplate of 917 K — a factor of 2.5, which would be a dramatic result.
**It is not a result, and I am not reporting it as one.**

$T_{\rm ENR}$ is *linear* in the lever $T_{\rm ns}-T_{\rm amb}$, and that lever
is measured as **+4.30 K**. But the two temperatures come from different
thermometer families, and their offset is:

| | |
|---|---|
| `therm2 − T_now`, over 167 files / ~6 h | **+4.561 ± 0.133 K** (ptp 0.587 K) |
| individual swing of each sensor over the same span | 2.1–2.4 K |

The two sensors track each other tightly with a **fixed offset the same size as
the entire lever**. Nothing in this dataset distinguishes *"the noise-source
load really is 4.5 K hotter"* from *"these two thermometers disagree by 4.5 K"*.
Propagating the offset as the lever uncertainty gives
$\sigma(T_{\rm ENR}) = 382$ K — **106 % of the value**, with a 2σ interval of
[0, 1124] K that **includes the nameplate**. Consistent with 917 K; incapable of
testing it.

### The route does not substitute for the bench — I was wrong about that

I previously recommended the cross-calibration *ahead of* the bench
characterisation, on the grounds that it would rescue the in-band ENR for this
campaign retroactively. **Two further checks show it will not, and I am
withdrawing that ordering.**

**Check A — the second triplet is not a triplet.** The 07-13 Phase-A block
(the only other RFNOFF/RFAMB/RFNON set in the campaign) gives, on input 0 in
55–85 MHz:

| state | power |
|---|---|
| RFAMB | 1.0902e6 |
| RFNOFF | 1.0902e6 |
| RFNON | 1.0903e6 |

All three identical to four significant figures — the noise source ON changes
nothing. The switch was not affecting that input on 07-13. (Consistent with
`CAMPAIGN.md:134`: the noise-source cadence was not set up until 07-17.) It
yields an unphysical negative gain and is **not evidence about anything**. So
there is exactly **one** usable triplet in the whole campaign, not two, and no
independent cross-check.

**Check B — the passive-pair premise fails a closure test.** On the single
usable (07-16) triplet, the two routes to the gain disagree:

| route | gain |
|---|---:|
| passive pair, $(P_{\rm noff}-P_{\rm amb})/(T_{\rm ns}-T_{\rm amb})$ | 1761 counts/K |
| Y-factor, $(P_{\rm non}-P_{\rm amb})/(T_{\rm hot}-T_{\rm amb})$ | 728 counts/K |
| **ratio** | **2.42** |

and that factor is exactly the 2.44 by which the in-band $T_{\rm ENR}$ falls
below nameplate — as the algebra requires, since both follow from the same
ratio. Independently: a 4.31 K lever at $T_{\rm sys}\approx 680$ K predicts
$\Delta P/P = 0.63\%$, and the observed step is **1.27 %**, twice as large.

So the premise that RFNOFF and RFAMB present as *passive loads at their
measured physical temperatures* is not satisfied. The noise-source path
carries the 30 dB pad and its own switch losses; its effective input
temperature is not simply `temp_therm2`. Two explanations remain degenerate:

1. the nameplate $T_{\rm ENR}$ is ~2.4× too high (and $T_{\rm ant}$ is
   therefore ~2.4× too high — it would be ~910 K, not 1845 K); or
2. the passive-pair lever is ~2× larger than the thermistors report, or the
   two paths differ in loss.

**Nothing in this dataset separates them.** A cross-calibration would drive
the *lever* uncertainty to 0.13 K and give $\sigma(T_{\rm ENR}) = 30$ K (8 %,
now radiometrically limited) — but on a number that disagrees with nameplate by
2.4× for unknown reasons. **That is precision without accuracy**, and it is
worse than useless if quoted: it would look authoritative.

Resolving it requires knowing the path losses, which is the bench measurement.
So **(c) is not substitutable — it is prerequisite.** Correct ordering:
bench-characterise diode + pad + switch path *first*; the cross-calibration and
the heater then make the in-band route a continuous monitor rather than an
independent determination.

*Flag for whoever carries the absolute scale downstream: there is an
unresolved factor of ~2.4 on $T_{\rm ant}$. The 354 K (19.2 %) ENR systematic in
the budget is a 1-sigma spread about the nameplate; it does **not** cover this
discrepancy, which is a possible ~2.4× scale error and is currently
uncharacterised in either direction.*

### The actionable part

The obstruction is cheap to remove, but only *after* the bench. Two fixes, both
next-deployment, both bearing on the term that dominates the entire B7 budget
at 354 K:

1. **Cross-calibrate the `tempctrl_load` thermistor against the
   `rfswitch_therm` family.** A bench afternoon, no field time. This alone
   converts the lever from degenerate to usable.
2. **Create a real lever — heat the ambient load.** `use_tempctrl` was `false`
   all campaign and the LOAD channel carried a sticky `sensor_tripped` latch,
   so the heater never ran. A commanded 20 K rise turns a degenerate 4 K lever
   into a 20 K one and makes this an in-band ENR good to a few percent —
   *continuously, every cycle, in situ*, which is strictly better than a
   one-time bench number because it would also track the 15 % drift.

That second point is the one I would put in front of Christian. The bench
measurement fixes the absolute scale once; **enabling the load heater would
measure the ENR on the same 12-minute cadence as everything else**, and the
drift result in Addendum 1 is exactly the argument for why a one-time number
is not sufficient.

## Next

1. Chase whether the VNA S11 files exist on the T7 — blocked on Aaron mounting
   the drives, escalated, not mine to chase.
2. Hand the memo to `scribe` for MEMO-series numbering.
3. Nothing further on diurnal-vs-monotonic; the data does not support it.

---

# Addendum 3 — int32 wrap audit, figures, provenance

_Added 2026-09-13, in response to natural-experimenter MEMO-006 and scribe's review._

## int32 accumulator wrap — the B7 solve is clean

MEMO-006 reports the int32 auto accumulators wrapping on the brightest
channels: 11.7 % of files campaign-wide, **25.6 % in the post-doubling era**,
which is exactly this calibration window. Audited directly (`audit_wrap.py`),
over the 637 curated Phase-C files 07-17 04:00 → 07-18 04:00, on input 4:

| state | samples | wrapped | wrapped in 55–85 MHz |
|---|---:|---:|---:|
| RFAMB | 8,499,200 | **0** | **0** |
| RFNON | 7,651,328 | **0** | **0** |
| RFANT | 120,713,216 | 2,114 (0.0018 %) | 3 |

**The states that produce every calibration solution — RFAMB and RFNON — carry
zero wrapped samples anywhere in the window.** So $g_{\rm rx}$, $T_{\rm rx}$ and
every number in MEMO-004 are structurally immune.

For $T_{\rm ant}$: within the MEMO-003 window (07-17 04:00–12:00) the wrapped
channels are 99.61, 104.74, 109.86, 125.49, 127.93, 129.88, 130.13, 131.59,
131.84, 133.30, 133.54 and 135.74 MHz — **none inside 55–85 MHz**. Across the
wider Phase-C span there is exactly one in-band occurrence: 76.42 MHz, 3
samples, in `corr_20260717_211956Z.h5` at 21:17 UTC — which falls inside the
19:42 → 01:24 observing gap, so no cal cycle uses it.

A guard now masks negative accumulations to NaN in `abscal.read_file`, with
nan-aware reductions throughout. **Both products were re-cut with the guard
active and every reported number is unchanged to the last printed digit**
($g$ 746.10, $T_{\rm rx}$ 380.7 K, $T_{\rm ant}$ 1526.4 K, $\sigma$ 282.0 K;
Phase C 1709.36 counts/s/K, 372.91 K, rms 56.81 K, ptp 333.03 K). That
invariance is the check: the guard is defensive, not corrective.

## Figures

All at 55–85 MHz; band stated in every caption; UTC on time axes.

| Figure | Content | Residual status |
|---|---|---|
| `figs/fig1_closure_residual.png` | Cross-closure residual vs frequency, 37 adjacent pairs + 2-D (pair × channel) map | **HELD OUT** |
| `figs/fig2_spectrum_residual.png` | $T_{\rm ant}(\nu)$ with $\beta$ fit and $-2.5$ reference, residual panel | in-sample |
| `figs/fig3_control_vs_excess.png` | $P_{\rm amb}$ (control) vs injected excess, both normalised to segment mean | in-sample |
| `figs/fig4_trx_timeseries.png` | $T_{\rm rx}(t)$ over 22.8 h, gaps broken, battery/outage marked | in-sample |

Two notes on fig 1. The scalar quoted in Claim "residuals and validation" —
**1.93 K** — is the rms of the *band-median* residual across the 37 pairs. The
figure's **2.13 K** is the rms over all (pair × channel) cells. Different
quantities, both correct; the band-median is the one that matters for a
calibration applied per cycle.

And the 2-D map shows the residual is dominated by **horizontal banding** —
per-cycle offsets that are near-frequency-independent. That is an independent
corroboration of Addendum 1: a scalar per-cycle drift, not a spectral
calibration error.

## Provenance

Both `.npz` products were regenerated at the same code state, after the wrap
guard, so one hash covers every number in both memos. Products are gitignored
and regenerate via the recipe in `abscal/README.md`.

| Product | Produced by | Code state |
|---|---|---|
| `window_0717.npz` | `run_window.py --phase C --start 2026-07-17T04:00:00Z --end 2026-07-17T12:00:00Z` | `2b398b5` |
| `trx_phaseC.npz` | `extend_trx.py` (defaults) | `2b398b5` |
| all four figures | `make_figures.py` | `2b398b5` |

Branch `abscal-b7-first-calibration`, off `main` at `a2a5927`.

`window_0717.npz` and `trx_phaseC.npz` now also carry `p_amb`, `p_on`, `n_on`,
`n_amb` and `dts`, so the Addendum-1 control is checkable directly from the
shipped file without re-running the scan.

## Corrections to earlier drafts of this memo

1. stat:syst was quoted as **1 : 225**; that is the 55–115 MHz figure. In this
   memo's stated 55–85 MHz band it is **1 : 292**. Corrected.
1b. Closure was quoted as median $-0.15$ K / rms **1.93** K / max 4.92 K; those
   are also 55–115 MHz. In 55–85 MHz: $-0.10$ / **1.96** / 4.93. Corrected.
   Three band-leaks from the same cause — `run_window.py` defaulted to
   50–120, `validate.py` to 55–115, the memo states 55–85. **All defaults are
   now 55–85 and `validate.py` prints its band as a header line.**
2. $T_{\rm rx}$ stability was quoted as 6.6 K rms; that is the 50–120 MHz
   default band. In 55–85 MHz it is **6.92 K**. Corrected.
3. $\beta$ now carries its band at every appearance — the same product gives
   $+0.419$ (55–85), $-0.961$ (55–115) and $+0.207$ (60–80).
4. The clock-glitch file was reported as an unmasked defect; it is already
   dropped under `snap-test-flip`. Corrected in Addendum 0.
5. The recommendation to put the thermistor cross-calibration *ahead of* the
   bench characterisation is **withdrawn** — see Addendum 2, Checks A and B.
