# Azimuth conventions in the beam-fit path — inconsistency write-up

beam-analyst, 2026-09-19. **Discussion document. Nothing implemented.**

Written for the joint discussion with geometer and Aaron on item 5 (the
polarization co-rotation question). Conventions are stated first as explicit
transformations, then the code is checked against them — not the reverse.
Cross-references MEMO-012 §4.2, §4.9, §4.10, §4.11 rather than restating them.

---

## 1. Conventions, stated as transformations

Each frame is given as a matrix and an explicit zero. "Body frame" and "top
frame" as bare names are where the confusion lived; they are not used here
without a definition.

| symbol | definition |
|---|---|
| **ENU** | topocentric, `x = East`, `y = North`, `z = Up`. Bearing of a vector `v` is `atan2(v_E, v_N)`. |
| `az` | the scalar in `pointing_table.az_deg`, body azimuth. **Its zero is not north** — no north anchor exists (MEMO-012 §4.11). |
| `el` | boresight zenith angle: `0` = zenith, `+90` = horizon, `+180` = nadir (geometer, Q8, 2026-09-16). |
| **R(az, el)** | the receiver rotation. `R = R_el(el) · R_az(az)`; `Rᵀ` maps an ENU vector into the receiver frame. |
| `az_axis` | the directed axis `R_az` turns about. `(0,0,+1)` = right-handed about **Up**; `(0,0,−1)` = right-handed about a **downward** shaft. |
| `alpha` | polarization angle. `field_top(a) = (−sin a, cos a, 0)` with `a = alpha + 90·arm`, expressed in the frame `R` maps *from* — i.e. ENU at `az = el = 0`. |

**The zero that matters:** `R(0, 0) = I`, so the frame `alpha` lives in is the
receiver frame *at `az_pot = 0`*, which the model identifies with ENU. That
identification is an assumption, not a measurement, and it is the same
assumption the `heading` vector relies on — `heading_from_enu(dE, dN, dU)` is a
true-ENU vector fed into the same `Rᵀ`. So the two stand or fall together: if
the `az = 0` frame is not ENU, the transmitter heading is misinterpreted too,
which is a far larger problem than any polarization question.

---

## 2. The live inconsistency: two opposite azimuth handednesses

| site | `az_axis` |
|---|---|
| `geometry.py:50` `rotation_matrix` (generic default) | `(0, 0, +1)` |
| `beam_sim.py:309` `__init__` default | `(0, 0, +1)` |
| `tx_model.py:72`, `:117` (**the production path**) | `(0, 0, −1)` |
| `explorer_model.rotations()` (my copy) | hardcoded, `sa = −sin(az)` |

Measured, not read (az = 37°, el = 0):

```
rotation_matrix(az_axis=+Z)      [[ 0.798636 -0.601815 0] [ 0.601815 0.798636 0] [0 0 1]]
rotation_matrix(az_axis=-Z)      [[ 0.798636  0.601815 0] [-0.601815 0.798636 0] [0 0 1]]
tx_model fast path               [[ 0.798636  0.601815 0] [-0.601815 0.798636 0] [0 0 1]]

fast path == -Z generic : True
fast path == +Z generic : False
+Z == transpose(-Z)     : True
```

So the two are exact inverses — a pure sign flip on `az` — and my
`explorer_model.rotations()` is the `−Z` case, identical to the production fast
path. My model is internally consistent with `tx_model`; it is `geometry.py`
and `beam_sim.py` that disagree with both.

**Why this is a documented-intent-vs-implementation mismatch, not a physics
disagreement.** `geometry.py:54` states the rule itself: `(0,0,−1)` is for *"an
encoder reporting a right-handed angle about a **downward** shaft."* If the pot
sense is right-handed about **+U**, as established 2026-09-18, then the
production path's `−Z` contradicts the condition its own docstring gives for
using it. Either something upstream (a sign already applied when `pot_az` became
`az_deg`) compensates, or it is a sign bug. **I cannot tell from the code, and
the fit cannot tell me either — see §3.**

---

## 3. Why no amount of beam data will settle it

ch712, surveyed heading, pure HFSS, `alpha` refit in each case:

| azimuth handedness | best `alpha` | normalized RMS |
|---|---|---|
| as shipped (`−Z`) | 41.0° | 0.3832 |
| sign-flipped (`+Z`) | 139.0° | 0.3791 |

`139 = 180 − 41`. Flipping the handedness costs **0.0040** in normalized RMS and
is absorbed almost exactly by `alpha → −alpha (mod 180)`. The residual 0.0040 is
the bowtie's departure from perfect mirror symmetry, not a discriminant.

**Consequence, and this is the load-bearing point for the joint discussion:**
the unresolved `±Z` handedness *is* the unresolved sign in the polarization
angle. They are one degeneracy, not two. Concretely, it is exactly the
reflection that broke the recent cross-frame comparison:

- my ch712 fit gives field angle **131.0°**;
- under the shipped convention that vector's ENU bearing is
  `atan2(−sin a, cos a) = −a`, i.e. axis **49.0°**;
- the model's `alpha` for an arm along the highline (bearing 307.836°) at
  `az_pot = 0` is **52.164°**, verified numerically (`|cos| = 1.00000000`),
  **not** 131° — so the two numbers are **78.84°** apart, and the earlier
  "3.16° agreement" compared an `alpha` against a bearing.

Answering geometer's still-open question directly: **52.164°**. It was answered
on 2026-09-18 and the message appears to have crossed; restating it here so the
discussion has it in one place. Under the arm-pair ambiguity the closest any
highline-referenced arm gets to my fit is the perpendicular arm at 142.16°,
**11.16°** away — still ~5× the ±2° photogrammetric uncertainty.

---

## 4. The claim that started this, and what is actually wrong with it

`TransmitterGeometry.field_top` (`geometry.py:176`) documents itself as *"the
driven E2 dipole arm in the top frame"* — i.e. **our** antenna's arm. But
`coupling()` applies `Rᵀ` to it, so the vector is treated as fixed in the
`az = 0` frame and **does not co-rotate with the antenna**. Measured:

```
e_body at az =  0.0 : (-0.75471, -0.65606, 0)
e_body at az = 90.0 : ( 0.65606, -0.75471, 0)
```

A dipole bolted to a rotating antenna must co-rotate; a world-fixed source
polarization must not. The code does the latter while the docstring names the
former. Exactly one of these is true:

1. **It is our arm** → the azimuth rotation is misapplied, and every fit in the
   beam-scan window is affected, because azimuth sweeps across the dataset.
2. **It stands for the transmitter's polarization** → the code is right, the
   docstring is wrong, and `alpha` must never be compared against our arm's
   orientation — which voids the §4.11 comparison independently of frames.

This is consistent with MEMO-012 §4.9 (azimuth rotates polarization, does not
steer the beam) — but §4.9 does not say *whose* polarization, and that is the
whole question.

---

## 5. What would settle each item

| question | settled by | not settled by |
|---|---|---|
| `±Z` handedness | the pot wiring / encoder sense, and whether a sign is already applied in building `az_deg` | beam RMS — degenerate to 0.0040 |
| whose polarization `field_top` is | HFSS port definition: is E2 the simulated receiving antenna's arm, or a source | any fit; both choices refit `alpha` to similar RMS |
| `alpha` ↔ sky bearing | both of the above, plus the `az = 0`-frame-is-ENU assumption in §1 | — |

**Recommended order:** resolve §4 first. If `field_top` is the transmitter's
polarization, the §4.11 arm comparison is void whatever the handedness is, and
the `±Z` question shrinks to a tidy-up. If it is our arm, there is a real
azimuth bug and the handedness question must be answered before anything is
refit.

**Not proposing an implementation.** The `±Z` disagreement should not be
"fixed" by making the defaults match until §4 is answered, because the current
production behaviour is self-consistent and any change silently flips the sign
of every reported `alpha`.
