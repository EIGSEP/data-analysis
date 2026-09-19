"""
Merge detected boundaries into events.jsonl.

Two operations:
  (1) Add scan-derived events for boundaries that are NOT within ±30 min of a
      field-notes event. These are precise "we saw something happen in the data
      that the notebook does not mention" markers.

        - SNAP test-flip pairs -> category='config-change', source='scan:snap-test-flip'
        - corr_acc_len change  -> category='config-change', source='scan:corr-acc-len'
        - run-boundary gaps >5min NOT within an existing outage/data-* event
                              -> category='data-gap',     source='scan:gap'
        - adc-mux clusters (Phase-C mux tuning after 04:10Z on 07-14) grouped
                              -> category='config-change', source='scan:mux-tuning'

  (2) Annotate every existing event (and new event) with 'matched_boundaries'
      listing detected boundaries within ±30 min, giving each event a
      file-precise attachment point.

Writes events.jsonl.new (chronologically sorted) alongside the original.
"""
import os
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

from eigsep_data.paths import get_campaign_root


def _campaign_root():
    """Campaign root; ``MARJUM_DATA_ROOT`` wins, else the package setting.

    This script anchored on its own ``__file__`` until it moved out of
    the campaign tree on 2026-09-19.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return Path(env)
    return get_campaign_root(required=True)


ROOT = str(_campaign_root())

with open(f"{ROOT}/events.jsonl") as f:
    evs = [json.loads(l) for l in f]
with open(f"{ROOT}/boundaries.jsonl") as f:
    bs = [json.loads(l) for l in f]


def parse(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def key_signals(sigs):
    """Concise summary of the signals on a boundary."""
    keep = (
        "snap_id", "ants", "adc_mux_sel", "adc_gain", "corr_acc_len",
        "wiring_hash", "obs_config_hash", "cal_cadence_active",
        "has_meta_motor", "has_meta_rfswitch", "has_meta_lidar",
        "has_meta_imu_el", "has_meta_potmon", "has_meta_tempctrl_lna",
        "has_meta_system_current",
        "gap_s", "filter_phase", "data_keys",
    )
    out = []
    for s in sigs:
        if s["signal"] in keep:
            b = s.get("before")
            a = s.get("after")
            if s["signal"] == "gap_s" and a is not None:
                out.append(f"gap={float(a)/60:.1f}min")
            elif s["signal"].startswith("has_meta_"):
                out.append(f"{s['signal']}:{b}->{a}")
            else:
                bs_ = str(b)[:40]
                as_ = str(a)[:40]
                out.append(f"{s['signal']}:{bs_}->{as_}")
    return out


# ============================================================
# STEP 1a: SNAP test-flip pairs (8 pairs)
# ============================================================
wiring = sorted([b for b in bs if b["category"] == "wiring"], key=lambda x: x["t_utc"])
snap_pairs = []
used = set()
for i, b in enumerate(wiring):
    if i in used:
        continue
    snap = next((s for s in b["signals"] if s["signal"] == "snap_id"), None)
    if not snap or snap.get("before") != "C000122" or snap.get("after") != "C000069":
        continue
    for j in range(i + 1, min(i + 3, len(wiring))):
        if j in used:
            continue
        b2 = wiring[j]
        snap2 = next((s for s in b2["signals"] if s["signal"] == "snap_id"), None)
        if snap2 and snap2.get("before") == "C000069" and snap2.get("after") == "C000122":
            if parse(b2["t_utc"]).timestamp() - parse(b["t_utc"]).timestamp() < 600:
                snap_pairs.append((b, b2))
                used.add(i)
                used.add(j)
                break

# The wiring changes NOT part of any test flip pair are the real
# phase A->B / B->C transitions. Ferret those out for reference.
real_wiring = [wiring[i] for i in range(len(wiring)) if i not in used]

# ============================================================
# STEP 1b: corr_acc_len change and adc-mux tuning window
# ============================================================
corr_config = [b for b in bs if b["category"] == "correlator-config"]
adc_mux = sorted([b for b in bs if b["category"] == "adc-mux"], key=lambda x: x["t_utc"])

# Group adc-mux boundaries into 'clusters' with <30 min gap between consecutive events
adc_clusters = []
cluster = []
last_t = None
for b in adc_mux:
    t = parse(b["t_utc"]).timestamp()
    if last_t is None or (t - last_t) < 1800:
        cluster.append(b)
    else:
        adc_clusters.append(cluster)
        cluster = [b]
    last_t = t
if cluster:
    adc_clusters.append(cluster)

# ============================================================
# STEP 1c: significant gaps not explained by existing events
# ============================================================
gap_bounds = []
for b in bs:
    if b["category"] != "run-boundary":
        continue
    for s in b["signals"]:
        if s["signal"] == "gap_s" and s.get("after"):
            gap_s = float(s["after"])
            if gap_s > 300:  # >5 min
                gap_bounds.append((b, gap_s))
            break

# ============================================================
# Attach detected boundaries to each existing event
# ============================================================
def find_matches(e_start_iso, e_end_iso, tol_before=1800, tol_after=1800):
    ts = parse(e_start_iso).timestamp() - tol_before
    te = parse(e_end_iso or e_start_iso).timestamp() + tol_after
    matches = []
    for b in bs:
        # Skip trivial signal-only boundaries — they'd swamp everything
        if b["category"] == "signal":
            continue
        # Skip cal-cadence blinks that come from scheduler
        if b["category"] == "cal-cadence":
            continue
        bt = parse(b["t_utc"]).timestamp()
        if ts <= bt <= te:
            matches.append({
                "t_utc": b["t_utc"],
                "file": b["file"],
                "category": b["category"],
                "signals": key_signals(b["signals"]),
            })
    return matches

def bounds_taken_by(existing_events):
    """Return set of (t_utc, category, file) covered by existing events."""
    taken = set()
    for e in existing_events:
        for m in e.get("matched_boundaries", []):
            taken.add((m["t_utc"], m["category"], m["file"]))
    return taken

# First attach matches to existing events
for e in evs:
    e["matched_boundaries"] = find_matches(e["t_start_utc"], e.get("t_end_utc"))

taken = bounds_taken_by(evs)

# ============================================================
# STEP 2: Add scan-derived events for the unmatched ones
# ============================================================
new_events = []

# 2a: SNAP test-flip pairs
for a, b in snap_pairs:
    if (a["t_utc"], a["category"], a["file"]) in taken:
        continue
    ants_sig = next((s for s in a["signals"] if s["signal"] == "ants"), None)
    ants_before = ants_sig.get("before") if ants_sig else "?"
    ants_after = ants_sig.get("after") if ants_sig else "?"
    new_events.append({
        "t_start_utc": a["t_utc"],
        "t_end_utc": b["t_utc"],
        "category": "config-change",
        "source": "scan:snap-test-flip",
        "affects": f"{a['file']}..{b['file']}",
        "notes": (
            f"Brief (~{int(parse(b['t_utc']).timestamp()-parse(a['t_utc']).timestamp())}s) SNAP swap: "
            f"flight config (C000122, {ants_before}) → scratch (C000069, {ants_after}) → back. "
            f"Probably check_snap or config verification on the second SNAP board. "
            f"Bracketing files are NOT flight-configured; treat as instrumental."
        ),
        "matched_boundaries": [
            {"t_utc": a["t_utc"], "file": a["file"], "category": a["category"], "signals": key_signals(a["signals"])},
            {"t_utc": b["t_utc"], "file": b["file"], "category": b["category"], "signals": key_signals(b["signals"])},
        ],
    })

# 2b: corr_acc_len change (correlator-config)
for b in corr_config:
    if (b["t_utc"], b["category"], b["file"]) in taken:
        continue
    accum = next((s for s in b["signals"] if s["signal"] == "corr_acc_len"), None)
    if not accum:
        continue
    new_events.append({
        "t_start_utc": b["t_utc"],
        "t_end_utc": None,
        "category": "config-change",
        "source": "scan:corr-acc-len",
        "affects": b["file"],
        "notes": (
            f"Correlator accumulator length changed: "
            f"{accum.get('before')} -> {accum.get('after')}. "
            f"Affects integration time; downstream calibration must handle the transition."
        ),
        "matched_boundaries": [
            {"t_utc": b["t_utc"], "file": b["file"], "category": b["category"], "signals": key_signals(b["signals"])},
        ],
    })

# 2c: adc-mux tuning cluster (only large clusters likely to represent a distinct operational window)
for cl in adc_clusters:
    # Skip if any boundary in cluster is already covered by an event
    covered = any((b["t_utc"], b["category"], b["file"]) in taken for b in cl)
    if covered:
        continue
    if len(cl) < 3:  # noise
        continue
    tstart = cl[0]["t_utc"]
    tend = cl[-1]["t_utc"]
    new_events.append({
        "t_start_utc": tstart,
        "t_end_utc": tend,
        "category": "config-change",
        "source": "scan:mux-tuning",
        "affects": f"{cl[0]['file']}..{cl[-1]['file']}",
        "notes": (
            f"ADC-mux tuning window: {len(cl)} rapid changes to mux_0to1/mux_4to5 "
            f"({(parse(tend).timestamp()-parse(tstart).timestamp())/60:.0f} min). "
            f"Phase-C mux-copy configuration being iterated."
        ),
        "matched_boundaries": [
            {"t_utc": b["t_utc"], "file": b["file"], "category": b["category"], "signals": key_signals(b["signals"])}
            for b in cl[:6]  # cap
        ],
    })

# 2d: significant gaps not already inside an existing event window
# Existing outage/data-gap events already have gap boundaries in their matched list.
# We only emit a new gap event if none of the outage/data-gap events overlap.
existing_outage_windows = []
for e in evs:
    if e["category"] in ("outage", "data-gap", "data-start", "data-end"):
        ts = parse(e["t_start_utc"]).timestamp() - 900
        te = parse(e.get("t_end_utc") or e["t_start_utc"]).timestamp() + 900
        existing_outage_windows.append((ts, te))

for b, gap_s in gap_bounds:
    bt = parse(b["t_utc"]).timestamp()
    if any(ts <= bt <= te for ts, te in existing_outage_windows):
        continue
    if (b["t_utc"], b["category"], b["file"]) in taken:
        continue
    new_events.append({
        "t_start_utc": iso(bt - gap_s),
        "t_end_utc": b["t_utc"],
        "category": "data-gap",
        "source": "scan:gap",
        "affects": b["file"],
        "notes": f"Recording gap of {gap_s/60:.1f} min between files. "
                 f"Not documented in field notes or data/README.",
        "matched_boundaries": [
            {"t_utc": b["t_utc"], "file": b["file"], "category": b["category"], "signals": key_signals(b["signals"])},
        ],
    })

# ============================================================
# Merge, sort, write
# ============================================================
all_events = evs + new_events
all_events.sort(key=lambda e: (e["t_start_utc"], e.get("category") or ""))

# Reorder each event dict for readability
ORDER = ("t_start_utc", "t_end_utc", "category", "source", "affects", "notes", "matched_boundaries")
def reorder(e):
    return {k: e[k] for k in ORDER if k in e}

with open(f"{ROOT}/events.jsonl.new", "w") as f:
    for e in all_events:
        f.write(json.dumps(reorder(e), ensure_ascii=False) + "\n")

print(f"Wrote {len(all_events)} events ({len(evs)} original + {len(new_events)} scan-derived)")
print(f"  SNAP-test flip pairs: {sum(1 for e in new_events if e['source']=='scan:snap-test-flip')}")
print(f"  corr-acc-len change:  {sum(1 for e in new_events if e['source']=='scan:corr-acc-len')}")
print(f"  Mux tuning windows:   {sum(1 for e in new_events if e['source']=='scan:mux-tuning')}")
print(f"  Undocumented gaps:    {sum(1 for e in new_events if e['source']=='scan:gap')}")
print()
print("Real (non-test) wiring boundaries that align with phase model:")
for b in real_wiring:
    print(f"  {b['t_utc']}  {b['file']}")
