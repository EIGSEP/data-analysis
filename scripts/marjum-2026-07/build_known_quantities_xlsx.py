#!/usr/bin/env python3
"""Build the portable Marjum 2026-07 known-quantities workbook.

The workbook is a presentation/export layer.  Its source of truth remains the
versioned JSON/JSONL/NPZ records named on the Sources sheet and in manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


CAMPAIGN = "marjum-2026-07"
RELEASE = "v0004"
GPS = {"east_highline_anchor": (39.24672, -113.40102),
       "west_highline_anchor": (39.24904, -113.40486),
       "pulley_plate_tiedown": (39.24786463007058, -113.40269273002308)}
LOCAL_TZ = ZoneInfo("America/Denver")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iso_local(value: str | None) -> str | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(LOCAL_TZ).isoformat(timespec="seconds")


def phase_at(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if value < "2026-07-14T04:10:43Z":
        return "A", "0/2 early (unconfirmed), then 3 (+4); no valid 3x4 cross"
    if value < "2026-07-15T00:32:17Z":
        return "B", "3 and 4; 4->5 mux when active; cross 35 valid only then"
    return "C", "0 box-gnd and 4 box-air; cross 04; mux copies are duplicates"


def flatten_pixel(value) -> tuple[float | None, float | None]:
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    return None, None


def add_table_sheet(wb: Workbook, name: str, headers: list[str], rows: list[list],
                    table_name: str, widths: dict[str, float] | None = None):
    ws = wb.create_sheet(name)
    ws.append(headers)
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 30
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    if rows:
        tab = Table(displayName=table_name, ref=ws.dimensions)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False,
            showLastColumn=False, showRowStripes=True, showColumnStripes=False
        )
        ws.add_table(tab)
    for i, header in enumerate(headers, 1):
        width = (widths or {}).get(header, min(max(len(header) + 2, 11), 24))
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    return ws


def source_catalog(root: Path) -> list[dict]:
    definitions = [
        ("SRC-GEOM-MANIFEST", f"marjum-2026-07/imgs/fits/{RELEASE}_marjum_geometry/manifest.json", "geometry release manifest", "current release; authoritative description"),
        ("SRC-CAMERAS", f"marjum-2026-07/imgs/fits/{RELEASE}_marjum_geometry/cameras.jsonl", "per-image camera solutions", "current release"),
        ("SRC-LABELS", f"marjum-2026-07/imgs/fits/{RELEASE}_marjum_geometry/labels.json", "image pixel labels", "current release"),
        ("SRC-SHARED", f"marjum-2026-07/imgs/fits/{RELEASE}_marjum_geometry/shared.json", "shared antenna/transmitter geometry", "current release"),
        ("SRC-GEOM-README", f"marjum-2026-07/imgs/fits/{RELEASE}_marjum_geometry/README.md", "geometry interpretation and caveats", "current release"),
        ("SRC-DEM", "marjum-2026-07/derived/dem/v0002/manifest.json", "UTM-georeferenced DEM the release is on", "published product"),
        ("SRC-GEOM-MEMO", "memos/memo-002-marjum-2026-07-geometry/memo-002.pdf", "anchors, positions, azimuth zero", "living memo (replaces legacy MEMO-012)"),
        ("SRC-POINTING-MEMO", "memos/memo-003-marjum-2026-07-pointing/memo-003.pdf", "mount, angle conventions, pointing fusion, azimuth zero", "living memo (replaces legacy MEMO-012/013)"),
        ("SRC-BEAM-MEMO", "memos/memo-005-marjum-2026-07-beam/memo-005.pdf", "beam reconstruction, polarization convention and handedness check", "living memo (replaces AZIMUTH_CONVENTIONS.md)"),
        ("SRC-EVENTS", "marjum-2026-07/events.jsonl", "curated campaign timeline", "59 curated records"),
        ("SRC-BOUNDARIES", "marjum-2026-07/boundaries.jsonl", "exact file-close transitions", "1,238 scan-derived boundaries"),
        ("SRC-CAMPAIGN", "marjum-2026-07/CAMPAIGN.md", "field-note transcription and campaign narrative", "secondary transcription with fieldnotes page tags"),
        ("SRC-DATA-README", "marjum-2026-07/data/README.md", "phase, live-input, outage, and corrupt-file rules", "authoritative phase table"),
        ("SRC-MODE-TABLE", "marjum-2026-07/curation/mode_table.jsonl", "per-file observing modes", "curated table"),
        ("SRC-BOXAIR-EMI-STATE", "marjum-2026-07/curation/boxair_emi_matched.jsonl", "box-air 1 MHz self-EMI state, matched detector (was tx_state_matched.jsonl, misread as transmitter state)", "derived classification"),
        ("SRC-TX-TRANSITIONS", "marjum-2026-07/curation/transmitter_transitions_boxgnd.jsonl", "transmitter on/off episodes from box-gnd, per integration", "derived classification"),
        ("SRC-HEIGHTS", "marjum-2026-07/curation/height_references.json", "height-datum reconciliation", "curated product"),
        ("SRC-MEMO-001", "memos/MEMO-001-campaign-and-program-baseline.md", "campaign baseline and chronology", "memo"),
        ("SRC-MEMO-002", "memos/MEMO-002-pointing-table-v0-beam-scan.md", "beam-scan timing and pointing", "memo"),
        ("SRC-MEMO-007", "memos/MEMO-007-comb-differencing-and-the-tx-comb-axis.md", "transmitter/comb timing interpretation", "memo"),
        ("SRC-MEMO-008", "memos/MEMO-008-component-events-and-an-undocumented-beam-scan-discontinuity.md", "component events and scan discontinuity", "memo"),
        ("SRC-GEOM-POSTERIOR", "marjum-2026-07/derived/geometry_posterior/v0004/manifest.json", "joint geometry fit the release is built from", "provisional point fit; chains not converged"),
        ("SRC-GEOM-CONVERGENCE", "marjum-2026-07/derived/geometry_posterior/v0004/run/convergence_report.json", "v0004 MCMC convergence diagnostics", "NOT CONVERGED"),
        ("SRC-GEOM-V0001", "marjum-2026-07/imgs/fits/v0001_marjum_geometry/shared.json", "historical release positions and hard brackets", "historical; superseded by v0004"),
    ]
    rows = []
    for source_id, rel, role, status in definitions:
        path = root / rel
        rows.append({
            "source_id": source_id,
            "path": rel,
            "format": path.suffix.lower().lstrip(".") or "directory",
            "role": role,
            "status": status,
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        })
    return rows


def build(root: Path, output: Path, force: bool = False) -> dict:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to replace published artifact: {output}")

    campaign = root / CAMPAIGN
    release_dir = campaign / f"imgs/fits/{RELEASE}_marjum_geometry"
    cameras = read_jsonl(release_dir / "cameras.jsonl")
    manifest = read_json(release_dir / "manifest.json")
    shared = read_json(release_dir / "shared.json")
    events = sorted(read_jsonl(campaign / "events.jsonl"), key=lambda r: r["t_start_utc"])
    from eigsep_terrain.dem import DEM
    dem = DEM(cache_file=str(campaign / "derived/dem/v0002/marjum_dem.npz"))
    if list(shared["frame"]["raster_origin_m"]) != [float(v) for v in dem.raster_origin]:
        raise ValueError("release frame and DEM v0002 raster origin differ")

    def geodetic(enu):
        if not enu:
            return None, None, None
        lat, lon, alt = dem.enu_to_latlon(enu)
        return float(lat), float(lon), float(alt)

    sources = source_catalog(root)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    commit = git_value(root, "rev-parse", "HEAD")
    branch = git_value(root, "branch", "--show-current")

    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    readme_rows = [
        ("Workbook", "Marjum 2026-07 known quantities"),
        ("Release", RELEASE),
        ("Generated UTC", generated),
        ("Repository commit", commit),
        ("Repository branch", branch),
        ("Role", "Portable generated view; source JSON/JSONL/NPZ and memos remain authoritative."),
        ("Geometry release", f"{manifest['release_id']} ({manifest['status']})"),
        ("Geometry coverage", f"{manifest['coverage']['labeled_images']} images: {manifest['coverage']['usable_camera_poses']} usable poses, {manifest['coverage']['excluded_camera_poses']} excluded/null poses"),
        ("Coordinate frame", "UTM raster grid of derived/dem/v0002: EPSG:6341 easting/northing minus the release's raster_origin_m; axes East/North/Up in metres."),
        ("Latitude/longitude", "Derived with eigsep_terrain DEM.enu_to_latlon (UTM frame) using derived/dem/v0002/marjum_dem.npz."),
        ("Altitude", "Converted geodetic altitude from enu_to_latlon; vertical datum is not independently verified. Do not silently substitute U for altitude."),
        ("Angle convention", "theta_ccw_from_east_deg is mathematical angle counter-clockwise about Up from East. Axis values are modulo 180 degrees; directed headings are modulo 360 degrees."),
        ("Camera rotation", "body_to_ENU = Rz(phi) @ Ry(theta) @ Rz(tilt), per geometry release manifest."),
        ("Time convention", "UTC is authoritative. Local display is America/Denver (MDT, UTC-06 during campaign). Filenames are file close-times."),
        ("Uncertainty", "None established. Release v0004 is a point fit selected from non-converged chains (memo 002); bound cells are blank. The v0001 brackets (1.7 m, 1.5 m) are historical."),
        ("MCMC warning", "The v0004 joint chains did not converge; no posterior interval exists."),
        ("Personal log", "The bound field notebook is not archived here. Timeline citations fieldnotes:p176-p181 are transcriptions/locators, not embedded primary pages."),
        ("Status vocabulary", "recommended = current use; candidate = unaccepted release/state; conditional = convention-dependent; superseded = retained historically; unresolved = insufficient evidence."),
        ("Blank cells", "Unknown or inapplicable. In particular, integration indices remain blank where evidence resolves only to files or approximate field-note times."),
    ]
    for row in readme_rows:
        ws.append(row)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 110
    for cell in ws["A"]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for cell in ws["B"]:
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"

    image_headers = [
        "image", "pose_status", "exclusion_reason", "e_m", "n_m", "u_m",
        "lat_deg", "lon_deg", "alt_m_unverified", "theta_rad", "phi_rad",
        "tilt_rad", "theta_deg", "phi_deg", "tilt_deg", "focal_length_px",
        "image_width_px", "image_height_px", "lens_group", "k1", "k2",
        "antenna_label_x_px", "antenna_label_y_px", "transmitter_label_x_px",
        "transmitter_label_y_px", "transmitter_conditioned", "fit_provenance",
        "geometry_release", "image_sha256", "source_id"
    ]
    image_rows = []
    for record in cameras:
        camera = record.get("camera") or {}
        enu = camera.get("position_enu_m")
        lat, lon, alt = geodetic(enu)
        ori = camera.get("orientation_rad") or {}
        shape = camera.get("image_shape_px") or {}
        distortion = camera.get("radial_distortion") or {}
        labels = record.get("labels") or {}
        ant_x, ant_y = flatten_pixel(labels.get("antenna_px") or labels.get("ant_px"))
        tx_x, tx_y = flatten_pixel(labels.get("transmitter_px"))
        vals = [ori.get(k) for k in ("theta", "phi", "tilt")]
        image_rows.append([
            record.get("image"), record.get("pose_status"),
            record.get("exclusion_reason") or record.get("reason"),
            *(enu or [None, None, None]), lat, lon, alt, *vals,
            *[math.degrees(v) if v is not None else None for v in vals],
            camera.get("focal_length_px"), shape.get("width"), shape.get("height"),
            camera.get("lens_group"), distortion.get("k1"), distortion.get("k2"),
            ant_x, ant_y, tx_x, tx_y, record.get("transmitter_conditioned"),
            record.get("fit_provenance"), record.get("release"),
            record.get("image_sha256"), "SRC-CAMERAS",
        ])
    add_table_sheet(
        wb, "Image Geometry", image_headers, image_rows, "ImageGeometry",
        {"image": 20, "exclusion_reason": 30, "fit_provenance": 32,
         "image_sha256": 66, "source_id": 18}
    )

    system_headers = [
        "entity", "record_type", "status", "e_m", "n_m", "u_m", "lat_deg",
        "lon_deg", "alt_m_unverified", "bound_value", "bound_unit", "bound_kind",
        "theta_ccw_from_east_deg", "angle_domain", "angular_uncertainty_deg",
        "compass_bearing_deg", "meaning", "source_id", "source_locator", "notes"
    ]
    system_rows = []

    def point_row(entity, enu, bound, bound_kind, status, source_id, locator, notes=""):
        lat, lon, alt = geodetic(enu)
        return [entity, "point", status, *enu, lat, lon, alt, bound, "m",
                bound_kind, None, None, None, None, "position", source_id, locator, notes]

    ant = shared["antenna_91m_era"]["position_enu_m"]
    tx = shared["transmitter"]["recommended_for_propagation"]["position_enu_m"]
    system_rows.append(point_row("antenna_91m_era", ant, None,
        "no uncertainty established", shared["antenna_91m_era"]["status"],
        "SRC-SHARED", "antenna_91m_era.position_enu_m", "91 m-era position, geometry v0004"))
    system_rows.append(point_row("transmitter", tx, None,
        "no uncertainty established", shared["transmitter"]["recommended_for_propagation"]["status"],
        "SRC-SHARED", "transmitter.recommended_for_propagation.position_enu_m", "geometry v0004"))
    for name, (lat, lon) in GPS.items():
        # U is the DEM elevation at the GPS horizontal position, not a geodetic altitude.
        e, n = (float(v) for v in np.ravel(dem.latlon_to_enu(lat, lon))[:2])
        system_rows.append(point_row(name, [e, n, float(dem.interp_alt(e, n))], None,
            "receiver accuracy not recorded", "measured GPS horizontal; DEM-derived U",
            "SRC-GEOM-MEMO", "azimuth-zero section", f"GPS receiver -> CalTopo; lat/lon {lat:.5f}, {lon:.5f}"))

    def axis_row(entity, theta, domain, sigma, compass, meaning, status, source, locator, notes=""):
        return [entity, "axis", status, None, None, None, None, None, None,
                None, None, None, theta, domain, sigma, compass, meaning,
                source, locator, notes]

    system_rows.append(axis_row("highline_east_to_west", 142.164, "directed modulo 360", 0.6,
        307.836, "GPS anchor-to-anchor highline direction, east anchor toward west anchor",
        "settled", "SRC-GEOM-MEMO", "sections 4.8 and 4.11"))
    system_rows.append(axis_row("antenna_arm_at_az_pot_zero", 142.164, "axis modulo 180", 0.6,
        127.836, "Dipole arm/highline axis at az_pot=0; opposite compass direction is equivalent",
        "settled from GPS anchors plus hardware statement", "SRC-GEOM-MEMO", "section 4.11"))
    system_rows.append(axis_row("lidar_or_boresight_plane_at_az_pot_zero", 52.164,
        "directed modulo 360", 0.6, 37.836,
        "daz; perpendicular to antenna arm/highline at az_pot=0",
        "settled", "SRC-GEOM-MEMO", "sections 1, 4.8, and 4.11"))
    system_rows.append(axis_row("transmitter_polarization_model_alpha", None,
        "model parameter; sky-axis conversion unresolved", None, None,
        "Beam-model alpha near 131 deg exists, but its ownership/frame and handedness were not settled; no physical theta is asserted here",
        "conditional / unresolved", "SRC-AZ-CONVENTIONS", "sections 1-4",
        "Do not substitute alpha=131 deg for theta_ccw_from_east without resolving field_top semantics and azimuth handedness."))
    add_table_sheet(
        wb, "System Geometry", system_headers, system_rows, "SystemGeometry",
        {"entity": 34, "status": 30, "bound_kind": 32, "meaning": 58,
         "source_locator": 30, "notes": 62}
    )

    timeline_headers = [
        "event_id", "t_start_utc", "t_end_utc", "t_start_local_mdt",
        "t_end_local_mdt", "category", "phase", "live_inputs", "affects",
        "first_boundary_file", "last_boundary_file", "first_integration",
        "last_integration", "boundary_signals", "source", "evidence_precision",
        "notes", "data_use_warning"
    ]
    timeline_rows = []
    for index, event in enumerate(events, 1):
        phase, inputs = phase_at(event.get("t_start_utc"))
        boundaries = event.get("matched_boundaries") or []
        files = [b.get("file") for b in boundaries if b.get("file")]
        signals = []
        for boundary in boundaries:
            for signal in boundary.get("signals") or []:
                if signal not in signals:
                    signals.append(signal)
        source = event.get("source") or ""
        precision = "exact/scan-derived file boundary" if source.startswith("scan:") or source == "derived" else "field-note time; often approximate"
        if source in {"readme", "csv+readme"}:
            precision = "curated documented interval"
        warning = ""
        if event.get("category") in {"config-change", "data-gap", "outage", "data-anomaly"}:
            warning = "Inspect/mask affected files before analysis; see event notes and data/README.md."
        timeline_rows.append([
            f"EVT-{index:03d}", event.get("t_start_utc"), event.get("t_end_utc"),
            iso_local(event.get("t_start_utc")), iso_local(event.get("t_end_utc")),
            event.get("category"), phase, inputs, event.get("affects"),
            files[0] if files else None, files[-1] if files else None,
            None, None, "; ".join(signals), source, precision,
            event.get("notes"), warning,
        ])
    add_table_sheet(
        wb, "Timeline", timeline_headers, timeline_rows, "CampaignTimeline",
        {"event_id": 12, "t_start_utc": 22, "t_end_utc": 22,
         "t_start_local_mdt": 27, "t_end_local_mdt": 27, "live_inputs": 54,
         "affects": 44, "first_boundary_file": 30, "last_boundary_file": 30,
         "boundary_signals": 70, "source": 22, "evidence_precision": 30,
         "notes": 90, "data_use_warning": 54}
    )

    source_headers = ["source_id", "path", "format", "role", "status", "present", "bytes", "sha256"]
    source_rows = [[r[h] for h in source_headers] for r in sources]
    add_table_sheet(
        wb, "Sources", source_headers, source_rows, "SourceCatalog",
        {"source_id": 24, "path": 92, "format": 12, "role": 44,
         "status": 48, "present": 11, "bytes": 14, "sha256": 66}
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)

    # Structural validation after serialization catches illegal sheet/table names
    # and confirms the important row-accounting invariants.
    check = load_workbook(output, read_only=False, data_only=False)
    assert check.sheetnames == ["README", "Image Geometry", "System Geometry", "Timeline", "Sources"]
    assert check["Image Geometry"].max_row == 1 + manifest["coverage"]["labeled_images"]
    assert check["Timeline"].max_row == 1 + len(events)
    assert sum(1 for r in cameras if r.get("pose_status") == "usable") == manifest["coverage"]["usable_camera_poses"]
    assert sum(1 for r in cameras if r.get("pose_status") != "usable") == manifest["coverage"]["excluded_camera_poses"]
    check.close()

    output_manifest = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "product": "known_quantities_workbook",
        "release": RELEASE,
        "status": "candidate",
        "generated_utc": generated,
        "generator": "data-analysis/scripts/marjum-2026-07/build_known_quantities_xlsx.py",
        "repository": {"commit": commit, "branch": branch, "dirty": bool(git_value(root, "status", "--porcelain"))},
        "workbook": {"path": output.relative_to(root).as_posix(), "sha256": sha256(output), "bytes": output.stat().st_size},
        "coverage": {
            "image_rows": len(cameras),
            "usable_image_poses": sum(r.get("pose_status") == "usable" for r in cameras),
            "excluded_image_poses": sum(r.get("pose_status") != "usable" for r in cameras),
            "system_geometry_rows": len(system_rows),
            "timeline_rows": len(events),
            "source_rows": len(sources),
        },
        "sources": sources,
        "known_limitations": [
            "Geometry release v0004 is a provisional point fit with no established uncertainty.",
            "Converted altitude uses a vertical datum that has not been independently verified.",
            "The original bound field notebook is absent; fieldnotes page tags point to transcriptions.",
            "Integration indices are not filled where the curated evidence resolves only to file boundaries.",
            "The v0004 joint MCMC chains are nonconverged; positions are a point selection.",
            "No physical transmitter-polarization theta is asserted until alpha/frame/handedness semantics are resolved.",
        ],
    }
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    readme_path = output.with_name("README.md")
    readme_path.write_text(
        f"# {CAMPAIGN} known quantities — {RELEASE}\n\n"
        "Portable XLSX export of current campaign geometry, key events, and provenance. "
        "It is generated from the source artifacts listed in the workbook and `manifest.json`; "
        "do not hand-edit it as a source of truth.\n\n"
        f"- Workbook: `{output.name}`\n"
        "- Generator: `data-analysis/scripts/marjum-2026-07/build_known_quantities_xlsx.py`\n"
        f"- Generated: `{generated}`\n"
        "- Status: **candidate**\n\n"
        "Important caveats are reproduced on the workbook's README sheet. In particular, "
        "geometry release v0004 has no established uncertainty, its MCMC chains are "
        "nonconverged, and the physical transmitter-polarization sky angle remains unresolved.\n",
        encoding="utf-8",
    )
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing draft artifact")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3],
                        help="workspace holding the campaign (default: beside data-analysis)")
    parser.add_argument("--output", type=Path, default=None, help="workbook path (default: the campaign product)")
    args = parser.parse_args()
    root = args.root
    output = args.output or root / CAMPAIGN / "derived/known_quantities" / RELEASE / f"{CAMPAIGN}_known_quantities_{RELEASE}.xlsx"
    result = build(root, output, force=args.force)
    print(json.dumps({"workbook": result["workbook"], "coverage": result["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
