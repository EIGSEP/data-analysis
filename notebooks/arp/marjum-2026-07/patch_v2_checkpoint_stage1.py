"""Stage 1 of the 2026-09-17 in-place correction of beam_fits_v2_review_checkpoint.

Patches the code cells (import path + corrected masks) and the structural
markdown that does not depend on freshly computed numbers. Narrative cells that
quote numbers are patched in stage 2, after execution.
"""
import nbformat as nbf

NB = "beam_fits_v2_review_checkpoint.ipynb"
nb = nbf.read(NB, as_version=4)


def src(j):
    return "".join(nb.cells[j].source)


def setsrc(j, s):
    nb.cells[j].source = s


# ---- cell 2: stop shadowing the repo copy of fit_beam_v2 with /tmp ----------
s = src(2)
assert 'sys.path.insert(0, "/tmp")' in s, "path insert not found"
s = s.replace(
    'sys.path.insert(0, "/tmp")',
    '# NOTE: /tmp is deliberately NOT on the path. fit_beam_v2.py lived only in\n'
    '# /tmp until 2026-09-17 and is now version-controlled next to this\n'
    '# notebook; inserting /tmp last would shadow the corrected copy with the\n'
    '# stale one.')
setsrc(2, s)

# ---- cell 16: apply the same two corrections the pipeline now applies ------
s = src(16)
anchor = 'clean_mask_full = clean_mask_full & v2.pointing_v1_valid_mask(data_full)'
assert anchor in s, "mask anchor not found"
s = s.replace(anchor, anchor + """

# The two corrections applied to the pipeline on 2026-09-17 (Aaron's ruling on
# the D2 review gate). Reusing v2's own functions so there is one definition of
# each mask, not a copy that can drift from the report this notebook reads.
#   - receiver_on_antenna_mask: per-sample metadata/rfswitch. 23 of the 226
#     files are calibration files; their off-antenna samples were being fit as
#     if they were beam measurements.
#   - el_solution_glitch_mask: pointing_table@v1.2's EL_SOLUTION_GLITCH bit.
_on_ant = v2.receiver_on_antenna_mask(data_full)
_no_glitch = v2.el_solution_glitch_mask(data_full)
print(f"mask: {int(clean_mask_full.sum())} -> "
      f"{int((clean_mask_full & _on_ant & _no_glitch).sum())} "
      f"(-{int((clean_mask_full & ~_on_ant).sum())} off-antenna, "
      f"-{int((clean_mask_full & _on_ant & ~_no_glitch).sum())} glitch)")
clean_mask_full = clean_mask_full & _on_ant & _no_glitch""")
setsrc(16, s)

# ---- cell 19: the PENDING REVISION banner is resolved ----------------------
setsrc(19, r"""> **REVISED 2026-09-17 — the exclusion below has been removed.** Q8 is closed.
> `geometer` established that `el` is the **boresight zenith angle**: `el = 0`
> zenith, `el = +90` horizon, `el = +180` nadir — proven independently of Aaron
> by LIDAR return character against elevation (493 real ground returns at
> `el ≈ +90` with median range 92.31 m; **zero** returns at `el ≈ −90`, i.e.
> sky; 2412 out-of-range sentinels at `el ≈ 0`, shooting across the canyon).
> The `+90 / −90` asymmetry is the sign-resolving observation.
>
> Section 1a places the transmitter essentially straight down — 85.8° below
> horizontal, i.e. near nadir. Under the now-resolved convention that puts it
> **on boresight at `|el| ≈ 180`**. The "wrap cluster" this section previously
> excluded as a suspected angle-wrap artifact is therefore the **most on-source
> pointing in the dataset**, and it is now counted in the coverage claim rather
> than footnoted out of it.

## 6. Coverage map (`pointing_table@v1.2`-only, independent of geometry/beam)

Computed on the corrected sample selection (off-antenna and
`EL_SOLUTION_GLITCH` samples removed), and reported in three parts: the total,
the on-source subset near nadir, and the rest.""")

# ---- after cell 20 (the coverage print), add the interpretation ------------
nb.cells.insert(21, nbf.v4.new_markdown_cell(r"""**Reading the coverage map.** Two caveats travel with it and are carried in
the JSON itself rather than only here:

- **The elevation zero is only bounded, not measured.** `geometer`'s MAD
  profile is flat within DEM quantisation over `el_nadir ≈ 87.5–91.5`, so the
  pointing table's el-zero offset is constrained to `|offset| ≲ 2°`. The
  zenith/nadir *sense* is settled; the zero *point* is not, to ~2°.
- **Absolute azimuth zero is still open.** The pot azimuth is a body-frame
  angle, not north-referenced, so the azimuth axis here has no absolute north
  anchor. Azimuth *coverage* is meaningful; an azimuth *bearing* read off this
  map is not."""))

nbf.write(nb, NB)
print(f"patched {NB}: {len(nb.cells)} cells")
