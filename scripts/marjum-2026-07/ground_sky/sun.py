"""The Sun as seen from the Marjum antenna: position, terrain shadow, beam response.

The Sun enters the ground/sky model as one extra column per frequency,

    T_sun(t) = S_sun * lambda^2 / (8 pi k) * G(s_hat(t)) * V(t),

with ``S_sun`` the flux density in SFU (fitted), ``G`` the beam gain toward the
Sun (same full-sphere normalization as the design matrix's rows) and ``V`` the
terrain visibility. The Sun is a point source here; the radio disk (~0.5-1.5 deg)
is small next to the Fresnel scale of the ridges, below.

Terrain visibility is traced from the DEM along the Sun's own bearing at each
time, not read from a fixed horizon profile, because the edge distance matters:
the ridges that set sunrise and sunset are 150-350 m away, so at these
wavelengths the edge diffracts over sqrt(lambda d / 2) ~ 2-6 deg, i.e. tens of
minutes of solar motion. ``V`` is the knife-edge power, 0.25 at the geometric
edge, or a hard step (``model="hard"``). ``model="flat"`` ignores the
terrain (visible whenever the Sun is above 0 deg) and exists as the
alternative hypothesis for timing tests.

Bearings: the DEM is NAD83(2011) / UTM 12N (EPSG:6341), whose grid north is
``grid_to_true_deg()`` (about -1.52 deg) from true north at the site. The Sun's
true bearing is converted to the grid before the DEM is read. The same rotation
applies to ``horizon_profiles_v0002``, whose bearings are grid bearings.
"""

from __future__ import annotations

import json
from functools import lru_cache

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy.special import fresnel

from eigsep_base.const import MARJUM_PASS

from common import campaign_root

K_B = 1.380649e-23
C_LIGHT = 299792458.0
SFU = 1e-22  # W m^-2 Hz^-1
R_EARTH = 6371e3
DEM_PATH = "derived/dem/v0001/marjum_dem.npz"
HORIZON_JSON = "curation/horizon_profiles_v0002.json"


@lru_cache(maxsize=None)
def grid_to_true_deg():
    """True bearing minus grid bearing at the site, degrees (bearings CCW from east).

    A direction at grid bearing b points at true bearing b + this.
    """
    from pyproj import Transformer

    lat, lon, _ = MARJUM_PASS
    tr = Transformer.from_crs("EPSG:4326", "EPSG:6341", always_xy=True)
    e0, n0 = tr.transform(lon, lat)
    e1, n1 = tr.transform(lon, lat + 0.01)
    return 90.0 - np.degrees(np.arctan2(n1 - n0, e1 - e0))


def true_horizon(profile):
    """A ``HorizonProfile`` with its grid bearings rotated to true bearings."""
    from eigsep_sim.design_matrix import HorizonProfile

    return HorizonProfile(profile.bearings_deg + grid_to_true_deg(),
                          profile.elev_rad)


@lru_cache(maxsize=None)
def _dem():
    with np.load(campaign_root() / DEM_PATH) as z:
        return (z["dem"].astype(float), int(z["e0_px"]), int(z["n0_px"]),
                float(z["res"]))


@lru_cache(maxsize=None)
def antenna_enu(era):
    """Antenna (E, N, U) in the DEM's local grid for a height era key.

    ``era`` may also be ``"ground+X"`` (X in metres): the point X above the
    ground under the antenna, for asking when the ground there is lit.
    """
    meta = json.loads((campaign_root() / HORIZON_JSON).read_text())
    e, n, _ = meta["antenna_enu_m"]
    if era.startswith("ground+"):
        return np.array([e, n, meta["ground_under_antenna_m"] + float(era[7:])])
    return np.array([e, n, meta["eras"][era]["antenna_u_m"]])


def geometry_release():
    """``shared.json`` of the geometry release the horizon profiles were built on.

    The antenna position comes from the horizon-profile product, so the
    transmitter must come from the same release: mixing releases would put
    the two ends of the antenna-to-transmitter vector in different fits.
    The release path and checksum are those the profile recorded.
    """
    import hashlib

    meta = json.loads((campaign_root() / HORIZON_JSON).read_text())
    rec = [i for i in meta["provenance"]["inputs"]
           if i["path"].endswith("_marjum_geometry/shared.json")]
    if len(rec) != 1:
        raise ValueError(f"{HORIZON_JSON} does not name exactly one geometry release")
    path = campaign_root().parent / rec[0]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != rec[0]["sha256"]:
        raise ValueError(f"{path} differs from the release {HORIZON_JSON} was built on")
    return json.loads(path.read_text())


def transmitter_enu():
    """Transmitter (E, N, U) from the horizon profiles' geometry release."""
    return np.array(geometry_release()["transmitter"]["recommended_for_propagation"]
                    ["position_enu_m"], float)


def _dem_at(e, n):
    dem, e0, n0, res = _dem()
    x, y = e / res + e0, n / res + n0
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < dem.shape[1] - 1) & (j < dem.shape[0] - 1)
    i, j = np.clip(i, 0, dem.shape[1] - 2), np.clip(j, 0, dem.shape[0] - 2)
    fx, fy = x - i, y - j
    v = (dem[j, i] * (1 - fx) * (1 - fy) + dem[j, i + 1] * fx * (1 - fy)
         + dem[j + 1, i] * (1 - fx) * fy + dem[j + 1, i + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def trace_horizon(bearing_true_deg, era, step_m=0.5, max_m=6000.0):
    """Horizon elevation (deg) and edge distance (m) along true bearings.

    Marches the DEM outward from the antenna to the tile edge, with geometric
    Earth curvature and no refraction (as ``horizon_profiles_v0002``).
    """
    ant = antenna_enu(era)
    r = np.arange(2.0, max_m, step_m)
    elev, dist = [], []
    for b in np.atleast_1d(bearing_true_deg):
        g = np.radians(b - grid_to_true_deg())
        u_ = _dem_at(ant[0] + r * np.cos(g), ant[1] + r * np.sin(g))
        el = np.arctan2(u_ - ant[2] - r**2 / (2 * R_EARTH), r)
        k = int(np.nanargmax(el))
        elev.append(np.degrees(el[k]))
        dist.append(r[k])
    return np.array(elev), np.array(dist)


def sun_altaz(t_unix):
    """Apparent topocentric Sun altitude and azimuth (deg, E of N), no refraction."""
    lat, lon, hgt = MARJUM_PASS
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=hgt * u.m)
    t = Time(np.atleast_1d(t_unix), format="unix")
    aa = get_sun(t).transform_to(AltAz(obstime=t, location=loc))
    return aa.alt.deg, aa.az.deg


def knife_edge_power(theta_deg, dist_m, wavelength_m):
    """Fresnel knife-edge power for a source ``theta`` above the edge.

    ``theta < 0`` is behind the edge. Broadcasts; returns 0.25 at theta = 0.
    """
    v = -np.radians(theta_deg) * np.sqrt(2 * dist_m / wavelength_m)
    s, c = fresnel(v)
    return 0.5 * ((0.5 - c) ** 2 + (0.5 - s) ** 2)


def sun_geometry(t_unix, eras):
    """Per-time Sun position and terrain edge along its bearing.

    ``eras`` is the horizon-profile era key per time (e.g. "87.5m"). Returns a
    dict of arrays: alt, az, enu (3, n), horizon_deg, edge_m, theta_deg.
    """
    t_unix = np.atleast_1d(t_unix)
    eras = np.broadcast_to(np.asarray(eras, dtype=object), t_unix.shape)
    alt, az = sun_altaz(t_unix)
    bearing = 90.0 - az
    hz = np.full(t_unix.shape, np.nan)
    edge = np.full(t_unix.shape, np.nan)
    for era in set(eras):
        m = eras == era
        hz[m], edge[m] = trace_horizon(bearing[m], era)
    a, b = np.radians(alt), np.radians(bearing)
    enu = np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])
    return {"alt": alt, "az": az, "enu": enu, "horizon_deg": hz,
            "edge_m": edge, "theta_deg": alt - hz}


def visibility(geom, freqs_hz, model="knife"):
    """Terrain visibility of the Sun, ``(nfreq, ntime)``."""
    lam = C_LIGHT / np.asarray(freqs_hz)[:, None]
    if model == "hard":
        return np.broadcast_to(geom["theta_deg"] > 0, (len(lam),
                               len(geom["alt"]))).astype(float)
    if model == "flat":  # no terrain: the geometric horizon at 0 deg
        return np.broadcast_to(geom["alt"] > 0, (len(lam),
                               len(geom["alt"]))).astype(float)
    if model == "knife":
        return knife_edge_power(geom["theta_deg"][None], geom["edge_m"][None], lam)
    raise ValueError(f"unknown visibility model {model!r}")


def sun_column(beam, rot_body2enu, geom, model="knife"):
    """Antenna temperature (K) per SFU of solar flux, ``(nfreq, ntime)``.

    ``beam`` is an ``eigsep_sim.design_matrix.HealpixBeam`` (total power);
    ``rot_body2enu`` is ``(ntime, 3, 3)``. Unpolarized source on one linear
    polarization: T = S A_e / 2k with A_e = lambda^2 G / 4 pi.
    """
    body = np.einsum("tji,jt->it", rot_body2enu, geom["enu"])
    b = beam(body)  # (nfreq, ntime)
    npix = beam.maps.shape[1]
    gain = b * npix / beam.maps.sum(axis=1)[:, None]
    lam = C_LIGHT / beam.freqs_hz[:, None]
    return (SFU * lam**2 / (8 * np.pi * K_B) * gain
            * visibility(geom, beam.freqs_hz, model))


def terrain_events(t0_unix, t1_unix, era, step_s=10.0):
    """Times the Sun's centre crosses the terrain horizon and 0 deg altitude.

    Returns a list of dicts (kind "rise"/"set", horizon "terrain"/"flat",
    t_unix, alt, az, horizon_deg, edge_m), each refined to ``step_s``.
    """
    t = np.arange(t0_unix, t1_unix, 60.0)
    g = sun_geometry(t, era)
    out = []
    for name, x in (("terrain", g["theta_deg"]), ("flat", g["alt"])):
        up = x > 0
        for j in np.flatnonzero(np.diff(up.astype(int))):
            tf = np.arange(t[j], t[j + 1] + step_s, step_s)
            gf = sun_geometry(tf, era)
            xf = gf["theta_deg"] if name == "terrain" else gf["alt"]
            k = int(np.argmax((xf > 0) == up[j + 1]))
            out.append({"kind": "rise" if up[j + 1] else "set", "horizon": name,
                        "t_unix": float(tf[k]), "alt": float(gf["alt"][k]),
                        "az": float(gf["az"][k]),
                        "horizon_deg": float(gf["horizon_deg"][k]),
                        "edge_m": float(gf["edge_m"][k])})
    return sorted(out, key=lambda e: e["t_unix"])
