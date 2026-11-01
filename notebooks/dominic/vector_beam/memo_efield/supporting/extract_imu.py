"""Pull imu_el accelerometer + times from every raster file (07-17 ~20:20-21:35 UTC) into imu_el_raster.npz."""
import h5py, json, glob, numpy as np
D = __import__("os").environ["DEPLOYMENT5_DATA"]  # .../deployment5_filtered
files = sorted(f for f in glob.glob(D + "/corr_20260717_*.h5") if "201500Z" <= f[-10:-3] <= "213500Z")
T, A, ST, cal = [], [], [], None
for fn in files:
    with h5py.File(fn, "r") as f:
        t = f["header/times"][()]; m = json.loads(f["metadata/imu_el"][()])
        if cal is None: cal = json.loads(f["header/imu_calibration"][()])["imu_el"]
        if len(m) != len(t): print("length mismatch", fn, len(m), len(t)); continue
        for ti, r in zip(t, m):
            ok = r.get("status") == "update" and r.get("accel_x") is not None
            T.append(ti); ST.append(ok); A.append([r["accel_x"], r["accel_y"], r["accel_z"]] if ok else [np.nan] * 3)
T, A, ST = np.array(T), np.array(A, float), np.array(ST)
print(len(files), "files", len(T), "samples", ST.mean().round(3), "valid; time span", (T.max() - T.min()) / 60, "min")
np.savez("imu_el_raster.npz", t=T, acc=A, ok=ST, bias=cal["accel_bias"], scale=cal["accel_scale"], M=np.array(cal["M"]))
