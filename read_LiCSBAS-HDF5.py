import h5py
import matplotlib.pyplot as plt
import csv

h5_path = "000002_028A_05385_191813_filt.hdf5"

with h5py.File(h5_path, "r") as f:
    print("keys:", list(f.keys()))

    disp = f["displacement"][()]      # (time, y, x)
    dates = f["date"][()]             # (time,)
    lon   = f["longitude"][()]        # (y, x)
    lat   = f["latitude"][()]         # (y, x)

print("disp, dates shape:", disp.shape, dates.shape)

iy, ix = 100, 120
ts = disp[:, iy, ix]

plt.figure()
plt.plot(dates, ts, ".-")
plt.xticks(rotation=45)
plt.ylabel("displacement (mm)")
plt.tight_layout()
plt.show()

out_csv = "timeseries_pixel_100_120.csv"
with open(out_csv, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["date", "disp_mm"])
    for d, v in zip(dates, ts):
        w.writerow([int(d), float(v)])