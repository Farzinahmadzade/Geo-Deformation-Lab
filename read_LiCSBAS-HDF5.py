import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd

data_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data")
h5_path = data_dir / "000002_028A_05385_191813_filt.hdf5"

# --- read HDF5 ---
with h5py.File(h5_path, "r") as f:
    print("keys:", list(f.keys()))

    disp = f["cum"][()]          # (time, y, x)
    dates = f["imdates"][()]     # (time,)
    lon   = f["post_lon"][()]    # (y, x)
    lat   = f["post_lat"][()]    # (y, x)

print("disp, dates shape:", disp.shape, dates.shape)

iy, ix = 100, 120
ts = disp[:, iy, ix]

# convert dates to string then datetime
dates_str = [d.decode() if isinstance(d, bytes) else str(d) for d in dates]
dates_dt = pd.to_datetime(dates_str, format="%Y%m%d")

# --- plot with datetime on x-axis ---
plt.figure()
plt.plot(dates_dt, ts, ".-")
plt.gcf().autofmt_xdate()
plt.ylabel("cumulative displacement (mm)")
plt.tight_layout()
plt.show()

# --- save CSV ---
out_csv = data_dir / "timeseries_pixel_100_120.csv"
with open(out_csv, "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["date", "cum_disp_mm"])
    for d, v in zip(dates_str, ts):
        w.writerow([d, float(v)])