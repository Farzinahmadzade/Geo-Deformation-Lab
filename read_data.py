import numpy as np
import h5py
from pathlib import Path
import pandas as pd

def load_timeseries(iy=100, ix=120):
    data_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data")
    h5_path = data_dir / "000002_028A_05385_191813_filt.hdf5"
    
    with h5py.File(h5_path, "r") as f:
        disp = f["cum"][()]
        dates = f["imdates"][()]
        dates_str = [d.decode() if isinstance(d, bytes) else str(d) for d in dates]
        dates_dt = pd.to_datetime(dates_str, format="%Y%m%d")
    
    ts_raw = disp[:, iy, ix].astype(np.float32)
    return ts_raw, dates_dt