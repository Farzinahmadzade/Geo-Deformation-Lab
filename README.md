# Geo-Deformation-Lab

A small research-oriented project for analyzing ground deformation time series from LiCSBAS InSAR products using 1‑D Tikhonov regularization (second‑derivative smoothing).

---

## Project goals

- Read LiCSBAS HDF5 products (COMET Subsidence Portal / LiCSBAS SBAS output).
- Extract cumulative displacement time series for selected pixels.
- Apply 1‑D Tikhonov regularization to smooth noisy InSAR time series while preserving the physical trend.
- Prepare the code base for future extensions:
  - space–time regularization in Sobolev subspaces (inspired by GRACE gravity-field filtering),
  - and AI/ML-based denoising and modeling.

---

## Repository structure

Geo-Deformation-Lab/

├── Data/

│      └── 000002_028A_05385_191813_filt.hdf5

├── io/                   # For future I/O modules

├── read_LiCSBAS-HDF5.py  # Main demo script

└── README.md

### `analysis/tikhonov_1d.py`

Implements a simple 1‑D Tikhonov smoother:

- `second_diff_matrix(n)`: builds an \((n-2) \times n\) finite‑difference matrix approximating the second derivative (curvature operator \(L\)).  
- `tikhonov_1d(y, alpha)`: solves
  \[
  \min_x \|x - y\|^2 + \alpha \|Lx\|^2
  \]
  via the linear system
  \[
  (I + \alpha L^\top L)\,x = y,
  \]
  and returns the smoothed time series.

This is a discrete version of classical Tikhonov regularization with a second‑derivative penalty.

### `read_LiCSBAS-HDF5.py`

Demonstration script that:

1. Reads a LiCSBAS HDF5 file:
   - cumulative displacement `cum` (mm),
   - velocity `vel` (mm/yr),
   - acquisition dates `imdates` (YYYYMMDD),
   - post‑stack grid coordinates `post_lat`, `post_lon`.
2. Extracts the cumulative displacement time series for a chosen pixel `(iy, ix)`.
3. Converts `imdates` to Python `datetime` objects using `pandas`.
4. Applies `tikhonov_1d` for selected values of the regularization parameter `alpha` (e.g. 1, 10, 100).
5. Plots:
   - the raw time series,
   - and the Tikhonov‑smoothed series on the same axes.
6. Exports the raw time series to CSV (date vs. cumulative displacement).

---

## Requirements

- Python 3.9+
- Packages:
  - `numpy`
  - `h5py`
  - `pandas`
  - `matplotlib`

Install with:

```pip install numpy h5py pandas matplotlib```

---

## Usage

1. Place a LiCSBAS HDF5 file in the `Data/` directory, e.g.:

```Data/000002_028A_05385_191813_filt.hdf5```

2. Edit the path at the top of `read_LiCSBAS-HDF5.py` if needed:

```
from pathlib import Path

data_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data")
h5_path = data_dir / "000002_028A_05385_191813_filt.hdf5"
```


3. Run the script:

```python read_LiCSBAS-HDF5.py```

You should see:

- a plot of raw cumulative displacement vs. time,
- a plot with both raw and Tikhonov‑smoothed curves for the chosen `alpha`,
- and a CSV file such as `timeseries_pixel_100_120.csv` in `Data/`.

---

## Interpretation of the regularization parameter

The regularization parameter `alpha` controls the trade‑off between fidelity to the data and smoothness:

- **Small `alpha` (e.g. 1)**:
  - The smoothed curve almost follows the raw data.
  - Only very high‑frequency noise and sharp spikes are reduced.
- **Moderate `alpha` (e.g. 10)**:
  - Good compromise: short‑scale noise is suppressed,
  - main physical features (trend, seasonal patterns, major breaks) are preserved.
- **Large `alpha` (e.g. 100)**:
  - Strong smoothing: the long‑term trend is emphasized,
  - small‑scale variations and short events may be over‑smoothed.

For now the parameter is chosen manually; later this can be automated using L‑curve, cross‑validation, or Bayesian criteria.

---

## Current status

- HDF5 I/O for LiCSBAS cumulative displacement time series is working.
- 1‑D Tikhonov smoothing with a second‑difference penalty is implemented and demonstrated on real InSAR subsidence data.
- Plots clearly show how different `alpha` values affect the balance between noise reduction and detail preservation.

---

## Planned extensions

- Implement automated selection of `alpha` (e.g. L‑curve analysis).
- Extend the regularization to 2‑D / 3‑D (space–time) using spatial finite‑difference operators on the LiCSBAS grid.
- Introduce Sobolev‑type norms and connect the implementation more closely to the gravity‑field regularization ideas from GRACE literature.
- Add region‑based time‑series extraction (spatial averaging over polygons).
- Use Tikhonov‑smoothed series as “clean labels” for training ML models (autoencoders, RNNs, etc.) for direct InSAR denoising.

---

## Acknowledgements / context

This project is inspired by:

- LiCSBAS and COMET’s Sentinel‑1 InSAR products.
- Classical Tikhonov regularization theory and its applications in geophysics and inverse problems.
- GRACE time‑variable gravity studies using Tikhonov‑type regularization in Sobolev subspaces.
