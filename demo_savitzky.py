import matplotlib.pyplot as plt
from pathlib import Path
from read_data import load_timeseries
from analysis.savitzky_golay import savitzky_golay_denoise

# read data
ts_raw, dates_dt = load_timeseries(iy=100, ix=120)

# Savitzky-Golay
ts_sg = savitzky_golay_denoise(ts_raw, window_length=51, polyorder=3)

# plot
plt.figure(figsize=(14, 7))
plt.plot(dates_dt, ts_raw, '.-', alpha=0.6, label='Raw InSAR Time Series', color='steelblue')
plt.plot(dates_dt, ts_sg, '-', linewidth=3, label='Denoised with Savitzky-Golay', color='green')
plt.gcf().autofmt_xdate()
plt.ylabel("Cumulative Displacement (mm)")
plt.xlabel("Date")
plt.title("Denoising InSAR Time Series using Savitzky-Golay Filter\nWestern Tehran Plain - Pixel (100,120)")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# save
figures_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data\figures")
figures_dir.mkdir(exist_ok=True)
out_fig = figures_dir / "savitzky_golay_denoised.png"
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Savitzky-Golay plot saved: {out_fig}")

plt.show()