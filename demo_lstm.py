import matplotlib.pyplot as plt
from pathlib import Path
from read_data import load_timeseries
from analysis.lstm_autoencoder import lstm_denoise

# read data
ts_raw, dates_dt = load_timeseries(iy=100, ix=120)

# LSTM
denoised_full = lstm_denoise(ts_raw)

# plot
plt.figure(figsize=(14, 7))
plt.plot(dates_dt, ts_raw, '.-', alpha=0.6, label='Raw InSAR Time Series', color='steelblue')
plt.plot(dates_dt, denoised_full, '-', linewidth=3, label='Denoised with LSTM Autoencoder', color='orange')
plt.gcf().autofmt_xdate()
plt.ylabel("Cumulative Displacement (mm)")
plt.xlabel("Date")
plt.title("Denoising InSAR Time Series using Deep Learning\nWestern Tehran Plain - Pixel (100,120)")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# save
figures_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data\figures")
figures_dir.mkdir(exist_ok=True)
out_fig = figures_dir / "lstm_denoised.png"
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"LSTM plot saved: {out_fig}")

plt.show()