import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import welch
from pathlib import Path
from analysis.tikhonov_1d import tikhonov_1d
from analysis.savitzky_golay import savitzky_golay_denoise
from analysis.lstm_autoencoder import lstm_denoise
from analysis.metrics import plot_psd
from analysis.metrics import calculate_metrics
from read_data import load_timeseries

# --- READ DATA ---
ts_raw, dates_dt = load_timeseries(iy=100, ix=120)

# --- APPLY METHODS ---
ts_tik = tikhonov_1d(ts_raw, alpha=10.0)
ts_sg = savitzky_golay_denoise(ts_raw)
ts_lstm = lstm_denoise(ts_raw)

# --- CALCULATE METRICS ---
def calculate_metrics(ts_raw, ts_denoised):
    time_numeric = np.arange(len(ts_raw))
    
    slope_denoised = linregress(time_numeric, ts_denoised).slope * 365.25
    
    trend_raw = linregress(time_numeric, ts_raw).intercept + linregress(time_numeric, ts_raw).slope * time_numeric
    residual_raw = ts_raw - trend_raw
    std_raw = np.std(residual_raw)
    
    trend_denoised = linregress(time_numeric, ts_denoised).intercept + linregress(time_numeric, ts_denoised).slope * time_numeric
    residual_denoised = ts_denoised - trend_denoised
    std_denoised = np.std(residual_denoised)
    
    var_raw = np.var(residual_raw)
    var_denoised = np.var(residual_denoised)
    var_red = (var_raw - var_denoised) / var_raw * 100 if var_raw > 0 else 0
    
    return {
        'STD (mm)': round(std_denoised, 2),
        'Var. Red (%)': round(var_red, 1),
        'Rate (mm/yr)': round(slope_denoised, 2)
    }

# Raw
metrics_raw = calculate_metrics(ts_raw, ts_raw)
metrics_raw['Var. Red (%)'] = 0.0

metrics_tik  = calculate_metrics(ts_raw, ts_tik)
metrics_sg   = calculate_metrics(ts_raw, ts_sg)
metrics_lstm = calculate_metrics(ts_raw, ts_lstm)

# --- TABLE ---
print("\n=== Quantitative Metrics ===")
print(f"{'Method':<18} {'STD (mm)':<10} {'Var. Red (%)':<15} {'Rate (mm/yr)'}")
print("-" * 60)
print(f"{'Raw':<18} {metrics_raw['STD (mm)']:<10} {metrics_raw['Var. Red (%)']:<15} {metrics_raw['Rate (mm/yr)']}")
print(f"{'Tikhonov':<18} {metrics_tik['STD (mm)']:<10} {metrics_tik['Var. Red (%)']:<15} {metrics_tik['Rate (mm/yr)']}")
print(f"{'Savitzky-Golay':<18} {metrics_sg['STD (mm)']:<10} {metrics_sg['Var. Red (%)']:<15} {metrics_sg['Rate (mm/yr)']}")
print(f"{'LSTM Autoencoder':<18} {metrics_lstm['STD (mm)']:<10} {metrics_lstm['Var. Red (%)']:<15} {metrics_lstm['Rate (mm/yr)']}")

# --- PSD PLOT ---
f, psd_raw = welch(ts_raw, fs=12)
_, psd_tik = welch(ts_tik, fs=12)
_, psd_sg = welch(ts_sg, fs=12)
_, psd_lstm = welch(ts_lstm, fs=12)

plt.figure(figsize=(10,6))
plt.loglog(f, psd_raw, label='Raw')
plt.loglog(f, psd_tik, label='Tikhonov')
plt.loglog(f, psd_sg, label='Savitzky-Golay')
plt.loglog(f, psd_lstm, label='LSTM Autoencoder')
plt.xlabel('Frequency (1/year)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density Comparison')
plt.legend()
plt.grid(True, which="both")

# SAVE PLOT
figures_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data\figures")
figures_dir.mkdir(exist_ok=True)
plt.savefig(figures_dir / "psd_comparison.png", dpi=300, bbox_inches='tight')
print("PSD plot saved.")

plt.show ()

# --- COMPARSION PLOT ---
plt.figure(figsize=(14,8))
plt.plot(dates_dt, ts_raw, '.-', alpha=0.5, label='Raw', color='gray')
plt.plot(dates_dt, ts_tik, '-', linewidth=2, label='Tikhonov')
plt.plot(dates_dt, ts_sg, '--', linewidth=2, label='Savitzky-Golay')
plt.plot(dates_dt, ts_lstm, '-', linewidth=3, label='LSTM Autoencoder', color='orange')
plt.gcf().autofmt_xdate()
plt.ylabel("Cumulative Displacement (mm)")
plt.xlabel("Date")
plt.title("Comparison of Denoising Methods - Western Tehran Plain")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# SAVE PLOT
plt.savefig(figures_dir / "denoising_comparison.png", dpi=300, bbox_inches='tight')
print("Comparison plot saved!")

plt.show()
plot_psd(ts_raw, ts_tik, ts_sg, ts_lstm, save_path="figures/psd.png")