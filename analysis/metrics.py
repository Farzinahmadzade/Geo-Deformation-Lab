import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.signal import welch

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


def plot_psd(ts_raw, ts_tik, ts_sg, ts_lstm, save_path=None):
    plt.figure(figsize=(10,6))
    f, psd_raw = welch(ts_raw, fs=12)
    _, psd_tik = welch(ts_tik, fs=12)
    _, psd_sg = welch(ts_sg, fs=12)
    _, psd_lstm = welch(ts_lstm, fs=12)
    
    plt.loglog(f, psd_raw, label='Raw')
    plt.loglog(f, psd_tik, label='Tikhonov')
    plt.loglog(f, psd_sg, label='Savitzky-Golay')
    plt.loglog(f, psd_lstm, label='LSTM Autoencoder')
    plt.xlabel('Frequency (1/year)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density Comparison')
    plt.legend()
    plt.grid(True, which="both")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PSD plot saved: {save_path}")
    
    plt.show()