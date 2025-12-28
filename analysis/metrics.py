import numpy as np
from scipy.stats import linregress
from scipy.signal import welch

def calculate_metrics(ts_raw, ts_denoised):
    # همان کد قبلی
    ...

def plot_psd(ts_raw, ts_tik, ts_sg, ts_lstm, save_path=None):
    # کد پلات PSD
    if save_path:
        plt.savefig(save_path)
    plt.show()