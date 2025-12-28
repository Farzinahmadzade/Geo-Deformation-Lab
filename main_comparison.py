from analysis.tikhonov_1d import tikhonov_1d
from analysis.savitzky_golay import savitzky_golay_denoise
from analysis.lstm_autoencoder import lstm_denoise
from analysis.metrics import calculate_metrics, plot_psd
from read_data import load_timeseries  # یا مستقیم بخون

# داده رو بخون
ts_raw, dates_dt = load_timeseries(iy=100, ix=120)

# اعمال روش‌ها
ts_tik = tikhonov_1d(ts_raw, alpha=10.0)
ts_sg = savitzky_golay_denoise(ts_raw)
ts_lstm = lstm_denoise(ts_raw)

# محاسبه و چاپ metrics
# ...

# پلات مقایسه
# کد پلات مقایسه

# پلات PSD
plot_psd(ts_raw, ts_tik, ts_sg, ts_lstm, save_path="figures/psd.png")