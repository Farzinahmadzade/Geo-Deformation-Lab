import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import welch, savgol_filter
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- SETTING ---
data_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data")
h5_path = data_dir / "000002_028A_05385_191813_filt.hdf5"
iy, ix = 100, 120
seq_len = 20
epochs = 200
batch_size = 16
hidden_size = 32
noise_factor = 0.05
alpha_tikhonov = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- READ DATA ---
with h5py.File(h5_path, "r") as f:
    disp = f["cum"][()]
    dates = f["imdates"][()]
    dates_str = [d.decode() if isinstance(d, bytes) else str(d) for d in dates]
    dates_dt = pd.to_datetime(dates_str, format="%Y%m%d")

ts_raw = disp[:, iy, ix].astype(np.float32)
print("Time series length:", len(ts_raw))

# --- TIKHONOV ---
def second_diff_matrix(n):
    L = np.zeros((n-2, n))
    for i in range(n-2):
        L[i, i] = 1
        L[i, i+1] = -2
        L[i, i+2] = 1
    return L

def tikhonov_1d(y, alpha):
    n = len(y)
    I = np.eye(n)
    L = second_diff_matrix(n)
    A = I + alpha * (L.T @ L)
    return np.linalg.solve(A, y)

ts_tikhonov = tikhonov_1d(ts_raw, alpha_tikhonov)

# --- Savitzky-Golay ---
ts_sg = savgol_filter(ts_raw, window_length=51, polyorder=3)

# --- LSTM Autoencoder ---
def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
    return np.array(xs, dtype=np.float32)

ts_norm = (ts_raw - ts_raw.mean()) / (ts_raw.std() + 1e-8)
sequences = create_sequences(ts_norm, seq_len)
noisy_sequences = sequences + noise_factor * np.random.normal(size=sequences.shape).astype(np.float32)

dataset = TensorDataset(torch.from_numpy(noisy_sequences), torch.from_numpy(sequences))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h_n, c_n) = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), seq_len, self.hidden_dim, device=x.device)
        output, _ = self.decoder(decoder_input, (h_n, c_n))
        output = self.output_proj(output)
        return output.squeeze(-1)

model = LSTMAutoencoder(hidden_dim=hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting LSTM training...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.8f}")

# --- Denoising with LSTM ---
model.eval()
with torch.no_grad():
    input_seqs = torch.from_numpy(sequences).to(device)
    denoised_seqs = model(input_seqs).cpu().numpy()
    
    denoised_full_norm = denoised_seqs[0].copy()
    for i in range(1, len(denoised_seqs)):
        denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[i][-1])
    
    denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[-1][-1])
    
    denoised_full = denoised_full_norm * (ts_raw.std() + 1e-8) + ts_raw.mean()

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

metrics_tik = calculate_metrics(ts_raw, ts_tikhonov)
metrics_sg = calculate_metrics(ts_raw, ts_sg)
metrics_lstm = calculate_metrics(ts_raw, denoised_full)

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
_, psd_tik = welch(ts_tikhonov, fs=12)
_, psd_sg = welch(ts_sg, fs=12)
_, psd_lstm = welch(denoised_full, fs=12)

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
plt.plot(dates_dt, ts_tikhonov, '-', linewidth=2, label='Tikhonov')
plt.plot(dates_dt, ts_sg, '--', linewidth=2, label='Savitzky-Golay')
plt.plot(dates_dt, denoised_full, '-', linewidth=3, label='LSTM Autoencoder', color='orange')
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