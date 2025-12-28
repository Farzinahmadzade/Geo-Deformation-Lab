import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd

# --- SETTING ---
data_dir = Path(r"K:\GitHub\Geo-Deformation-Lab\Data")
h5_path = data_dir / "000002_028A_05385_191813_filt.hdf5"
iy, ix = 100, 120
seq_len = 20
epochs = 200
batch_size = 16
hidden_size = 32
noise_factor = 0.05

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

# --- NORMALIZE ---
ts_mean = ts_raw.mean()
ts_std = ts_raw.std() + 1e-8
ts_norm = (ts_raw - ts_mean) / ts_std

# --- Make sequences ---
def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
    return np.array(xs, dtype=np.float32)

sequences = create_sequences(ts_norm, seq_len)
print("Number of sequences:", len(sequences))

# --- NOISE ---
noisy_sequences = sequences + noise_factor * np.random.normal(size=sequences.shape).astype(np.float32)

# --- DataLoader ---
dataset = TensorDataset(torch.from_numpy(noisy_sequences), torch.from_numpy(sequences))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- LSTM Autoencoder Model ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        _, (h_n, c_n) = self.encoder(x)  # h_n: (num_layers, batch, hidden)
        
        # Repeat hidden state for decoder
        decoder_hidden = (h_n, c_n)
        
        # Start decoder with zeros or use teacher forcing with noisy input projected
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        
        output, _ = self.decoder(decoder_input, decoder_hidden)
        output = self.output_proj(output)
        
        return output.squeeze(-1)  # (batch, seq_len)

model = LSTMAutoencoder(hidden_dim=hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- TRAIN ---
print("Starting training...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.8f}")

# --- Denoising on the entire series ---
model.eval()
with torch.no_grad():
    input_seqs = torch.from_numpy(sequences).to(device)
    denoised_seqs = model(input_seqs).cpu().numpy()  # (252, 20)
    
    denoised_full_norm = denoised_seqs[0].copy()
    
    for i in range(1, len(denoised_seqs)):
        denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[i][-1])
    
    denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[-1][-1])
    
    print("Final reconstructed length:", len(denoised_full_norm))
    
    denoised_full = denoised_full_norm * ts_std + ts_mean

# --- PLOT ---
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

# SAVE
figures_dir = data_dir / "figures"
figures_dir.mkdir(exist_ok=True)
out_fig = figures_dir / "final_autoencoder_denoised.png"
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"Final figure saved: {out_fig}")

plt.show()