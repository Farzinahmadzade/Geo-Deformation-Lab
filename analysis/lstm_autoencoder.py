import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def lstm_denoise(ts_raw: np.ndarray, 
                 seq_len: int = 20, 
                 epochs: int = 200, 
                 batch_size: int = 16, 
                 hidden_size: int = 32, 
                 noise_factor: float = 0.05,
                 seed: int = 42) -> np.ndarray:
    """
    Denoise a time series using LSTM Autoencoder.
    
    Parameters
    ----------
    ts_raw : np.ndarray
        Raw time series (1D array)
    seq_len : int
        Sequence length for LSTM
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    hidden_size : int
        Hidden dimension
    noise_factor : float
        Noise level for training
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Denoised time series (same length as input)
    """
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize
    ts_mean = ts_raw.mean()
    ts_std = ts_raw.std() + 1e-8
    ts_norm = (ts_raw - ts_mean) / ts_std
    
    # Create sequences
    def create_sequences(data, seq_len):
        xs = []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i + seq_len])
        return np.array(xs, dtype=np.float32)
    
    sequences = create_sequences(ts_norm, seq_len)
    
    # Add noise
    noisy_sequences = sequences + noise_factor * np.random.normal(size=sequences.shape).astype(np.float32)
    
    # DataLoader
    dataset = TensorDataset(torch.from_numpy(noisy_sequences), torch.from_numpy(sequences))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
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
            decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_dim, device=x.device)
            output, _ = self.decoder(decoder_input, (h_n, c_n))
            output = self.output_proj(output)
            return output.squeeze(-1)
    
    model = LSTMAutoencoder(hidden_dim=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    for epoch in range(epochs):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
    
    # Denoise
    model.eval()
    with torch.no_grad():
        input_seqs = torch.from_numpy(sequences).to(device)
        denoised_seqs = model(input_seqs).cpu().numpy()
        
        # Reconstruct full series
        denoised_full_norm = denoised_seqs[0].copy()
        for i in range(1, len(denoised_seqs)):
            denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[i][-1])
        denoised_full_norm = np.append(denoised_full_norm, denoised_seqs[-1][-1])
        
        denoised_full = denoised_full_norm * ts_std + ts_mean
    
    return denoised_full