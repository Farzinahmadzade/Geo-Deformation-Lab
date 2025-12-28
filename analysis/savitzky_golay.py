import numpy as np
from scipy.signal import savgol_filter

def savitzky_golay_denoise(ts: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for denoising.
    
    Parameters
    ----------
    ts : np.ndarray
        Raw time series
    window_length : int
        Length of the filter window (must be odd)
    polyorder : int
        Order of the polynomial used to fit the samples
    
    Returns
    -------
    np.ndarray
        Denoised time series
    """
    if window_length % 2 == 0:
        window_length += 1  # باید فرد باشه
    return savgol_filter(ts, window_length=window_length, polyorder=polyorder)