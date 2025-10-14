import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------- Normalization ----------------------

def normalize_data_with_mean_std(data):
    for i in range(data.shape[0]):
        seg = data[i]
        mean = seg.mean()
        std = seg.std() + 1e-5
        data[i] = (seg - mean) / std
    return data.astype(np.float32)

def normalize_data_min_max(data):
    for i in range(data.shape[0]):
        seg = data[i]
        seg_min = seg.min()
        seg_max = seg.max()
        seg = (seg - seg_min) / (seg_max - seg_min + 1e-5)
        data[i] = seg
    return data.astype(np.float32)

def normalize_data_per_channel(data):
    for i in range(data.shape[0]):
        seg = data[i]
        seg_norm = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-5)
        data[i] = seg_norm
    return data.astype(np.float32)

def apply_normalization(data, method='zscore'):
    if method == 'zscore':
        return normalize_data_with_mean_std(data)
    elif method == 'minmax':
        return normalize_data_min_max(data)
    elif method == 'per_channel':
        return normalize_data_per_channel(data)
    else:
        raise ValueError("Unknown normalization method. Choose from: 'zscore', 'minmax', 'per_channel'")

def normalize_across_dataset(data, scaler=None):
    """
    Apply normalization across the full dataset using sklearn's StandardScaler.
    Assumes data shape is (N, T, C) and reshapes to (N*T, C) for scaling.
    """
    N, T, C = data.shape
    reshaped = data.reshape(N * T, C)
    scaler = scaler or StandardScaler()
    scaled = scaler.fit_transform(reshaped)
    return scaled.reshape(N, T, C), scaler

# ---------------------- Filtering ----------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=250, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    for i in range(data.shape[0]):
        seg = data[i]
        filtered_seg = lfilter(b, a, seg, axis=0)
        data[i] = filtered_seg
    return data.astype(np.float32)

# ---------------------- Baseline Correction ----------------------

def baseline_correction(data, baseline=20):
    for i in range(data.shape[0]):
        seg = data[i]
        baseline_segment = seg[0:baseline, :]  # first 20 samples as baseline
        baseline_mean = baseline_segment.mean(axis=0)
        seg_corrected = seg - baseline_mean
        data[i] = seg_corrected
    return data.astype(np.float32)

# ---------------------- PCA ----------------------

def apply_pca(data,n_components):
    N, T, C = data.shape
    x_reshaped = data.reshape(-1, C)  # (N*T, C)
    
    pca = PCA(n_components=n_components)
    x_reduced = pca.fit_transform(x_reshaped)  # (N*T, n_components)
    
    x_reduced = x_reduced.reshape(N, T, n_components)  # back to (N, T, k)
    return x_reduced


