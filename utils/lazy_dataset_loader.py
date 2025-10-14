import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils.preprocessing import apply_bandpass_filter, baseline_correction, apply_normalization

def get_lazy_paths(data_dir, label_dir):
    data_paths = []
    label_paths = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".pkl"):
            data_paths.append(os.path.join(data_dir, file))
            sub_id = file.split(".")[0]
            label_paths.append(os.path.join(label_dir, sub_id, "After_remarks.mat"))
    return data_paths, label_paths

def load_labels(mat_path):
    mat = loadmat(mat_path)
    after_remark = mat['After_remark']
    raw_scores = after_remark['score'].squeeze()
    raw_vids = after_remark['vid'].squeeze()
    scores = np.array([s.squeeze() for s in raw_scores])
    vids = np.array([v.squeeze() for v in raw_vids])
    sorted_idx = np.argsort(vids)
    labels = np.argmax(scores[sorted_idx, :8], axis=1)
    return labels

class LazyEEGDataset(Dataset):
    def __init__(self, data_paths, label_paths,
                 apply_filter=True, lowcut=1.0, highcut=50.0, fs=250, filter_order=4,
                 apply_baseline=True, baseline_window=20,
                 normalize=True, normalization_method='per_channel'):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.index_map = self._build_index()
        
        self.apply_filter = apply_filter
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.filter_order = filter_order
        self.apply_baseline = apply_baseline
        self.baseline_window = baseline_window
        self.normalize = normalize
        self.normalization_method = normalization_method
       

    def _build_index(self):
        index_map = []
        for i, path in enumerate(self.data_paths):
            with open(path, 'rb') as f:
                eeg = pickle.load(f)
            for j in range(len(eeg)):
                index_map.append((path, self.label_paths[i], j))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        data_path, label_path, trial_idx = self.index_map[idx]
        with open(data_path, 'rb') as f:
            eeg = pickle.load(f)
        labels = load_labels(label_path)
        trial = eeg[trial_idx][:, :7500]  # (32, 7500)
        reshaped = trial.reshape(32, 30, 250).transpose(1, 2, 0)  # (30, 250, 32)
        segment = reshaped[0]
        return torch.tensor(segment, dtype=torch.float32), torch.tensor(labels[trial_idx], dtype=torch.long)

    def __getitem__(self, idx):
        data_path, label_path, trial_idx = self.index_map[idx]
        with open(data_path, 'rb') as f:
            eeg = pickle.load(f)
        labels = load_labels(label_path)
        trial = eeg[trial_idx][:, :7500]  # (32, 7500)

        reshaped = trial.reshape(32, 30, 250).transpose(1, 2, 0)  # (30, 250, 32)
        segment = reshaped[0]
        
        # -------------- Preprocessing --------------
        if self.apply_filter:
            segment = apply_bandpass_filter([segment], self.lowcut, self.highcut, self.fs, self.filter_order)[0]
        if self.apply_baseline:
            segment = baseline_correction([segment], self.baseline_window)[0]
        if self.normalize:
            segment = apply_normalization([segment], method=self.normalization_method)[0]

        return torch.tensor(segment, dtype=torch.float32), torch.tensor(labels[trial_idx], dtype=torch.long)