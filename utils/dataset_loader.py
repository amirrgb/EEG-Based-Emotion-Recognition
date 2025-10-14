import os
import numpy as np
import torch
import pickle
import gc
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from utils.preprocessing import apply_bandpass_filter, baseline_correction, apply_normalization, apply_pca
from utils.feature_extraction import  calculate_psd, calculate_de
from sklearn.model_selection import train_test_split
from utils.lazy_dataset_loader import LazyEEGDataset, get_lazy_paths

global processed_dir, label_dir, save_dir
processed_dir = r"C:\FACED dataset\Processed_data"
label_dir = r"C:\FACED dataset\Data"
save_dir = r"C:\FACED dataset\Preprocessed_data"

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def get_subject_list(processed_dir, selected_subjects=None):
    subjects = sorted([f for f in os.listdir(processed_dir) if f.endswith(".pkl")])
    if selected_subjects:
        subjects = [s for s in subjects if s.split(".")[0] in selected_subjects]
    return subjects

def load_eeg_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        eeg_data = pickle.load(f)
    return eeg_data

def load_labels(mat_path):
    mat = loadmat(mat_path)
    after_remark = mat['After_remark']
    raw_scores = after_remark['score'].squeeze()
    raw_vids = after_remark['vid'].squeeze()
    scores = np.array([s.squeeze() for s in raw_scores])        # (28, 12)
    vids = np.array([v.squeeze() for v in raw_vids])            # (28,)
    sorted_idx = np.argsort(vids)
    labels = np.argmax(scores[sorted_idx, :8], axis=1)
    return sorted_idx, labels

def segment_reshape_eeg_data(eeg_data, sorted_idx, labels, window_size=250, stride=250):
    X_all = []
    y_all = []
    eeg_sorted = eeg_data[sorted_idx]  # (28, 32, 7500)
    for trial, label in zip(eeg_sorted, labels):
        trial = trial[:, :7500] 
        T = trial.shape[1]
        
        for start in range(0, T - window_size + 1, stride):
            seg = trial[:, start:start + window_size]  # shape: (32, window)
            seg = seg.T  # (window, 32)
            X_all.append(seg)
            y_all.append(label)
            
    return X_all, y_all

def load_processed_data(processed_dir, label_dir, window_size=250, stride=250, selected_subjects=None):
    X_all = []
    y_all = []

    subjects = get_subject_list(processed_dir, selected_subjects)

    for sub_file in subjects:
        sub_id = sub_file.split(".")[0]  # sub000
        eeg_path = os.path.join(processed_dir, sub_file)
        mat_path = os.path.join(label_dir, sub_id, 'After_remarks.mat')

        eeg_data = load_eeg_data(eeg_path)
        sorted_idx, labels = load_labels(mat_path)
        X, y = segment_reshape_eeg_data(eeg_data, sorted_idx, labels, window_size,stride)

        X_all.extend(X)
        y_all.extend(y)

    return np.array(X_all).astype(np.float32), np.array(y_all).astype(np.float32)

def load_data(load_method="normal",window_size=250, stride=250, apply_filter=False, lowcut=1.0,
    highcut=50.0, fs_value=250, filter_order=4,apply_baseline=False, baseline_window=20,
    normalize=False, normalization_method='per_channel',apply_train_test_split=False,test_size=0.2,
    random_seed=42,batch_size=32,save_if_not_pre_exists=False,X_all=None,y_all=None,num_classes=2,
    use_psd=False,psd_bands=[2,3],use_psd_pca=False,use_de=False,de_bands=[2,3],use_de_pca=False,use_pca=False,sub_ids=None):

    if load_method == "lazy":
        train_loader, test_loader =load_lazy_data(apply_filter, lowcut, highcut, fs_value, filter_order,
                  apply_baseline, baseline_window, normalize, normalization_method,
                  test_size, random_seed, batch_size)
        return train_loader, test_loader
    
    file_name = "pre_XY"
    file_name += f"_win{window_size}_str{stride}"
    if apply_filter:
        file_name += f"_low{int(lowcut)}_high{int(highcut)}_fs{fs_value}_order{filter_order}"
    if apply_baseline:
        file_name += f"_base{baseline_window}"
    if normalize:
        file_name += f"_norm_{normalization_method}"
    if use_psd:
        file_name += f"_psd{psd_bands}"
        if use_psd_pca:
            file_name += "pca"
    if use_de and window_size >= 500:
        file_name += f"_de{de_bands}"
        if use_de_pca:
            file_name += "pca"
    if use_pca:
        file_name += "_pca"
    file_name += f".pkl"

    full_path = os.path.join(save_dir, file_name)
    print("full_path :",full_path)
    
    if os.path.isfile(full_path):
        print("file exists , loading ...")
        with open(full_path, 'rb') as f:
            X_all, y_all = pickle.load(f)
        print("Existing Data Shape:", X_all.shape, y_all.shape)
    else:
        print("file not exists, creating ...")
        if (X_all is not None) and (y_all is not None):
            print("X_all and y_all passed")
        else:
            X_all, y_all = load_processed_data(
                processed_dir=processed_dir,
                label_dir=label_dir,
                window_size=window_size,
                stride=stride,
                selected_subjects=sub_ids)
            
        print("Loaded Data Shape:", X_all.shape, y_all.shape)

        features = []

        if apply_filter:
            print("start to bandpass filter")
            X_all = apply_bandpass_filter(X_all, lowcut, highcut, fs_value, filter_order)

        psd_all_std = None
        de_all_std = None

        if use_psd:
            print("start to calculate psd")
            psd_all = calculate_psd(X_all,psd_bands,256,use_psd_pca,16)  # (N, T, C')
            psd_all_std = psd_all.std()
            print("psd_all after normalization:", psd_all.shape, psd_all.min(), psd_all.max(), psd_all.mean(), psd_all_std)
            features.append(psd_all)

        if use_de and window_size >= 500:
            print("start to calculate de")
            de_all = calculate_de(X_all,de_bands,use_de_pca,16)  # (N, T, C'')
            de_all_std = de_all.std()
            print("de_all after normalization:", de_all.shape, de_all.min(), de_all.max(), de_all.mean(), de_all_std)
            features.append(de_all)

        if apply_baseline:
            print("start to baseline correction")
            X_all = baseline_correction(X_all, baseline_window)

        if use_pca:
            print("start to pca")
            X_all = apply_pca(X_all,16)

        if normalize:
            print("start to normalization")
            X_all = apply_normalization(X_all, method=normalization_method)

        if use_psd or use_de:
            print("start to concat data")
            X_all_std = X_all.std()
            print("X_all after normalization:", X_all.shape, X_all.min(), X_all.max(), X_all.mean(), X_all_std)
            features.insert(0, X_all)
            X_all = np.concatenate(features, axis=2)

            if ((use_psd) and np.abs(X_all_std - psd_all_std) > 0.5) or ((use_de) and window_size >= 500 and
                                                                          np.abs(X_all_std - de_all_std) > 0.5):
                print("start to normalizing combined X_all because of std difference :","X_all_std : ",X_all_std,
                "psd_all_std :",psd_all_std if psd_all_std else "undefined","de_all_std :",de_all_std if de_all_std else "undefined")
                print("combined X_all before normalization:", X_all.shape, X_all.min(), X_all.max(), X_all.mean(), X_all.std())
                X_all = apply_normalization(X_all, method=normalization_method)
                print("combined X_all after normalization:", X_all.shape, X_all.min(), X_all.max(), X_all.mean(), X_all.std())

        print("Proccesed Data Shape:", X_all.shape, y_all.shape)

        if (save_if_not_pre_exists or load_method == "pre") and sub_ids == None:
            print("saving preprocessed file to :\n",full_path)
            with open(full_path, 'wb') as f:
                pickle.dump((X_all, y_all), f)

    if num_classes == 2:
        print("start to binarize labels")
        positive_classes = [4, 5, 6, 7]
        y_all = np.array([1 if lbl in positive_classes else 0 for lbl in y_all])
        print("y_all classes : ",np.unique(y_all),"Class counts:", np.bincount(y_all))

    if apply_train_test_split:
        print("start to train_test_split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_seed)
        del X_all
        del y_all
        gc.collect()
        print("start to dataloader creation")
        num_pos = (y_train == 1).sum()
        num_neg = (y_train == 0).sum()
        ratio = num_neg / num_pos
        N,T,input_size = X_train.shape
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        del X_train, X_test, y_train, y_test
        del train_dataset
        del test_dataset
        gc.collect()
        print("returned train_loader, test_loader")
        return train_loader, test_loader, ratio, input_size
    
    print("Final Data Shape:", X_all.shape, y_all.shape)
    print("returned X_all, y_all")
    return X_all, y_all

def load_lazy_data(apply_filter, lowcut, highcut, fs_value, filter_order,
                  apply_baseline, baseline_window, normalize, normalization_method,
                  test_size, random_seed, batch_size):
    print("start to lazy loading")
    data_paths, label_paths = get_lazy_paths(processed_dir, label_dir)
    full_dataset = LazyEEGDataset(
            data_paths, label_paths,
            apply_filter=apply_filter,
            lowcut=lowcut, highcut=highcut, fs=fs_value, filter_order=filter_order,
            apply_baseline=apply_baseline, baseline_window=baseline_window,
            normalize=normalize, normalization_method=normalization_method)
    print("start to train_test_split")
    train_size = int((1 - test_size) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print("end of train_test_split")    
    print("start to dataloader creation")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("dataloader created")
    return train_loader, test_loader


