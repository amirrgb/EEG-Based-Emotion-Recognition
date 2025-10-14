import numpy as np
import mne
import pickle as pkl
import os
from scipy.signal.windows import hann
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_psd(x_all, bands=[2, 3], stft_n=256, use_pca=False, n_components=16):
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
    selected_bands = [freq_bands[i] for i in bands]
    fs = 250  # Sampling frequency
    N, T, C = x_all.shape
    all_bands_processed_features = []

    print("Calculating PSD features band by band...")
    for band_idx, (low, high) in enumerate(selected_bands):
        current_band_psd_list = []
        for seg in x_all: # seg shape: (T, C)
            seg_t = seg.T  # (C, T)
            hdata = seg_t * hann(T)
            fft_data = np.fft.fft(hdata, n=stft_n)
            energy_graph = np.abs(fft_data[:, :stft_n // 2])
            relative_energy = energy_graph / (np.sum(energy_graph, axis=1, keepdims=True) + 1e-8)
            
            start = int(np.floor(low / fs * stft_n))
            end = int(np.floor(high / fs * stft_n))
            band_psd = np.mean(relative_energy[:, start:end] ** 2, axis=1)  # (C,)
            current_band_psd_list.append(band_psd)
            
        band_features_flat = np.stack(current_band_psd_list, axis=0)
        
        if use_pca:
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(band_features_flat)
            all_bands_processed_features.append(reduced_features.astype(np.float32))
        else:
            all_bands_processed_features.append(band_features_flat.astype(np.float32))

    final_features_flat = np.concatenate(all_bands_processed_features, axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(final_features_flat)
    final_features = np.repeat(scaled_features[:, np.newaxis, :], T, axis=1)
    return final_features

def calculate_de(x_all, bands=[2, 3], use_pca=False, n_components=16):
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
    selected_bands = [freq_bands[i] for i in bands]
    fs = 250
    N, T, C = x_all.shape
    all_bands_processed_features = []

    print("Calculating DE features band by band...")
    for band_idx, (low, high) in enumerate(selected_bands):
        current_band_de_list = [
        0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(mne.filter.filter_data(seg.T.astype(np.float64),
        fs, l_freq=low, h_freq=high, verbose=False), axis=1) + 1e-8))
        for seg in x_all  # x_all shape is (N, T, C)
        ]
        band_features_flat = np.stack(current_band_de_list, axis=0)
        if use_pca:
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(band_features_flat)
            all_bands_processed_features.append(reduced_features.astype(np.float32))
        else:
            all_bands_processed_features.append(band_features_flat.astype(np.float32))

    final_features_flat = np.concatenate(all_bands_processed_features, axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(final_features_flat)
    final_features = np.repeat(scaled_features[:, np.newaxis, :], T, axis=1)
    return final_features


# n_vids = 28
# freq = 250
# nsec = 30
# nchn = 32
# #
# freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

# def _get_relative_psd(relative_energy_graph, freq_bands, sample_freq, stft_n=256):
#     start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
#     end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
#     # print(start_index, end_index)
#     psd = np.mean(relative_energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
#     # print('psd:', psd.shape)
#     return psd

# def extract_psd_feature(data, window_size, freq_bands, stft_n=256):
#     sample_freq = freq
#     # Ptr operation
#     if len(data.shape) > 2:
#         data = np.squeeze(data)
#     n_channels, n_samples = data.shape
#     point_per_window = int(sample_freq * window_size)
#     window_num = int(n_samples // point_per_window)
#     psd_feature = np.zeros((window_num, len(freq_bands), n_channels))
#     # print('psd feature shape:', psd_feature.shape)

#     for window_index in range(window_num):
#         start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
#         window_data = data[:, start_index:end_index]
#         hdata = window_data * hann(point_per_window)
#         fft_data = np.fft.fft(hdata, n=stft_n)
#         # print('fft_data shape:',fft_data.shape)
#         energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
#         # print('energy_graph.shape:', energy_graph.shape)
#         relative_energy_graph = energy_graph/np.sum(energy_graph)

#         for band_index, band in enumerate(freq_bands):
#             # band_ave_psd = _get_average_psd(energy_graph, band, sample_freq, stft_n)
#             band_relative_psd = _get_relative_psd(relative_energy_graph, band, sample_freq, stft_n)
#             psd_feature[window_index, band_index, :] = band_relative_psd

#     return psd_feature

# def extract_features():
#     datapaths = os.listdir(processed_dir)
#     datapaths.sort()
#     PSD_save_dir = os.path.join(save_dir,"PSD")
#     if not os.path.exists(PSD_save_dir):
#         os.makedirs(PSD_save_dir)
#     DE_save_dir = os.path.join(save_dir,"DE")
#     if not os.path.exists(DE_save_dir):
#         os.makedirs(DE_save_dir)

#     # (28,32,30,4)
#     subs_psd = np.zeros((n_vids, nchn, nsec, len(freq_bands)))
#     subs_de = np.zeros((n_vids, nchn, nsec, len(freq_bands)))

#     for idx, sub in enumerate(datapaths):

#         f = open(os.path.join(processed_dir, sub), 'rb')

#         data_sub = pkl.load(f)
#         # print(data_sub.shape)        # (28, 32, 7500)

#         # PSD features
#         for video in range(0, n_vids):
#             # (30, 5, 32)
#             psd_data = extract_psd_feature(data_sub[video,:,:], 1, freq_bands)
#             # (32, 30, 5)
#             psd_data = np.transpose(psd_data, (2, 0, 1))
#             subs_psd[video,:,:,:] = psd_data

#         # save the PSD and DE features
#         f = open(PSD_save_dir + '/' + str(sub), 'wb')
#         pkl.dump(subs_psd, f)
#         f.close()

#         # DE features
#         for i in range(len(freq_bands)):
#             for video in range(0, n_vids):
#                 data_video = data_sub[video,:,:]
#                 low_freq = freq_bands[i][0]
#                 high_freq = freq_bands[i][1]
#                 # band pass filter for the specific data band
#                 # shape (32, 7500)
#                 data_video_filt = mne.filter.filter_data(data_video, freq, l_freq=low_freq, h_freq=high_freq)
#                 # shape (32, 30, 250)
#                 data_video_filt = data_video_filt.reshape(nchn, -1, freq)
#                 print('data filtered :', data_video_filt.shape)
#                 de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_video_filt, 2)))
#                 print(de_one.shape)
#                 # n_subs, video, channel,  second, frequency
#                 subs_de[video, : , :, i] = de_one

#         g = open(DE_save_dir + '/' + str(sub), 'wb')
#         pkl.dump(subs_de, g)
#         g.close()


# def extract_psd_de_from_segments(X_all, freq_bands, sample_freq=250, stft_n=256):
#     all_psd_features = []
#     all_de_features = []

#     for segment in X_all:
#         segment = np.array(segment).T  # shape: (n_channels, window_size)
#         n_channels, window_size = segment.shape

#         # ---------------- PSD Feature ----------------
#         hdata = segment * hann(window_size)
#         fft_data = np.fft.fft(hdata, n=stft_n)
#         energy_graph = np.abs(fft_data[:, :stft_n // 2])
#         relative_energy_graph = energy_graph / np.sum(energy_graph, axis=1, keepdims=True)

#         psd_features = []
#         de_features = []
#         for band in freq_bands:
#             start_idx = int(np.floor(band[0] / sample_freq * stft_n))
#             end_idx = int(np.floor(band[1] / sample_freq * stft_n))
#             band_energy = energy_graph[:, start_idx:end_idx]
#             band_relative_energy = relative_energy_graph[:, start_idx:end_idx]

#             psd_band = np.mean(band_relative_energy ** 2, axis=1)  # shape: (n_channels,)
#             psd_features.append(psd_band)

#             # ---------------- DE Feature ----------------
#             filtered_band = band_energy  # Approximation (no filter)
#             de_band = 0.5 * np.log(2 * np.pi * np.exp(1) * np.var(filtered_band, axis=1))  # shape: (n_channels,)
#             de_features.append(de_band)

#         # Transpose to (n_channels, n_bands)
#         psd_features = np.stack(psd_features, axis=1)
#         de_features = np.stack(de_features, axis=1)

#         all_psd_features.append(psd_features)
#         all_de_features.append(de_features)

#     return np.array(all_psd_features), np.array(all_de_features)













