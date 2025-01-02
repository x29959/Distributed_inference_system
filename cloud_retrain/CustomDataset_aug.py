import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 資料增強函數
def add_noise(data, noise_factor=0.1):
    noise = np.random.randn(*data.shape) * noise_factor
    augmented_data = data + noise
    return augmented_data

def time_shift(data, shift_max=10):
    shift = np.random.randint(-shift_max, shift_max)
    augmented_data = np.roll(data, shift, axis=0)
    return augmented_data

def time_stretch(data, target_length, stretch_factor=0.8):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    stretched_data = torch.nn.functional.interpolate(data_tensor.unsqueeze(0).unsqueeze(0), size=(target_length, data.shape[1]), mode='bilinear', align_corners=True)
    return stretched_data.squeeze(0).squeeze(0).numpy()

def time_reverse(data):
    reversed_data = data[::-1].copy()
    return reversed_data

def random_drop(data, drop_prob=0.1):
    drop_mask = np.random.rand(*data.shape) > drop_prob
    augmented_data = data * drop_mask
    return augmented_data

def augment_data(data, target_length):
    if np.random.rand() > 0.1:
        data = add_noise(data)
    if np.random.rand() > 0.1:
        data = time_shift(data)
    if np.random.rand() > 0.1:
        data = time_stretch(data, target_length=target_length, stretch_factor=np.random.uniform(1, 1.2))
    if np.random.rand() > 0.1:
        data = time_reverse(data)
    if np.random.rand() > 0.1:
        data = random_drop(data)
    return data

class CustomDataset(Dataset):
    def __init__(self, csv_path, window_size, augment=False):
        data = pd.read_csv(csv_path, header=0).values
        features = data[:, :99].astype(np.float32)
        labels = data[:, 99].astype(np.float32).reshape(-1, 1)
        
        self.data_list = []
        self.window_size = window_size
        self.augment = augment

        for i in range(len(features) - window_size):
            features_subset = features[i:i + window_size]
            labels_subset = labels[i + window_size - 1]  # 使用窗口的最後一個標籤
            self.data_list.append((features_subset, labels_subset))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        
        if self.augment:
            x = augment_data(x, target_length=self.window_size)
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y