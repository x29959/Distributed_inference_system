# retrain_dataset.py

from torch.utils.data import Dataset
import torch

class retrain_dataset(Dataset):
    def __init__(self, window_size, data, labels):
        """
        初始化數據集。

        :param window_size: 窗口大小
        :param data: 手勢資料（特徵）
        :param labels: 標籤
        """
        self.data_list = []
        self.window_size = window_size
        self.features = torch.FloatTensor(data)  # 特徵
        self.labels = torch.LongTensor(labels)    # 標籤

        for i in range(len(self.features) - window_size):
            features_subset = self.features[i:i + window_size]
            label_subset = self.labels[i + window_size]
            self.data_list.append([features_subset, label_subset])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return x, y