import numpy as np
import torch


class HWDataset:
    def __init__(self,
                 X_npy_path: str,
                 y_npy_path: str) -> None:
        self.X = np.load(X_npy_path)
        self.y = np.load(y_npy_path)

    def __getitem__(self, index):
        X = torch.FloatTensor(self.X[index])
        y = float(self.y[index])
        return X, y

    def __len__(self):
        return self.X.shape[0]
