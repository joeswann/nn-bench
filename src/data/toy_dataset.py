import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_classes=10):
        self.data = self.generate_data(num_samples, seq_length, num_classes)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx][:-1], dtype=torch.long),
            "target": torch.tensor(self.data[idx][1:], dtype=torch.long)
        }

    def generate_data(self, num_samples, seq_length, num_classes):
        data = []
        for _ in range(num_samples):
            x = np.linspace(0, 4 * np.pi, seq_length)
            y = np.sin(x) + np.random.normal(0, 0.1, seq_length)
            # Convert continuous values to discrete classes
            y_classes = np.digitize(y, bins=np.linspace(-1.1, 1.1, num_classes+1)[:-1])
            # Ensure classes are in the range [0, num_classes-1]
            y_classes = np.clip(y_classes, 0, num_classes - 1)
            data.append(y_classes)
        return np.array(data)

class ToyData:
    def __init__(self, num_samples=1000, seq_length=50, num_classes=10):
        self.dataset = ToyDataset(num_samples, seq_length, num_classes)
        self.input_size = num_classes
        self.output_size = num_classes

    def get_data(self):
        return self.dataset, self.input_size, self.output_size

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_hyperparams(self):
        return {
            'steps': 5,
            'step_size': 0.1,
            'solver': 'RungeKutta',
            'adaptive': True
        }
