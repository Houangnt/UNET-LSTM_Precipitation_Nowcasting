import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def get_data(configs):
    files = os.listdir(configs.path_root)
    files = sorted(files)
    sample_data = []
    for file in tqdm(files):
        real_path = os.path.join(configs.path_root, file)
        image = Image.open(real_path)
        # Convert image to RGB (3 channels)
        image = image.convert('RGB')
        image = image.resize((configs.img_width, configs.img_width))
        data = np.array(image).astype(np.float32)  # No need to expand dimensions
        sample_data.append(data)
    sample_data = np.stack(sample_data, 0)
    return sample_data

class SplitDataset(Dataset):
    def __init__(self, data, configs):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.configs = configs

    def __getitem__(self, item):
        # (b, 128, 128, 3)
        inputs = self.data[item: item + self.configs.total_length]
        mask_true = torch.zeros((self.configs.pred_length - 1, self.configs.img_width, self.configs.img_width, 3),
                                dtype=torch.float32)
        return inputs, mask_true

    def __len__(self):
        return len(self.data) - self.configs.total_length
