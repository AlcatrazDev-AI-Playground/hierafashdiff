import sys
import json
import cv2
import numpy as np
sys.path[0] = "/kaggle/hierafashdiff"
from torch.utils.data import Dataset
from utils.config import *

class MyDataset(Dataset):
    def __init__(self):
        json_path = dataset_root + "train.json"
        with open(json_path, 'rt') as f:
            res = json.load(f)
            self.data = res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['pose']
        target_filename = item['gt']
        prompt = item['caption']

        source = cv2.imread(dataset_root + source_filename)
        target = cv2.imread(dataset_root + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
