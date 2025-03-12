from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import os.path as p
import glob

class HighlighterSet(Dataset):
    def __init__(self, root_dir:str, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(.5, .5),
            transforms.GaussianBlur(1)
    ])):
        self.root, self.transform = root_dir, transform
        self.meta = pd.read_csv(f'{root_dir}/metadata.csv')
        self.len = len(self.meta)

    def run_transform(self, filename:str, is_map:bool):
        filepath = p.join(self.root, 'images', filename) if not is_map else p.join(self.root, 'maps', filename)
        img = np.array(Image.open(filepath), dtype=np.float32)
        img = self.transform(img)

        return img

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        highlighted = self.run_transform(entry.map_filename, True).clip(0,1)
        #mi, ma = highlighted.min(1, keepdim=True)[0], highlighted.max(1, keepdim=True)[0]
        #highlighted = (highlighted-mi)/ma
        return (self.run_transform(entry.img_filename, False), highlighted)
    
    def __len__(self):
        return self.len

