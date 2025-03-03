from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os.path as p
import os
import glob

class ClassificationSet(Dataset):
    def __init__(self, root_dir:str, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(.5, .5),
            transforms.GaussianBlur(1)
    ])):
        self.root, self.transform = root_dir, transform
        self.meta = pd.read_csv(f'{root_dir}/metadata.csv')
        self.len = len(self.meta)

    def transform_pass(self, transfer_dir:str):
        '''Loads the data from the main directory and runs our transforms on the files.
            Once we have a good idea of what pre-processing is ideal so we don't have
            to run pre-processessing on every file load.
        '''
        if not p.isdir(transfer_dir): os.mkdir(transfer_dir)
        files = glob.glob(f'{self.root}/*')
        for f in files:
            basename = f.split('/')[-1]
            img = self.run_transform(basename)
            img = Image.fromarray(img.numpy(), mode='L')
            img.save(p.join(transfer_dir,basename))
    
    def run_transform(self, filename:str):
        img = np.array(Image.open(p.join(self.root, 'images', filename)), dtype=np.float32)
        img = self.transform(img)

        return img

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        return (self.run_transform(entry.img_filename), entry.target)
    
    def __len__(self):
        return self.len

