from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms.v2 as v2
import pandas as pd
import numpy as np
import shutil
import os.path as p
import os
import glob

class ClassificationSet(Dataset):
    def __init__(self, root_dir:str, meta:pd.DataFrame=[], transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])):
        self.root, self.transform = root_dir, transform
        self.meta = meta if len(meta)>0 else pd.read_csv(f'{root_dir}/metadata.csv')
        self.len = len(self.meta)

    def transform_pass(self, transfer_dir:str, delete_prior_contents:bool = False):
        '''Loads the data from the main directory and runs our transforms on the files.
            Once we have a good idea of what pre-processing is ideal so we don't have
            to run pre-processessing on every file load.
        '''
        if not p.isdir(transfer_dir): os.mkdir(transfer_dir)
        if delete_prior_contents: 
            for file in glob.glob(f'{transfer_dir}/images/*.png'):
                os.remove(file)

        files = glob.glob(f'{self.root}/images/*.png')
        for f in files:
            basename = f.split('/')[-1]
            img = self.run_transform(basename)
            img = Image.fromarray((img.numpy()[0]).astype(np.uint8), mode='L')
            img.save(p.join(p.join(transfer_dir, 'images'),basename))
        shutil.copy(f'{self.root}/metadata.csv', f'{transfer_dir}/metadata.csv')

    def run_transform(self, filename:str):
        img = np.array(Image.open(p.join(self.root, 'images', filename)), dtype=np.float32)
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        return (self.run_transform(entry.img_filename), entry.target)
    
    def __len__(self):
        return self.len

