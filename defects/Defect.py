import numpy as np
import pandas as pd
import numpy.random as r
import matplotlib.pyplot as plt
from PIL import Image

class Defect():
    def __init__(self, growth_odds:int = 10, growth_factor:float = .5, initial_growth:int = 1,  size:int = 64, id:int = r.randint(100,999)):
        ''''''
        self.id, self.growth_odds, self.growth_factor, self.filename, self.map_filename = id, growth_odds, growth_factor, f'{self.__class__.__name__}_{id}', f'{self.__class__.__name__}_{id}_map', 
        self.frame = r.random((size,size)) # Create our base frame image
        self.map = np.zeros((size,size))
        self.co = np.array((r.randint(0,size-1), r.randint(0,size-1))) # Seed coordinant to grow the defect around
        self.defect_co = [self.co]
        self.growth = initial_growth

        self.saved_timestamps = []

    def advance(self) -> None:
        '''Randomly decide if indent growing'''
        if self.random_gate(self.growth_odds): self.growth += self.growth_factor

    def paint_area(self, point:tuple) -> None:
        '''Paints an area on the image array and map array'''
        s0,e0 = point[0]-self.growth, point[0]+self.growth
        s1,e1 = point[1]-self.growth, point[1]+self.growth
        self.frame[s0:e0,s1:e1] = np.clip(self.frame[s0:e0,s1:e1] + 0.3, 0, 4.5)
        self.frame[point[0],point[1]] = np.clip(self.frame[point[0],point[1]] + .1, 0, 5)
        self.map[s0:e0,s1:e1] = 1

    def check_bounds(self, co) -> np.array:
        '''Checks whether the given coordinant is out of bounds and sets it in bounds if so'''
        if co[0] >= self.frame.shape[0]: co[0] = self.frame.shape[0] - 1
        elif co[0] < 0: co[0] = 0
        if co[1] >= self.frame.shape[1]: co[1] = self.frame.shape[1] - 1
        elif co[1] < 1: co[1] = 0
        return co
    
    def random_gate(self, odds:int) -> bool:
        '''Returns True if after a coin flip with the given odds. Odds should be > 0 and < 100'''
        return r.randint(0,100) >= odds

    def save_image(self, time_stamp:int, dir:str) -> None:
        '''Saves the resultant image to the given directory'''
        self.saved_timestamps.append(time_stamp)
        for p in [(self.filename, self.frame),(self.map_filename, self.map)]:
            out = f'{dir}/images/{p[0]}_{time_stamp}.png'
            im = Image.fromarray((p[1]*100).astype(np.uint8), mode='L')
            im.save(out)
        
    def row(self) -> pd.DataFrame:
        '''Returns a row with the information of the sample'''
        if len(self.saved_timestamps) == 0: print('No timestamp saved yet, makesure to run .save_image before trying to run .row')
        return pd.DataFrame([{'id':self.id, 'img_filename':f'{self.filename}_{timestamp}.png', 'map_filename':f'{self.map_filename}_{timestamp}.png','target':self.__class__.__name__} for timestamp in self.saved_timestamps])

    def show(self, with_map:bool = False, axe = plt) -> None:
        ''''''
        
        if with_map:
            axe.subplot(1,2,2); plt.imshow(self.map)
            axe.subplot(1,2,1); plt.imshow(self.frame)
        else: axe.imshow(self.frame)
        #plt.show()