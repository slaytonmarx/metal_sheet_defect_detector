import logging
import numpy as np
import pandas as pd
import numpy.random as r
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Defect:
    def __init__(
        self, growth_odds: int = 10, growth_factor: float = 0.5, initial_growth: int = 1,
        size: int = 32, id: int = None
    ):
        try:
            self.id = id if id is not None else r.randint(100, 999)
            self.growth_odds = growth_odds
            self.growth_factor = growth_factor
            self.filename = f'{self.__class__.__name__}_{self.id}'
            self.map_filename = f'{self.__class__.__name__}_{self.id}_map'
            self.frame = r.random((size, size))  # Create base frame image
            self.map = np.zeros((size, size))
            self.co = np.array((r.randint(0, size - 1), r.randint(0, size - 1)))  # Seed coordinate
            self.defect_co = [self.co]
            self.growth = initial_growth
            self.saved_timestamps = []
            logging.info(f"Initialized Defect with ID: {self.id}")

        except Exception as e:
            logging.error(f"Error initializing Defect: {e}", exc_info=True)

    def advance(self) -> None:
        '''Advances the defect in time, randomly deciding if it should grow in
            size from it's germination points.
        '''
        try:
            if self.random_gate(self.growth_odds):
                self.growth += self.growth_factor
                logging.info(f"Defect {self.id} growth increased to {self.growth}")

        except Exception as e:
            logging.error(f"Error advancing Defect: {e}", exc_info=True)

    def paint_area(self, point: tuple) -> None:
        '''Paints an area on the image array and map array.'''
        try:
            s0, e0 = max(0, point[0] - self.growth), min(self.frame.shape[0], point[0] + self.growth)
            s1, e1 = max(0, point[1] - self.growth), min(self.frame.shape[1], point[1] + self.growth)
            
            self.frame[s0:e0, s1:e1] = np.clip(self.frame[s0:e0, s1:e1] + 0.3, 0, 4.5)
            self.frame[point[0], point[1]] = np.clip(self.frame[point[0], point[1]] + 0.1, 0, 5)
            self.map[s0:e0, s1:e1] = 1

        except Exception as e:
            logging.error(f"Error painting area at {point}: {e}", exc_info=True)

    def check_bounds(self, co) -> np.array:
        '''Ensures coordinates are within valid bounds'''
        try:
            co[0] = np.clip(co[0], 0, self.frame.shape[0] - 1)
            co[1] = np.clip(co[1], 0, self.frame.shape[1] - 1)
            return co
        
        except Exception as e:
            logging.error(f"Error checking bounds for coordinate {co}: {e}", exc_info=True)
            return np.array([0, 0])

    def random_gate(self, odds: int) -> bool:
        '''Returns True if a random chance meets the given odds.'''
        try:
            return r.randint(0, 100) >= odds
        
        except Exception as e:
            logging.error(f"Error in random_gate with odds {odds}: {e}", exc_info=True)
            return False

    def save_image(self, time_stamp: int, dir: str) -> None:
        '''Saves the resultant image to the given directory.'''
        try:
            self.saved_timestamps.append(time_stamp)
            out_frame = f'{dir}/images/{self.filename}_{time_stamp}.png'
            out_map = f'{dir}/images/{self.map_filename}_{time_stamp}.png'

            Image.fromarray((self.frame * 100).astype(np.uint8), mode='L').save(out_frame)
            Image.fromarray((self.map * 100).astype(np.uint8), mode='L').save(out_map)
            
            logging.info(f"Saved images: {out_frame}, {out_map}")

        except Exception as e:
            logging.error(f"Error saving images for Defect {self.id}: {e}", exc_info=True)

    def row(self) -> pd.DataFrame:
        '''Returns a row with the information of the sample.'''
        try:
            if not self.saved_timestamps:
                logging.warning("No timestamp saved yet. Run .save_image before calling .row.")
                return pd.DataFrame()

            return pd.DataFrame([
                {
                    'id': self.id,
                    'img_filename': f'{self.filename}_{timestamp}.png',
                    'map_filename': f'{self.map_filename}_{timestamp}.png',
                    'target': self.__class__.__name__,
                } for timestamp in self.saved_timestamps
            ])
        
        except Exception as e:
            logging.error(f"Error generating row for Defect {self.id}: {e}", exc_info=True)
            return pd.DataFrame()

    def show(self, with_map: bool = False, axe=plt) -> None:
        '''Displays the defect image, optionally with its map.'''
        try:
            if with_map:
                axe.subplot(1, 2, 2)
                plt.imshow(self.map, cmap='gray')
                axe.subplot(1, 2, 1)
                plt.imshow(self.frame, cmap='gray')
            else:
                axe.imshow(self.frame, cmap='gray')
            plt.show()

        except Exception as e:
            logging.error(f"Error displaying Defect {self.id}: {e}", exc_info=True)