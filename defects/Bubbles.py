import logging
import numpy as np
import numpy.random as r
from defects.Defect import Defect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Bubbles(Defect):
    '''Defect which creates small bubbles at random points on the metal sheet.
        Bubble sites are randomly selected and then grow larger as the defect
        advances.
    '''

    def __init__(self, id: int, new_bubble_odds: int = 50, growth_odds: int = 10, growth_factor: float = 0.5, initial_growth: int = 1):
        try:
            super().__init__(growth_odds, round(growth_factor), initial_growth, id=id)
            self.new_bubble_odds = new_bubble_odds
            
            # Generate a random number of bubble germination points
            for _ in range(r.randint(1, 10)):
                self.generate_bubble()

            # Advance five steps to ensure initial presence
            self.advance(5)

            logging.info(f"Initialized Bubble with ID: {self.id}")

        except Exception as e:
            logging.error(f"Error initializing Bubble Defect: {e}", exc_info=True)

    def advance(self, steps: int = 1):
        '''Advances the bubble size from the germination points by the given
            number of steps.
        '''
        try:
            super().advance()
            
            for _ in range(r.randint(1, steps + 1)):
                if self.random_gate(self.new_bubble_odds):
                    self.generate_bubble()
                
                for idx, co in enumerate(self.defect_co):
                    new_co = self.check_bounds(co + np.array((r.randint(-2, 2), r.randint(-2, 2))))
                    self.defect_co[idx] = new_co  # Ensure defect_co list is updated
                    self.paint_area(new_co)

        except Exception as e:
            logging.error(f"Error advancing Bubble Defect: {e}", exc_info=True)

    def generate_bubble(self):
        '''Picks a new random coordinate and creates a bubble germination point
            there.
        '''

        try:
            co = self.check_bounds(np.array((r.randint(0, self.frame.shape[0] - 1), r.randint(0, self.frame.shape[1] - 1))))
            self.defect_co.append(co)
            self.frame[tuple(co)] = 3  # Corrected tuple unpacking
            logging.info(f"Generated new bubble at {co}")

        except IndexError as e:
            logging.error(f"Coordinant error during bubble generation at co{co}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error in generating a bubble defect: {e}", exc_info=True)
