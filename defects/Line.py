from defects.Defect import Defect
import numpy.random as r
import numpy as np

class Line(Defect):
    '''Creates a metal plate sample with a wrinkle in it's construction'''
    def __init__(self, id:int, divergence_odds:int = 85, growth_odds:int = 10, growth_factor:float = 1, initial_growth:int = 1):
        super().__init__(growth_odds, growth_factor, initial_growth, id=id)
        self.divergence_odds = divergence_odds
        self.edge_co = [self.co.copy(), self.co.copy()]

        # Initialize the vectors of growth
        vector_check = 0
        while not vector_check:
            self.l1 = np.array((r.randint(-2,2), r.randint(-2,2)))
            vector_check = abs(self.l1).sum()
        self.l2 = -self.l1
        self.vectors = [self.l1, self.l2]

        self.advance(5)

    def advance(self, steps:int=1):
        '''Apply advance function to each of the line's edges'''
        super().advance()

        # Randomly perturb vectors
        for i in range(len(self.vectors)):
            if self.random_gate(self.divergence_odds):
                v = np.clip(self.vectors[i] + np.array((r.randint(-2,2),r.randint(-2,2))), -2, 2)
                if abs(v).sum() != 0: self.vectors[i] = v
        
        # Apply steps
        for i in range(r.randint(1, steps+1)):
            for j in range(2):
                self.edge_co[j] = self.check_bounds(self.edge_co[j] + self.vectors[j])
                self.defect_co.append(self.edge_co[j])
        
        # Increase area
        for co in self.defect_co:
            self.paint_area(co)
            #r.random((self.growth*2, self.growth*2))*self.growth_factor