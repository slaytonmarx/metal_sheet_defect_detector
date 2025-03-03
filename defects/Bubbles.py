from defects.Defect import Defect
import numpy.random as r
import numpy as np

class Bubbles(Defect):
    def __init__(self, id:int, new_bubble_odds:int = 50, growth_odds:int = 10, growth_factor:float = .5, initial_growth:int = 1):
        super().__init__(growth_odds, round(growth_factor), initial_growth, id=id)
        self.new_bubble_odds = new_bubble_odds
        
        for i in range(r.randint(1,10)): self.generate_bubble()

        self.advance(5)

    def advance(self, steps:int = 1):
        super().advance()

        for i in range(r.randint(1, steps+1)):
            if self.random_gate(self.new_bubble_odds): self.generate_bubble()

            for co in self.defect_co:
                co = self.check_bounds(co + np.array((r.randint(-2,2),r.randint(-2,2))))
                self.paint_area(co)

    def generate_bubble(self):
        '''Picks a new random coordinant and adds it to the defect list'''
        co = self.check_bounds((np.array((r.randint(0,self.frame.shape[0]-1), r.randint(0,self.frame.shape[1]-1)))))
        self.defect_co.append(co)
        self.frame[*co] = 3
