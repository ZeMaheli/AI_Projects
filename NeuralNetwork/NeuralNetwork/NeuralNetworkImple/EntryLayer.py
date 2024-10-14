import numpy as np
    
class EntryLayer():
    def __init__(self, input_size):
        self.input_size = input_size
        self.y = [0] * input_size

    def propagate(self, x):
        self.y = x
        return self.y