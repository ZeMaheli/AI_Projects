import numpy as np
import BaseLayer

""" class EntryLayer():
    def __init__(self, input_size):
        self.input_size = input_size
        self.y = [0] * input_size

    def propagate(self, x):
        self.y = x
        return self.y """

class EntryLayer:
    def __init__(self, input_size):
        self._y = [0] * input_size  # Use a private attribute to store the data

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value  # Allow setting the 'y' property

    def propagate(self, x):
        self.y = x  # Now this will call the setter and update self._y
        return self.y
