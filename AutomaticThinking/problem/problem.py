from abc import abstractmethod


class Problem:
    def __init__(self, initial_state, operators):
        self.initial_state = initial_state
        self.operators = operators

    @abstractmethod
    def objective(self, state):
        pass

    @abstractmethod
    def heuristic(self, state):
        pass