from memoria_aprendizagem import LearningMemory


class SparseLearningMemory(LearningMemory):
    def __init__(self, default_value = 0.0):
        self.default_value = default_value
        self.memory = {}
        
    def update(self, state, action, q):
        self.memory[(state, action)] = q
        
    def Q(self, state, action):
        return self.memory.get((state, action), self.default_value)