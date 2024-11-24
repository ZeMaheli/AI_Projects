from memories.base_memory import LearningMemory


class SparseLearningMemory(LearningMemory):
    def __init__(self, default_value = 0.0):
        self.default_value = default_value
        self.memory = {}
        
    def update(self, state, action, q):
        state = tuple(tuple(s) if isinstance(s, list) else s for s in state)
        action = tuple(action) if isinstance(action, list) else action
        self.memory[(state, action)] = q
        
    def Q(self, state, action):
        state = tuple(tuple(s) if isinstance(s, list) else s for s in state)
        action = tuple(action) if isinstance(action, list) else action
        return self.memory.get((state, action), self.default_value)