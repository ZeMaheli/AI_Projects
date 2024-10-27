from abc import ABC, abstractmethod

class LearningMemory():
    @abstractmethod
    def update(self, state, action, q):
        pass

    @abstractmethod
    def Q(self, state, action):
        pass

class ActionSelection():
    def __init__(self, learn_mem):
        self.learn_mem = learn_mem

    @abstractmethod
    def select_action(state):
        pass

class ReinforcementLearn:
    def __init__(self, alpha, gama, learn_mem, action_select):
        self.alpha = alpha
        self.gama = gama
        self.learn_mem = learn_mem
        self.action_select = action_select

    def reinforcement_learning(learn_mem, action_select, alpha, gama):
        pass

    def learn(state, action, r, sn, an):
        pass

class ReinforcementLearningMechanism():
    def __init__(self, ):
        