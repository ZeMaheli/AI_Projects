from agents.q_learning import QLearning
from memories.base_memory import LearningMemory
from models.tr_model import TRModel
from policies.base_policy import ActionSelection


class DynaQ(QLearning):
    def __init__(self, learn_mem: LearningMemory, action_select: ActionSelection, alpha, gamma, simulations):
        super().__init__(learn_mem, action_select, alpha, gamma)
        self.simulations = simulations
        self.model = TRModel()

    def learn(self, state, action, reward, next_state, next_action=None):
        super().learn(state, action, reward, next_state)
        self.model.update(state, action, reward, next_state)
        self.simulate()

    def simulate(self):
        for i in range(self.simulations):
            state, action, reward, next_state = self.model.showcase()
            super().learn(state, action, reward, next_state)