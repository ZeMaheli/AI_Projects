from memories.base_memory import LearningMemory
from memories.memory_experience import MemoryExperience
from agents.q_learning import QLearning
from policies.base_policy import ActionSelection


class QME(QLearning):
    def __init__(self, learn_mem: LearningMemory, action_select: ActionSelection, alpha, gama, simulations,
                 max_dimension):
        super().__init__(learn_mem, action_select, alpha, gama)
        self.simulations = simulations
        self.memory_experience = MemoryExperience(max_dimension)

    def learn(self, state, action, reward, next_state, next_action=None):
        super().learn(state, action, reward, next_state)
        experience = (state, action, reward, next_state)
        self.memory_experience.update(experience)
        self.simulate()

    def simulate(self):
        samples = self.memory_experience.showcase(self.simulations)
        for (state, action, reward, next_state) in samples:
            super().learn(state, action, reward, next_state)
