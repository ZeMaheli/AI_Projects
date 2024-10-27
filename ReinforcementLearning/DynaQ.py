from TRModel import TRModel


class DynaQ(QLearning):
    def __init__(self, learn_mem, action_select, alpha, gama, num_sim):
        super().__init__(learn_mem, action_select, alpha, gama)
        self.num_sim = num_sim
        self.model = TRModel()

    def learn(self, state, action, reward, next_state):
        super().learn(state, action, reward, next_state)
        self.model.update(state, action, reward, next_state)
        self.simulate()

    def simulate(self):
        for i in range(self.num_sim):
            state, action, reward, next_state = self.model.showcase()
            super().learn(state, action, reward, next_state)