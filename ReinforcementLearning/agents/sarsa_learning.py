from agents.base_agent import ReinforcementLearn


class SarsaLearning(ReinforcementLearn):
    def learn(self, state, action, reward, next_state, next_action=None):
        current_q = self.learn_memory.Q(state, action)
        next_q = self.learn_memory.Q(next_state, next_action)
        updated_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.learn_memory.update(state, action, updated_q)