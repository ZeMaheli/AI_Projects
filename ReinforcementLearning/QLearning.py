from memoria_aprendizagem import ReinforcementLearn


class QLearning(ReinforcementLearn):
    def learn(self, state, action, reward, next_state):
        an = super().action_select.max_action(next_state)
        qsa = 