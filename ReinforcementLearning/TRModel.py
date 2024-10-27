import random


class TRModel:
    def __init__(self):
        self.T = {}
        self.R = {}

    def update(self, state, action, reward, next_state):
        self.T[(state,action)] = next_state
        self.R[(state,action)] = reward

    def showcase(self):
        state, action = random.randint(0, self.T.__sizeof__() - 1)
        next_state = self.T[(state, action)]
        reward = self.R[(state, action)]
        return state, action, reward, next_state


    //Crias ambiente, agente recebe instancia de ambiente e mecanismo, no agente tens run que implementa o q-learning