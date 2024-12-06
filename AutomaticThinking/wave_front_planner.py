from xmlrpc.client import MININT

from wave_front_algorithm import WaveFront


class WaveFrontPlanner:
    def __init__(self, model, gamma, v_max):
        self.model = model
        self.wave_front = WaveFront(gamma, v_max)
        self.V = {}

    def plan(self):
        V = self.wave_front.propagate_value(self.model)
        policy = {s: max(self.model.A, key=lambda a: self.action_value(s, a)) for s in self.model.S if
                  s not in self.model.objectives}
        return policy

    def action_value(self, state, action):
        next_state = self.model.T(state, action)
        return self.V.get(next_state, MININT)
