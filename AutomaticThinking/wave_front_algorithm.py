from xmlrpc.client import MAXINT


class WaveFront:
    def __init__(self, gamma, v_max):
        self.gamma = gamma  # 0.95 ... 0.98
        self.v_max = v_max  # 1

    def propagate_value(self, model):
        V = {}
        wave_front = []

        for s in model.objectives:
            V[s] = self.v_max
            wave_front.append(s)

        while wave_front:
            s = wave_front.pop(0)

            for next_state in model.adjacents(s):
                v = V[s] * pow(self.gamma, model.distance(s, next_state))
                if v > V.get(next_state, MAXINT):
                    V[next_state] = v
                    wave_front.append(next_state)
        return V
