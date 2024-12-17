class MoveOperator():
    def __init__(self, environment, movement):
        self.environment = environment
        self.movement = movement

    def apply(self, state):
        return self.environment.simulate_action(state, self.movement)

    def cost(self, state, successor_state):
        return self.environment.distance(state, successor_state)