class WorldModel:
    def __init__(self, environment):
        self.environment = environment

    @property
    def states(self):
        return self.environment.positions

    @property
    def actions(self):
        return self.environment.movements

    @property
    def objectives(self):
        return self.environment.target_positions

    def transition_successor_state(self, state, action):
        return self.environment.simulate_action(state, action)

    def distance(self, state1, state2):
        return self.environment.distance(state1, state2)

    def adjacents(self, state):
        adjacents = []
        for action in self.actions:
            next_state = self.transition_successor_state(state, action)
            if next_state:
                adjacents.append(next_state)
        return adjacents