import random


class TRModel:
    def __init__(self):
        self.T = {}
        self.R = {}

    def update(self, state, action, reward, next_state):
        self.T[(state, action)] = next_state
        self.R[(state, action)] = reward

    def showcase(self):
        """
        Samples a random transition (state, action, reward, next_state) from the model.
        """
        if not self.T:
            raise ValueError("No transitions available in the model.")

        # Choose a random key from self.T
        state_action = random.choice(list(self.T.keys()))

        # Extract values from the dictionaries
        next_state = self.T[state_action]
        reward = self.R[state_action]

        # Unpack the (state, action) key
        state, action = state_action

        return state, action, reward, next_state

