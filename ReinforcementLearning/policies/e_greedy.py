import random

from memories.base_memory import LearningMemory
from policies.base_policy import ActionSelection


class EGreedy(ActionSelection):
    """
    Implements the epsilon-greedy policy for action selection.

    Inherits:
        ActionSelection

    Attributes:
        actions (list): List of all possible actions.
        epsilon (float): The probability of selecting a random action.
    """

    def __init__(self, learn_mem: LearningMemory, actions, epsilon):
        """
        Initializes the epsilon-greedy policy.

        Args:
            learn_mem (LearningMemory): Memory to access Q-values.
            actions (list): List of all possible actions.
            epsilon (float): Exploration probability.
        """
        self.learn_mem = learn_mem
        super().__init__(self.learn_mem)
        self.actions = actions
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() > self.epsilon:
            return self.advantage(state)
        else:
            return self.explore()

    def max_action(self, state):
        random.shuffle(self.actions)
        return max(self.actions, key=lambda a: self.learn_mem.Q(state, a))

    def advantage(self, state):
        return self.max_action(state)

    def explore(self):
        return random.choice(self.actions)
