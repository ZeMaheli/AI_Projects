from abc import ABC, abstractmethod

from memories.base_memory import LearningMemory


class ActionSelection(ABC):
    """
    Abstract base class for action selection policies.

    Attributes:
        learn_mem (LearningMemory): Memory to access Q-values for decisions.
    """

    def __init__(self, learn_mem: LearningMemory):
        """
        Initializes the action selection policy.

        Args:
            learn_mem (LearningMemory): Memory to access Q-values.
        """
        self.learn_mem = learn_mem

    @abstractmethod
    def select_action(self, state):
        """
        Selects an action based on the policy.

        Args:
            state: The current state.

        Returns:
            action: The action selected.
        """
        pass

    @abstractmethod
    def max_action(self, state):
        """
        Finds the action with the maximum Q-value for the given state.

        Args:
            state: The current state.

        Returns:
            action: The action with the maximum Q-value.
        """
        pass
