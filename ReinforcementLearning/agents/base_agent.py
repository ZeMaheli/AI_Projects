from abc import abstractmethod, ABC

from memories.base_memory import LearningMemory
from policies.base_policy import ActionSelection


class ReinforcementLearn(ABC):
    """
    Abstract base class for reinforcement learning agents.

    Attributes:
        learn_memory (LearningMemory): The memory to store Q-values or experiences.
        action_select (ActionSelection): The policy for action selection.
        alpha (float): The learning rate (default: 0.1).
        gamma (float): The discount factor (default: 0.9).
    """

    def __init__(self, learn_memory: LearningMemory, action_select: ActionSelection, alpha=0.1, gamma=0.9):
        """
        Initializes the reinforcement learning agent.

        Args:
            learn_memory (LearningMemory): Memory to store Q-values.
            action_select (ActionSelection): Policy for selecting actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.learn_memory = learn_memory
        self.action_select = action_select
        self.alpha = alpha
        self.gama = gamma

    @abstractmethod
    def learn(self, state, action, reward, next_state, next_action=None):
        """
        Abstract method for updating Q-values or policy.

        Args:
            state: The current state.
            action: The action taken in the current state.
            reward: The reward received for the action.
            next_state: The next state resulting from the action.
            next_action: The next action (for SARSA-like algorithms).
        """
        pass
