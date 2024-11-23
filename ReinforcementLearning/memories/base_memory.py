from abc import abstractmethod, ABC


class LearningMemory(ABC):
    @abstractmethod
    def update(self, state, action, q):
        pass

    @abstractmethod
    def Q(self, state, action):
        pass