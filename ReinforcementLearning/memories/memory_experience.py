from random import sample


class MemoryExperience:
    def __init__(self, max_dimensions):
        self.max_dimensions = max_dimensions
        self.memory = []

    def update(self, experience):
        if len(self.memory) >= self.max_dimensions:
            self.memory.pop(0)

        self.memory.append(experience)

    def showcase(self, number):
        sample_size = min(number, len(self.memory))
        return sample(self.memory, sample_size)
