from search.best_first_search import BestFirstSearch


class WeightedAStar(BestFirstSearch):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def f(self, node):
        return node.cost + (1 + self.epsilon) * self.problem.heuristic(node.state)
