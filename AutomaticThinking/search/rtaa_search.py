from search.best_first_search import BestFirstSearch


class RTAASearch(BestFirstSearch):
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.heuristic = {}

    def f(self, node):
        return node.cost + self.heuristic.get(node.state, self.problem.heuristic(node.state))

    def solve_rtaa(self, problem):
        solution = super().solve(problem, self.max_depth, True)
        if solution:
            u = solution[-1]
            for v in self.explored.values():
                self.heuristic[v.state] = self.f(u) - v.cost
