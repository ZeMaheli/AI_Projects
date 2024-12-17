from search.best_first_search import BestFirstSearch


class GreedySearch(BestFirstSearch):

    def f(self, node):
        return self.problem.heuristic(node.state)