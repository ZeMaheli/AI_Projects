from best_first_search import BestFirstSearch


class AStarSearch(BestFirstSearch):

    def f(self, node):
        return node.cost + self.problem.heuristic(node.state)
