from best_first_search import BestFirstSearch


class UniformCostSearch(BestFirstSearch):

    def f(self, node):
        return node.cost