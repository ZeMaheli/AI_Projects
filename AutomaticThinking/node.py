class Node:
    def __init__(self, state, operator=None, predecessor=None, cost=0):
        self.state = state
        self.operator = operator
        self.predecessor = predecessor
        self.cost = cost

        if predecessor:
            self.depth = self.predecessor.depth + 1
        else:
            self.depth = 0
