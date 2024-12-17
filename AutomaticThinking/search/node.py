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

    def __lt__(self, other):
        return self.cost < other.cost  # Compare nodes based on their cost
