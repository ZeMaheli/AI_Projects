import math
from abc import abstractmethod

from node import Node
from priority_border import PriorityBorder


class BestFirstSearch:
    def __init__(self):
        self.border = PriorityBorder()
        self.explored = {}
        self.problem = None

    def solve(self, problem, max_depth=math.inf, parcial=False):
        self.problem = problem
        self.cleanup_memory()
        node = Node(problem.initial_state)
        self.memorize(node)

        while not self.border.empty():
            node = self.border.remove()

            if problem.objective(node.state):
                return self.generate_solution(node)
            elif node.depth < max_depth:
                for successor_node in self.expand(problem, node):
                    self.memorize(successor_node)
            elif parcial:
                return self.generate_solution(node)

        return None

    def cleanup_memory(self):
        self.border.cleanup()
        self.explored.clear()

    def memorize(self, node):
        if node.state not in self.explored or node.cost < self.explored[node.state].cost:
            self.border.insert(node, self.f(node))
            self.explored[node.state] = node

    def expand(self, problem, node):
        successors = []
        state = node.state
        for operator in problem.operators:
            successor_state = operator.apply(state)
            if successor_state:
                cost = node.cost + operator.cost(state, successor_state)
                successor_node = Node(successor_state, operator, node, cost)
                successors.append(successor_node)

        return successors

    def generate_solution(self, final_node):
        solution = []
        node = final_node

        while node:
            solution.insert(0, node)
            node = node.predecessor

        return solution

    @abstractmethod
    def f(self, node):
        """Abstract method to be implemented by subclasses."""
        pass
