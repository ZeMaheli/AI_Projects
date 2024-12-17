from problem.move_operator import MoveOperator
from problem.problem import Problem


class PlanningProblem(Problem):
    def __init__(self, environment, initial_state, operators):
        super().__init__(environment.initial_position,
                       [MoveOperator(environment, movement) for movement in environment.movements])

        self.environment = environment
        self.final_state = environment.target_positions[0]

    def objective(self, state):
        return state == self.final_state

    def heuristic(self, state):
        return self.environment.distance(state, self.final_state)
