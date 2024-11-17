from test_environments.problem import Problem


class PlanningProblem(Problem):
    def __init__(self, environment):
        super.__init__(environment.initial_position, MoveOperator(environment, environment.movements))