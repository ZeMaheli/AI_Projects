from problem.environment import Environment
from problem.planning_problem import PlanningProblem
from search.a_star_search import AStarSearch
from test_environments.parse_file import parse_maze_file

initial_position, obstacles, targets, size = parse_maze_file(
    "C:\\Users\\Lenovo\\Desktop\\AI_Projects\\AutomaticThinking\\test_environments\\amb2.txt")
# Print the results
print("Initial Position:", initial_position)
print("Obstacles:", obstacles)
print("Targets:", targets)
print("Size: ", size)

size_x, size_y = size

environment = Environment((size_x, size_y), initial_position, targets, obstacles)

# Define the planning problem
problem = PlanningProblem(environment, initial_position, environment.movements)

# Solve the problem using A* Search
solver = AStarSearch()
solution = solver.solve(problem)
print(solution)
# Display the solution
if solution:
    print("Path found:")
    environment.show_route(solution)
else:
    print("No path found.")
