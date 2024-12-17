import math


class Environment:
    def __init__(self, dimension, initial_position, target_positions, obstacles_positions):
        self.x_max, self.y_max = dimension
        self.initial_position = initial_position
        self.target_positions = target_positions
        self.obstacles_positions = obstacles_positions
        self.positions = [(x, y) for x in range(self.x_max) for y in range(self.y_max) if
                          (x, y) not in self.obstacles_positions]

        self.movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def simulate_action(self, position, movement):
        (x, y) = position
        (dx, dy) = movement
        new_position = (x + dx, y + dy)

        if self.valid_position(new_position):
            return new_position

    def valid_position(self, position):
        (x, y) = position

        return 0 <= x < self.x_max and 0 <= y < self.y_max and (x, y) not in self.obstacles_positions

    def distance(self, first_position, second_position):
        return math.dist(first_position, second_position)

    def show_route(self, route):
        print()
        for y in range(self.y_max):
            for x in range(self.x_max):
                if (x, y) in route:
                    print('.', end="")
                elif (x, y) in self.target_positions:
                    print('A', end="")
                elif (x, y) in self.obstacles_positions:
                    print('#', end="")
                else:
                    print(" ", end="")
            print()
        print()
