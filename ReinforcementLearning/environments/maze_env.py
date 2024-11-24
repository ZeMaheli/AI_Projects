import random


class MazeEnv:
    def __init__(self, maze_size):
        self.maze_size = maze_size
        self.maze = self.generate_complex_maze()
        self.agent_position = (0, 0)  # Starting position (0, 0)
        self.goal_position = (self.maze_size[0] - 1, self.maze_size[1] - 1)  # Goal at bottom-right corner
        self.done = False

    def generate_complex_maze(self):
        width, height = self.maze_size
        """Generates a complex maze using recursive backtracking."""
        maze = [[1 for _ in range(width)] for _ in range(height)]
        stack = []
        start = (0, 0)
        stack.append(start)
        maze[start[1]][start[0]] = 0  # Mark the start as open

        def get_neighbors(x, y):
            """Get the unvisited neighboring cells."""
            neighbors = []
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # 4 possible moves (up, down, left, right)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                    neighbors.append((nx, ny))
            return neighbors

        while stack:
            x, y = stack[-1]
            neighbors = get_neighbors(x, y)
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Knock down the wall between (x, y) and (nx, ny)
                maze[ny][nx] = 0
                maze[(y + ny) // 2][(x + nx) // 2] = 0  # Remove wall between
                stack.append((nx, ny))
            else:
                stack.pop()

        # Ensure start and goal positions are open
        maze[0][0] = 0
        maze[height - 1][width - 1] = 0
        maze[height - 1][width - 2] = 0
        return maze

    def reset(self):
        self.agent_position = (0, 0)  # Reset agent position
        self.done = False
        return self.agent_position  # Return starting position

    def step(self, action):
        new_position = 'up'
        x, y = self.agent_position
        if action == 'up':
            new_position = (x, y - 1)
        elif action == 'down':
            new_position = (x, y + 1)
        elif action == 'left':
            new_position = (x - 1, y)
        elif action == 'right':
            new_position = (x + 1, y)

        # Ensure the new position is within bounds and not blocked
        if len(self.maze) > new_position[0] >= 0 == self.maze[new_position[0]][new_position[1]] and 0 <= new_position[
            1] < len(self.maze[0]):
            self.agent_position = new_position

        # Check if the agent has reached the goal
        reward = -1  # Default reward for each step
        self.done = self.is_goal_state(self.agent_position)
        if self.done:
            reward = 100  # Large reward for reaching the goal

        return self.agent_position, reward, self.done

    def is_goal_state(self, state):
        return state == self.goal_position

    def print_maze(self):
        """Prints the maze to the console."""
        for row in self.maze:
            print(' '.join(['#' if cell == 1 else '.' for cell in row]))
