def parse_maze_file(file_path):
    """
    Reads a maze file and extracts the agent's position, obstacles, and targets.

    Args:
        file_path (str): Path to the maze file.

    Returns:
        tuple: (initial_position, obstacles_positions, target_positions)
            - initial_position: Tuple (x, y) for the agent's position (@)
            - obstacles_positions: List of tuples [(x, y), ...] for obstacles (O)
            - target_positions: List of tuples [(x, y), ...] for targets (A)
    """
    initial_position = None
    obstacles_positions = []
    target_positions = []
    size = [0, 0]

    with open(file_path, 'r') as file:
        for y, line in enumerate(file):  # y represents the row index
            size[0] += 1
            for x, char in enumerate(line.strip()):  # x represents the column index
                size[1] += 1
                if char == '@':
                    print("@", end="")
                    initial_position = (x, y)
                elif char == 'O':
                    print("#", end="")
                    obstacles_positions.append((x, y))
                elif char == 'A':
                    print("A", end="")
                    target_positions.append((x, y))
                else:
                    print(" ", end="")
            print()
        size[1] = int(size[1] / size[0])

    return initial_position, obstacles_positions, target_positions, size
