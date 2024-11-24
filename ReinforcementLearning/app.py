from agents.dyna_q import DynaQ
from agents.qme import QME
from agents.sarsa_learning import SarsaLearning
from environments.maze_env import MazeEnv
from memories.sparse_learning_memory import SparseLearningMemory
from policies.e_greedy import EGreedy
from utils.utils import train_with_metrics, plot_comparison


def main():
    # Create environment
    maze_size = (10, 10)
    maze_size_x, maze_size_y = maze_size
    env = MazeEnv(maze_size)
    # Environment created

    actions = ['up', 'down', 'left', 'right']
    alpha = 0.1
    gamma = 0.9
    no_simulations = 5
    epsilon = 0.1

    env.print_maze()

    # Initialize agents
    dyna_q_agent = DynaQ(SparseLearningMemory(),
                         EGreedy(SparseLearningMemory(), actions, epsilon),
                         alpha, gamma, no_simulations)

    sarsa_agent = SarsaLearning(SparseLearningMemory(),
                                EGreedy(SparseLearningMemory(), actions, epsilon), alpha, gamma)

    qme_agent = QME(SparseLearningMemory(),
                    EGreedy(SparseLearningMemory(), actions, epsilon), alpha, gamma, no_simulations,
                    maze_size_x * maze_size_y)

    # Train the agents and get their performance metrics
    print("Training DynaQ Agent")
    dyna_q_rewards, dyna_q_steps, dyna_q_time, dyna_q_success_rate = train_with_metrics(dyna_q_agent, env, episodes=100)

    print("Training Sarsa Agent")
    sarsa_rewards, sarsa_steps, sarsa_time, sarsa_time_success_rate = train_with_metrics(sarsa_agent, env, episodes=100)

    print("Training QME Agent")
    qme_rewards, qme_steps, qme_time, qme_success_rate = train_with_metrics(qme_agent, env, episodes=100)

    # Plot the results
    plot_comparison(dyna_q_rewards, sarsa_rewards, qme_rewards, dyna_q_time, sarsa_time, qme_time,
                    dyna_q_steps, sarsa_steps, qme_steps, dyna_q_success_rate, sarsa_time_success_rate,
                    qme_success_rate)


if __name__ == "__main__":
    main()
