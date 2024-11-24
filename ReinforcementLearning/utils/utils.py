import time

import matplotlib.pyplot as plt


def train_with_metrics(agent, env, episodes):
    total_rewards = []
    total_steps = []
    success_count = 0  # Track successful episodes
    start_time = time.time()  # Start time for measuring the training duration

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = 0
        steps = 0

        while not done:
            action = agent.action_select.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            episode_rewards += reward
            steps += 1
            state = next_state

        # Check if the goal was reached at the end of the episode
        if env.is_goal_state(state):
            success_count += 1

        total_rewards.append(episode_rewards)
        total_steps.append(steps)

    total_time = time.time() - start_time  # Time taken for the whole training process
    success_rate = (success_count / episodes) * 100  # Calculate success rate as a percentage
    return total_rewards, total_steps, total_time, success_rate


def plot_comparison(dyna_q_rewards, sarsa_rewards, qme_rewards,
                    dyna_q_time, sarsa_time, qme_time,
                    dyna_q_steps, sarsa_steps, qme_steps,
                    dyna_q_success, sarsa_success, qme_success):
    episodes = range(1, len(dyna_q_rewards) + 1)

    # Plot cumulative rewards
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.plot(episodes, dyna_q_rewards, label="DynaQ", color='b')
    plt.plot(episodes, sarsa_rewards, label="Sarsa", color='g')
    plt.plot(episodes, qme_rewards, label="QME", color='r')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Comparison")
    plt.legend()

    # Plot total steps
    plt.subplot(1, 3, 2)
    plt.plot(episodes, dyna_q_steps, label="DynaQ", color='b')
    plt.plot(episodes, sarsa_steps, label="Sarsa", color='g')
    plt.plot(episodes, qme_steps, label="QME", color='r')
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Steps per Episode Comparison")
    plt.legend()

    # Bar chart for training time
    plt.subplot(1, 3, 3)
    agents = ['DynaQ', 'Sarsa', 'QME']
    times = [dyna_q_time, sarsa_time, qme_time]
    success_rates = [dyna_q_success, sarsa_success, qme_success]
    width = 0.4  # Bar width
    x = range(len(agents))

    # Plot training times
    plt.bar(x, times, width=width, label="Training Time", color=['b', 'g', 'r'])

    # Overlay success rates
    plt.bar([i + width for i in x], success_rates, width=width, label="Success Rate (%)", color=['c', 'lime', 'pink'])

    plt.xticks([i + width / 2 for i in x], agents)
    plt.ylabel("Time (s) / Success Rate (%)")
    plt.title("Training Time and Success Rate Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()
