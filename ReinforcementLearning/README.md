# Reinforcement Learning Library and Maze Application

## Overview

This library implements several Reinforcement Learning (RL) agents to solve maze navigation tasks. It includes agents
such as **Dyna-Q**, **SARSA**, and **QME**, each with unique learning strategies. The library tracks metrics like
rewards, steps per episode, training time, and success rates for comprehensive evaluation.

---

## Table of Contents

1. [Features](#features)
2. [Agents Overview](#agents-overview)
3. [Metrics](#metrics)
4. [Code Examples](#code-examples)
5. [Usage](#usage)

---

## Features

- **Reinforcement Learning Agents**:
    - Dyna-Q
    - SARSA
    - QME
- Tracks and visualizes:
    - Cumulative Rewards
    - Steps per Episode
    - Training Time
    - Success Rate (% of episodes reaching the goal)
- Customizable maze environments.

---

## Agents Overview

### 1. **Dyna-Q**

- Combines **Q-learning** with a simulated model to accelerate learning.
- Uses **e-greedy action selection** and **sparse learning memory**.

### 2. **SARSA**

- An on-policy learning method that updates Q-values using actual trajectories.
- Implements **e-greedy** for exploration.

### 3. **QME**

- A hybrid method focused on improving exploration efficiency.
- Utilizes **sparse learning memory** for learning stability.

---

## Metrics

- **Cumulative Reward**: Measures the total reward accumulated during each episode.
- **Steps per Episode**: Tracks the number of steps taken to complete the maze in each episode.
- **Training Time**: Total time to train the agent over the specified episodes.
- **Success Rate**: Percentage of episodes where the agent successfully reaches the maze's goal.

---

## Code Examples

### Training Agents with Metrics

```python
dyna_q_rewards, dyna_q_steps, dyna_q_time, dyna_q_success = train_with_metrics(dyna_q_agent, env, episodes)

sarsa_rewards, sarsa_steps, sarsa_time, sarsa_success = train_with_metrics(sarsa_agent, env, episodes)

qme_rewards, qme_steps, qme_time, qme_success = train_with_metrics(qme_agent, env, episodes)
```

### Plotting Comparison of Agents

```python
plot_comparison(dyna_q_rewards, sarsa_rewards, qme_rewards,
                dyna_q_time, sarsa_time, qme_time,
                dyna_q_steps, sarsa_steps, qme_steps,
                dyna_q_success, sarsa_success, qme_success)
```

## Usage

### Customizing the Maze

```python
maze_size = (10, 10)
maze_size_x, maze_size_y = maze_size
env = MazeEnv(maze_size)
```

### Training an Agent

```python
agent = DynaQAgent(**parameters **)

rewards, steps, time, success_rate = train_with_metrics(agent, environment, episodes)
```
