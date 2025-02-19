import numpy as np
import gymnasium as gym
from collections import defaultdict

class SarsaCliffWalkingAgent:
    def __init__(self, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = gym.make("CliffWalking-v0")
        self.episodes = episodes
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.convergence = []
        self.cumulative_rewards = []  # Track cumulative rewards per episode

    def policy(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.Q[state])

    def train(self, progress_callback=None):
        """SARSA (on-policy TD control)"""
        for episode_num in range(1, self.episodes + 1):
            state, _ = self.env.reset()
            action = self.policy(state)
            done = False
            total_reward = 0  # Track cumulative reward

            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.policy(next_state) if not done else None

                target = reward + (self.gamma * self.Q[next_state][next_action] if not done else 0)
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])

                state, action = next_state, next_action
                total_reward += reward  # Accumulate total reward

            avg_q_value = np.mean([np.max(q) for q in self.Q.values()])
            self.convergence.append(avg_q_value)
            self.cumulative_rewards.append(total_reward)  # Store total reward per episode

            if progress_callback and episode_num % 500 == 0:
                progress_callback(episode_num / self.episodes)

        return self.Q, self.convergence, self.cumulative_rewards

    def get_policy(self):
        """Extracts the best action for each state in the Cliff Walking grid."""
        policy_grid = np.full((4, 12), " ")
        actions = {0: "↑", 1: "→", 2: "↓", 3: "←"}  # Mapping actions to arrows

        for state in range(48):  # 4x12 grid
            best_action = np.argmax(self.Q[state])  # Best action per state
            row, col = divmod(state, 12)
            policy_grid[row, col] = actions[best_action]

        return policy_grid
