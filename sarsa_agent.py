import numpy as np
import gymnasium as gym
from collections import defaultdict
import time

class SarsaCliffWalkingAgent:
    def __init__(self, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = gym.make("CliffWalking-v0")
        self.episodes = episodes
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.convergence = []

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

            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.policy(next_state) if not done else None

                target = reward + (self.gamma * self.Q[next_state][next_action] if not done else 0)
                self.Q[state][action] += self.alpha * (target - self.Q[state][action])

                state, action = next_state, next_action

            avg_q_value = np.mean([np.max(q) for q in self.Q.values()])
            self.convergence.append(avg_q_value)

            if progress_callback and episode_num % 500 == 0:
                progress_callback(episode_num / self.episodes)

        return self.Q, self.convergence

    def simulate_game(self):
        """Simulates a game of Cliff Walking step by step."""
        state, _ = self.env.reset()
        game_log = ["Starting Cliff Walking simulation...\n"]
        grid = np.full((4, 12), '⬜')  # Initialize grid

        while True:
            action = self.policy(state)
            next_state, reward, done, _, _ = self.env.step(action)

            row, col = divmod(state, 12)
            grid[row, col] = "🤖"  # Mark agent's position

            log_text = f"🏃 Agent moved to **({row}, {col})** | Action: {action} | Reward: {reward}\n"
            game_log.append(log_text)

            if done:
                game_log.append("🏁 **Simulation Complete!**")
                break

            state = next_state

        return "\n".join(game_log)
