import numpy as np
import gymnasium as gym
from collections import defaultdict

class MonteCarloBlackjackAgent:
    def __init__(self, episodes=10000, gamma=1.0):
        self.env = gym.make("Blackjack-v1")
        self.episodes = episodes
        self.gamma = gamma
        self.returns = defaultdict(list)
        self.V = defaultdict(float)
        self.convergence = []
        self.cumulative_rewards = []  # Track cumulative rewards per episode

    def policy(self, state):
        """Fixed policy: Stick if sum >= 18, otherwise hit."""
        return 0 if state[0] >= 18 else 1  

    def train(self, progress_callback=None):
        """Monte Carlo policy evaluation using first-visit method."""
        for episode_num in range(1, self.episodes + 1):
            state, _ = self.env.reset()
            episode = []
            done = False
            total_reward = 0  # Track cumulative reward

            while not done:
                action = self.policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode.append((state, reward))
                state = next_state
                total_reward += reward  # Accumulate total reward

            G = 0
            visited_states = set()
            for t in reversed(range(len(episode))):
                state, reward = episode[t]
                G = self.gamma * G + reward
                if state not in visited_states:
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
                    visited_states.add(state)

            self.convergence.append(np.mean(list(self.V.values())))
            self.cumulative_rewards.append(total_reward)  # Store total reward per episode

            if progress_callback and episode_num % 1000 == 0:
                progress_callback(episode_num / self.episodes)

        return self.V, self.convergence, self.cumulative_rewards
