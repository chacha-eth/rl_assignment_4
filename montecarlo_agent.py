import numpy as np
import gymnasium as gym
from collections import defaultdict
import time

class MonteCarloBlackjackAgent:
    def __init__(self, episodes=10000, gamma=1.0):
        self.env = gym.make("Blackjack-v1")
        self.episodes = episodes
        self.gamma = gamma
        self.returns = defaultdict(list)
        self.V = defaultdict(float)
        self.convergence = []

    def policy(self, state):
        """Fixed policy: Stick if sum >= 18, otherwise hit."""
        return 0 if state[0] >= 18 else 1  

    def train(self, progress_callback=None):
        """Monte Carlo policy evaluation using first-visit method."""
        for episode_num in range(1, self.episodes + 1):
            state, _ = self.env.reset()
            episode = []
            done = False

            while not done:
                action = self.policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                episode.append((state, reward))
                state = next_state

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

            if progress_callback and episode_num % 1000 == 0:
                progress_callback(episode_num / self.episodes)

        return self.V, self.convergence

    def simulate_game(self):
        """Simulates a Blackjack game step by step."""
        state, _ = self.env.reset()
        game_log = ["Starting new game...\n"]

        while True:
            action = self.policy(state)
            next_state, reward, done, _, _ = self.env.step(action)
            action_text = "Stick ✋" if action == 0 else "Hit 🎯"

            game_log.append(f"👤 Player Sum: **{state[0]}** | 🃏 Dealer Card: **{state[1]}**\n")
            game_log.append(f"🎲 Action: **{action_text}** | 🏆 Reward: **{reward}**\n")

            state = next_state
            if done:
                game_log.append("🎉 **Game Over!** 🎉")
                break

        return "\n".join(game_log)
