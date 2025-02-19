import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from montecarlo_agent import MonteCarloBlackjackAgent
from sarsa_agent import SarsaCliffWalkingAgent

# Ensure the figures directory exists
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Function to plot and save heatmap of Monte Carlo value function
def plot_value_function(V, title, filename):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    V_matrix = np.full((len(player_sums), len(dealer_cards)), np.nan)

    for state, value in V.items():
        player_sum, dealer_card, usable_ace = state
        if 12 <= player_sum <= 21 and 1 <= dealer_card <= 10 and not usable_ace:
            V_matrix[player_sum - 12, dealer_card - 1] = value

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(V_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=dealer_cards, yticklabels=player_sums, ax=ax)
    ax.set_xlabel("Dealer's Visible Card")
    ax.set_ylabel("Player Sum")
    ax.set_title(title)

    # Save figure
    fig_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(fig_path)
    plt.close()

    st.image(fig_path, caption=title)

# Function to plot and save heatmap of SARSA Q-values
def plot_q_values(Q, title, filename):
    q_values_matrix = np.zeros((4, 12))  # 4x12 grid
    for state in Q:
        q_values_matrix[state // 12, state % 12] = np.max(Q[state])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(q_values_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_xlabel("Grid Columns")
    ax.set_ylabel("Grid Rows")
    ax.set_title(title)

    # Save figure
    fig_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(fig_path)
    plt.close()

    st.image(fig_path, caption=title)

# Function to plot and save convergence graph
def plot_convergence(convergence, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(convergence) + 1), convergence, label="Value Estimate Over Time")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average V(s) or Q(s, a)")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    # Save figure
    fig_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(fig_path)
    plt.close()

    st.image(fig_path, caption=title)

# Function to plot and save cumulative rewards per episode
def plot_cumulative_rewards(rewards, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(rewards) + 1), rewards, label="Cumulative Reward", color="orange")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid()

    # Save figure
    fig_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(fig_path)
    plt.close()

    st.image(fig_path, caption=title)

# Function to visualize and save the learned policy for Cliff Walking
def plot_policy(policy_grid, title, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(np.zeros((4, 12)), cbar=False, linewidths=0.5, linecolor="gray")
    
    for i in range(4):
        for j in range(12):
            plt.text(j + 0.5, i + 0.5, policy_grid[i, j], ha="center", va="center", fontsize=14)

    plt.xticks(np.arange(12) + 0.5, np.arange(12))
    plt.yticks(np.arange(4) + 0.5, np.arange(4))
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    plt.title(title)

    # Save figure
    fig_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(fig_path)
    plt.close()

    st.image(fig_path, caption=title)

# UI setup
st.sidebar.header("Select Algorithm")

# Select environment (Monte Carlo for Blackjack, SARSA for Cliff Walking)
env_choice = st.sidebar.selectbox("Choose an Algorithm & Environment", ["Monte Carlo - Blackjack", "SARSA - Cliff Walking"])
num_episodes = st.sidebar.slider("Number of Episodes", 1000, 20000, 10000, step=1000)

# Set main title dynamically
st.title(f"Reinforcement Learning: {env_choice}")

# Placeholder for both graphs
graphs_placeholder = st.container()

# Train model only when the button is clicked
if st.sidebar.button("Train Agent"):
    with graphs_placeholder:
        if env_choice == "Monte Carlo - Blackjack":
            st.write(f"Training Monte Carlo agent for Blackjack ({num_episodes} episodes)...")
            agent = MonteCarloBlackjackAgent(episodes=num_episodes)
            progress_bar = st.progress(0)
            V, convergence, rewards = agent.train(progress_bar.progress)

            st.subheader("Monte Carlo Algorithm - Blackjack Results")
            plot_value_function(V, "Monte Carlo State-Value Function for Blackjack", "monte_carlo_blackjack_value_function.png")
            plot_convergence(convergence, "Monte Carlo Convergence Plot for Blackjack", "monte_carlo_blackjack_convergence.png")
            plot_cumulative_rewards(rewards, "Monte Carlo Cumulative Rewards per Episode", "monte_carlo_blackjack_rewards.png")

        elif env_choice == "SARSA - Cliff Walking":
            st.write(f"Training SARSA agent for Cliff Walking ({num_episodes} episodes)...")
            agent = SarsaCliffWalkingAgent(episodes=num_episodes)
            progress_bar = st.progress(0)
            Q, convergence, rewards = agent.train(progress_bar.progress)

            policy_grid = agent.get_policy()  # Extract learned policy

            st.subheader("SARSA Algorithm - Cliff Walking Results")
            plot_q_values(Q, "SARSA Q-Value Heatmap for Cliff Walking", "sarsa_cliff_walking_q_values.png")
            plot_convergence(convergence, "SARSA Convergence Plot for Cliff Walking", "sarsa_cliff_walking_convergence.png")
            plot_cumulative_rewards(rewards, "SARSA Cumulative Rewards per Episode", "sarsa_cliff_walking_rewards.png")
            # plot_policy(policy_grid, "SARSA Learned Policy Visualization", "sarsa_cliff_walking_policy.png")
