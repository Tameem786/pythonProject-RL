import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Define a flexible PolicyNetwork class that supports varying number of layers
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super(PolicyNetwork, self).__init__()
        # Create the list of layers starting with the first hidden layer
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]

        # Add more hidden layers based on num_layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer that maps the hidden layer to action probabilities
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.model = nn.Sequential(*layers) # Combine layers into a sequential model

    def forward(self, x):
        action_probs = torch.softmax(self.model(x), dim=-1)
        return action_probs

# REINFORCE algorithm implementation
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, hidden_dim=128, num_layers=2):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32) # Convert state to tensor
        action_probs = self.policy(state) # Get action probabilities from policy network
        dist = Categorical(action_probs) # Create a distribution based on probabilities
        action = dist.sample() # Sample an action from the distribution
        self.log_probs.append(dist.log_prob(action)) # Save the log probability of this action
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []

        # Compute discounted rewards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy loss
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Update policy
        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss)
        policy_loss.backward()
        self.optimizer.step()

        # Clear log_probs and rewards
        self.log_probs = []
        self.rewards = []


def train_agent(env, agent, episodes=1000, max_steps=1000):
    rewards = []

    for episode in range(episodes):
        state = env.reset()[0] # Reset the environment for each episode
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        agent.update_policy()
        rewards.append(total_reward)
        print(f'Episode: {episode}, Total Reward: {total_reward}')
    return rewards # Return the list of rewards for plotting

# Experiment to test performance with varying numbers of layers in the policy network
def experiment_varying_layer(layers, env_name='CartPole-v1', episodes=100, runs_per_layer=10, hidden_dim=128, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}
    # Loop through different numbers of layers
    for num_layers in layers:
        avg_rewards = np.zeros(episodes)
        # Run multiple runs for each layer configuration to average out randomness
        for run in range(runs_per_layer):
            agent = REINFORCE(state_dim, action_dim, lr=lr, hidden_dim=hidden_dim, num_layers=num_layers)
            rewards = train_agent(env, agent, episodes)
            avg_rewards += np.array(rewards)

        avg_rewards /= runs_per_layer
        results[num_layers] = avg_rewards
        print(f"Completed for {num_layers} layers")

    return results

# Experiment for varying learning rates
def experiment_varying_lr(lrs, env_name='CartPole-v1', episodes=1000, runs_per_lr=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}
    # Loop through different learning rates
    for lr in lrs:
        avg_rewards = np.zeros(episodes)

        for run in range(runs_per_lr):
            agent = REINFORCE(state_dim, action_dim, lr=lr)
            rewards = train_agent(env, agent, episodes)
            avg_rewards += np.array(rewards)

        avg_rewards /= runs_per_lr
        results[lr] = avg_rewards
        print(f"Completed for lr = {lr}")

    return results

# Experiment for varying gamma (discount factor)
def experiment_varying_gamma(gammas, env_name='CartPole-v1', episodes=1000, runs_per_gamma=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {}

    for gamma in gammas:
        avg_rewards = np.zeros(episodes)

        for run in range(runs_per_gamma):
            agent = REINFORCE(state_dim, action_dim, lr=0.001, gamma=gamma)
            rewards = train_agent(env, agent, episodes)
            avg_rewards += np.array(rewards)

        avg_rewards /= runs_per_gamma
        results[gamma] = avg_rewards
        print(f"Completed for lr = {gamma}")

    return results

def plot_results(results, title="Reward vs Episode"):
    plt.figure(figsize=(10, 6))
    # Plot the rewards for each configuration
    for num_layers, rewards in results.items():
        plt.plot(rewards, label=f"layers={num_layers}")

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # Vary learning rates
    gammas = [0.1, 0.5, 0.95, 0.99]  # Vary learning rates
    layers = [1, 2, 3, 4, 5]
    # results = experiment_varying_lr(lrs, episodes=1000, runs_per_lr=10)
    # results = experiment_varying_gamma(gammas, episodes=1000, runs_per_gamma=10)
    results = experiment_varying_layer(layers, episodes=1000, runs_per_layer=10)

    # plot_results(results, title="Effect of Learning Rate on CartPole Performance")
    # plot_results(results, title="Effect of Gamma on CartPole Performance")
    plot_results(results, title="Effect of Number of Layers on CartPole Performance")
