# Import necessary libraries for the environment, neural network, optimization, tensorboard, and other utilities
# gymnasium: used to create the reinforcement learning environment
# torch: used for building and training the neural network
# random and numpy: for sampling and manipulating data
# deque: for implementing the replay buffer
# SummaryWriter: for logging data to TensorBoard

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from collections import deque

# Initialize SummaryWriter to log metrics for TensorBoard visualization
writer = SummaryWriter()

# Define a Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        # Initialize the neural network layers with input size equal to the state size and output size equal to the number of actions
        super(DQN, self).__init__()
        # First fully connected layer with 64 units
        self.fc1 = nn.Linear(state_size, 64)
        # Second fully connected layer with 64 units
        self.fc2 = nn.Linear(64, 64)
        # Output layer that predicts Q-values for each action
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # Apply ReLU activation function on first and second layers and pass the result to the output layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output raw Q-values for each action
        return self.fc3(x)

# Define a Replay Buffer class to store and sample experiences for training
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        # Initialize the replay buffer with a maximum size
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        # Store the experience tuple (state, action, reward, next_state, done) in the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a batch of experiences randomly from the buffer
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)

# Define a DQN Agent class that interacts with the environment and learns from experiences
class DNQAgent:
    def __init__(self, state_size=6, action_size=3, hidden_size=64, learning_rate=0.001, gamma=0.99, buffer_size=10000, batch_size=64):
        # Initialize agent parameters including state/action size, learning rate, discount factor (gamma), and memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_size = action_size
        # Initialize the Q-network and target network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        # Copy the weights from the Q-network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        # Optimizer for updating the Q-network weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Replay buffer to store past experiences
        self.memory = ReplayBuffer(buffer_size)

    def step(self, state, action, reward, next_state, done, epoch):
        # Store the experience in the replay buffer
        self.memory.push(state, action, reward, next_state, done)
        # If there are enough experiences in the buffer, sample and update the Q-network
        if len(self.memory) > self.batch_size:
            self.update_model(epoch)

    def act(self, state, eps=0):
        # Choose an action using an epsilon-greedy policy
        if random.random() > eps:
            # Select the action with the highest Q-value (exploitation)
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Select a random action (exploration)
            return random.choice(np.arange(self.action_size))

    def update_model(self, epoch):
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Convert the experience batch to PyTorch tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(np.array(dones)).float()
        # Compute Q-values for the current states and actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # Compute Q-values for the next states
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        # Compute the expected Q-values using the Bellman equation
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        # Compute the loss between the Q-values and expected Q-values
        loss = nn.MSELoss()(q_values, expected_q_values)
        # Perform backpropagation to minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update the target network to have the same weights as the Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())

# Define a training loop to train the agent on the environment
def train(agent, env, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.99, target_update=10):
    scores = [] # List to store the score per episode
    scores_window = deque(maxlen=100) # Rolling window for the last 100 scores
    eps = eps_start # Initialize epsilon for the epsilon-greedy policy

    for i_episode in range(1, n_episodes+1):
        # Reset the environment at the start of each episode
        state, _ = env.reset()
        score = 0

        while True:
            # Select an action using the epsilon-greedy policy
            action = agent.act(state, eps)
            # Take a step in the environment and observe the next state, reward, and done signal
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store the experience in the replay buffer and update the agent's model
            agent.step(state, action, reward, next_state, done, i_episode)
            state = next_state
            score += reward

            if done:
                break
        # Append the episode score to the window and the total scores list
        scores_window.append(score)
        scores.append(score)
        # Update epsilon for the next episode (decay for less exploration over time)
        eps = max(eps_end, eps_decay*eps)

        writer.add_scalar('Score/Episode', score, i_episode)
        # Print the average score over the last 100 episodes
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")
        # Update the target network periodically
        if i_episode % target_update == 0:
            agent.update_target_network()
        # Save the model every 100 episodes and print the current average score
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # Stop training early if the average score is good enough
        if i_episode % 100 == 0 and np.mean(scores_window) >= -100:
            break
    return scores

def load_model(agent, load_path='model.pth'):
    checkpoint = torch.load(load_path)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimized_state_dict'])
    print(f"Model loaded from episode {checkpoint['episode']} with an average score of {checkpoint['avg_score']:.2f}")

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DNQAgent(state_size, action_size)
scores = train(agent, env)
# Save the trained model
torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'optimized_state_dict': agent.optimizer.state_dict()
}, 'model.pth')

writer.flush()

writer.close()

env.close()

# Function to load a trained model from a saved checkpoint
def load_trained_model(agent, load_path='model.pth'):
    checkpoint = torch.load(load_path)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimized_state_dict'])
    print(f"Model loaded from {load_path}")


env = gym.make('Acrobot-v1', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DNQAgent(state_size, action_size)
load_trained_model(agent, 'model.pth')

state, _ = env.reset()
done = False
while not done:
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
