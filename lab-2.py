import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import random
import numpy as np
from collections import deque

writer = SummaryWriter()

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 0.001  # Learning rate
EPSILON_START = 1.0  # Initial epsilon for exploration
EPSILON_END = 0.1  # Minimum epsilon
EPSILON_DECAY = 0.999  # Decay rate for epsilon
MEMORY_SIZE = 10000  # Replay memory size
BATCH_SIZE = 128  # Batch size for training
TARGET_UPDATE = 5  # Target network update frequency

# Create the environment
env = gym.make('MountainCar-v0')


# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Create the agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get input and output dimensions
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        # Networks
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # Update target network initially
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        # Q-value of the current state
        q_values = self.q_network(states).gather(1, actions)

        # Q-value of the next state using the target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        # Loss function and optimization step
        loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# Main training loop
agent = DQNAgent(env)
num_episodes = 5000

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # MountainCar gives a -1 reward for every time step
        # Encourage the agent to reach the goal by flipping the reward
        reward = 0 if not done else 100

        # Reward shaping to encourage moving towards the goal
        # position, velocity = next_state
        # reward = abs(position - (-0.5))  # Reward for moving towards the flag at position 0.5
        # if done:
        #     reward = 100  # High reward for reaching the goal

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state

        if done:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
            break

    writer.add_scalar('Episode Reward', total_reward, episode)

    agent.decay_epsilon()

    # Update the target network periodically
    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()

save_path = "checkpointv1.pth"

# Save both model and optimizer state dicts
torch.save({
    'model_state_dict': agent.q_network.state_dict(),
    'optimizer_state_dict': agent.optimizer.state_dict(),
}, save_path)

writer.close()
env.close()

# env = gym.make('MountainCar-v0', render_mode='human')
# agent = DQNAgent(env)
#
# checkpoint = torch.load("checkpoint.pth")
#
# # Load the model state dict
# agent.q_network.load_state_dict(checkpoint['model_state_dict'])
#
# # Load the optimizer state dict (for resuming training)
# agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
# print(f"Model and optimizer loaded.")
#
# state, _ = env.reset()
# done = False
#
# while not done:
#     action = agent.select_action(state)
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
#     state = next_state
#
# env.close()