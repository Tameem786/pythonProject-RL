import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Define the Actor network architecture
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # Define two fully connected layers: first layer maps input state to 128 units
        self.fc1 = nn.Linear(state_size, 128)
        # Second layer maps the 128 units to the number of possible actions
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state)) # Apply ReLU activation
        action_probs = torch.softmax(self.fc2(x), dim=-1) # Output action probabilities using softmax
        return action_probs

# Define the Critic network architecture
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        # Define two fully connected layers: first layer maps input state to 128 units
        self.fc1 = nn.Linear(state_size, 128)
        # Second layer outputs a single value representing the state value
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state)) # Apply ReLU activation
        return self.fc2(x) # Output the state value

# Set hyperparameters for training
learning_rate_actor = 1e-3 # Learning rate for the Actor
learning_rate_critic = 1e-3 # Learning rate for the Critic
gamma = 0.99 # Discount factor for future rewards
num_episodes = 1000 # Total number of episodes to train
batch_size = 64 # Size of the batch (not currently used in your code)
update_frequency = 5 # How often to update the Actor network

# Create a TensorBoard writer for logging
writer = SummaryWriter()

def train(actor, critic, env, num_episodes):
    # Initialize optimizers for the Actor and Critic networks
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate_critic)

    total_rewards = [] # List to store total rewards per episode

    for episode in range(num_episodes):
        # Reset the environment at the beginning of each episode
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0) # Add a batch dimension
        done = False # Flag to track if the episode has ended
        total_reward = 0 # Initialize total reward for the episode
        step = 0 # Step counter for update frequency

        while not done: # Loop until the episode is done
            action_probs = actor(state) # Get action probabilities from the Actor
            action = torch.multinomial(action_probs, num_samples=1).item() # Sample an action based on the probabilities

            next_state, reward, done, _, _ = env.step(action) # Execute the action in the environment
            next_state = torch.FloatTensor(next_state).unsqueeze(0) # Convert next state to tensor

            # Calculate TD target and TD error
            state_value = critic(state) # Critic's estimated value for the current state
            next_state_value = critic(next_state).detach() # Critic's estimated value for the next state
            td_target = reward + (1-done)*gamma*next_state_value # Calculate TD target
            td_error = td_target - state_value # TD error

            # Update the Critic network
            critic_loss = td_error.pow(2).mean() # Mean squared TD error
            critic_optimizer.zero_grad() # Zero gradients before backpropagation
            critic_loss.backward() # Backpropagate the loss
            critic_optimizer.step() # Update the Critic

            step += 1 # Increment step counter
            if step % update_frequency == 0: # Update Actor every 'update_frequency' steps
                log_prob = torch.log(action_probs.squeeze(0)[action]) # Get log probability of the action taken
                actor_loss = -log_prob * td_error.detach() # Calculate Actor loss
                actor_loss = actor_loss.mean() # Ensure scalar loss
                actor_optimizer.zero_grad() # Zero gradients before backpropagation
                actor_loss.backward() # Backpropagate the loss
                actor_optimizer.step() # Update the Actor

            state = next_state # Move to the next state
            total_reward += reward # Accumulate total reward

        writer.add_scalar('Total Reward', total_reward, episode) # Log total reward for the episode
        total_rewards.append(total_reward) # Append total reward for the episode
        if episode % 10 == 0: # Print progress every 10 episodes
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    writer.close() # Close the TensorBoard writer
    torch.save(actor.state_dict(), 'actor.pth') # Save the Actor model
    torch.save(critic.state_dict(), 'critic.pth') # Save the Critic model

def test(agent, env):
    # Reset the environment for testing
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0) # Add a batch dimension
    done = False # Flag to track if the episode has ended
    total_reward = 0 # Initialize total reward for testing

    while not done: # Loop until the episode is done
        env.render() # Render the environment
        action_probs = agent(state) # Get action probabilities from the Actor
        action = torch.multinomial(action_probs, num_samples=1).item() # Sample an action based on the probabilities

        next_state, reward, done, _, _ = env.step(action) # Execute the action in the environment
        state = torch.FloatTensor(next_state).unsqueeze(0) # Convert next state to tensor
        total_reward += reward # Accumulate total reward

    print(f"Total Reward in Test: {total_reward}")

if __name__ == '__main__':
    env = gym.make('LunarLander-v2') # Create the LunarLander-v2 environment, please add render_mode='human' while testing.
    state_size = env.observation_space.shape[0] # Get the size of the state space
    action_size = env.action_space.n # Get the size of the action space

    actor = Actor(state_size, action_size) # Instantiate the Actor
    critic = Critic(state_size) # Instantiate the Critic

    train(actor, critic, env, num_episodes) # Train the model

    # actor.load_state_dict(torch.load('actor.pth')) # Load the trained Actor model
    # test(actor, env) # Test the model
