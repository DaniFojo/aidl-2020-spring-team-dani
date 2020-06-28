import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = "CartPole-v0"

env = gym.make(env_name) # One single env (A2C, synchronous)

# Hyperparameters
hparams = {
    'hidden_size':32,
    'learning_rate':1e-3
}

# Constants
GAMMA = 0.99
num_steps = 500
max_episodes = 3000

class Actor(nn.Module):
    """     The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).
            The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).
            Both the Critic and Actor functions are parameterized with neural networks. """

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax()
        )
        
    def forward(self, x):
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist

class Critic(nn.Module):
    """     The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).
            The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients).
            Both the Critic and Actor functions are parameterized with neural networks. """

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        return value


def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns

def plot(frame_idx, rewards):
    plt.plot(rewards,'b-')
    plt.title('frame %s. reward: %s' % (episode_idx, rewards[-1]))
    plt.pause(0.0001)


num_inputs  = env.observation_space.shape[0]
num_actions = env.action_space.n


actor = Actor(num_inputs, num_actions, hparams['hidden_size']).to(device)
critic = Critic(num_inputs, num_actions, hparams['hidden_size']).to(device)

optimizer_actor = optim.Adam(actor.parameters(), lr=hparams['learning_rate'])
optimizer_critic = optim.Adam(critic.parameters(), lr=hparams['learning_rate'])

episode_idx    = 0
test_rewards = []

all_lengths = []
average_lengths = []
all_rewards = []

while episode_idx < max_episodes:

    log_probs = []
    values    = []
    rewards   = []
    advantages  = []
    entropy = 0

    state = env.reset()

    # rollout trajectory
    for steps in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist = actor.forward(state)
        value = critic.forward(state)

        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        advantage = reward + (1-done)*GAMMA*critic(torch.FloatTensor(next_state).to(device)) - critic(state)
        advantages.append(advantage)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.from_numpy(np.array(reward)).to(device))

        optimizer_actor.zero_grad()
        actor_loss  = -log_prob * advantage.detach()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss = advantage.pow(2)
        critic_loss.backward()
        optimizer_critic.step()
        
        state = next_state

        if done: 
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                average_reward = np.mean(all_rewards[-10:])
                if episode_idx % 10 == 0:                    
                    print("episode: {}, avg_reward: {}, total length: {}, average length: {} \n".format(episode_idx, average_reward, steps, average_lengths[-1]))
                break    

    episode_idx += 1
    
    returns = compute_returns(rewards, GAMMA)

    log_probs = torch.stack(log_probs)
    returns   = torch.stack(returns)
    values    = torch.stack(values)