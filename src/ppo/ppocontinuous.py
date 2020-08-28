import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
#from torch.distributions import MultivariateNormal



# Hyper parameters
gamma = 0.95
clip_ratio = 0.2
pi_lr = 0.003  # Learning rate for policy optimizer.
vf_lr = 0.003  # Learning rate for value function optimizer.
k_epochs = 4
update_every_j_timestep = 50
max_episode_length = 1_000
max_steps = 500
critic_hidden_size = 32
actor_hidden_size = 32
render = False


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Actor
class Actor(nn.Module):
    def __init__(self, num_of_observations, posibles_actions, hidden_size=actor_hidden_size):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_of_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, posibles_actions),
            nn.Tanh(),
        )

    def forward(self, x):
        probs = self.actor(x)
        return probs

    def act(self, state, memory):

        state = torch.from_numpy(state).float().to(device)

        probs = self.forward(state)
        
        // Como calcular la sigma de las probs

        dist = Normal(probs, covariance_matrix)
        action = dist.sample()
        logprobs = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprobs)
        return action.item()
    
    def play(self, state):
        state = torch.from_numpy(state).float().to(device)
        probs = self.forward(state)
        _, action = probs.max(0)
        return action.item()


# Critic
class Critic(nn.Module):
    def __init__(self, num_of_observations, posibles_actions, hidden_size=critic_hidden_size):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_of_observations, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1),
        )

    def forward(self, x):
        probs = self.actor(x)
        return probs

    def evaluate(self, state, action, actor):
        action_probs = actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self,
                 num_of_observations,
                 posibles_actions,
                 environment_name,
                 actor_hidden_size=actor_hidden_size,
                 critic_hidden_size=critic_hidden_size,
                 pi_lr=pi_lr,
                 vf_lr=vf_lr,
                 gamma=gamma,
                 k_epochs=k_epochs,
                 clip_ratio=clip_ratio,
                 ):
        # Current
        self.actor = Actor(num_of_observations=num_of_observations,
                           posibles_actions=posibles_actions, hidden_size=actor_hidden_size).to(device)
        self.critic = Critic(num_of_observations=num_of_observations,
                             posibles_actions=posibles_actions, hidden_size=critic_hidden_size).to(device)

        # Old
        self.actor_old = Actor(num_of_observations=num_of_observations,
                               posibles_actions=posibles_actions, hidden_size=actor_hidden_size).to(device)
        self.critic_old = Critic(num_of_observations=num_of_observations,
                                 posibles_actions=posibles_actions, hidden_size=critic_hidden_size).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.actor_critic = optim.Adam(self.critic.parameters(), lr=vf_lr)

        # Save parameters
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.clip_ratio = clip_ratio
        self.environment_name = environment_name
        # Loss
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = self.compute_returns(self.gamma, memory)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for e in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.critic.evaluate(
                old_states, old_actions, self.actor)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)
            

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio,
                                1+self.clip_ratio) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.actor_optimizer.zero_grad()
            self.actor_critic.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.actor_critic.step()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clean memory
        memory.clear_memory()

    def compute_returns(self, gamma, memory):
        R = 0
        returns = []
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                R = 0
            R = reward + gamma * R
            returns.insert(0, R)

        # Normalizing the returns:
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns


def train(environment_name,
          solved_reward,
          gamma=0.95,
          clip_ratio=0.2,
          pi_lr=0.003,  # Learning rate for policy optimizer.
          vf_lr=0.003,  # Learning rate for value function optimizer.
          k_epochs=4,
          update_every_j_timestep=50,
          max_episode_length=1_000,
          max_steps=500,
          critic_hidden_size=32,
          actor_hidden_size=32,
          render=False,
          random_seed=None,
          posibles_actions=None):


    # logging variables
    running_reward = 0
    avg_length = 0
    log_interval = 10


    # Environment
    env = gym.make(environment_name)
    num_of_observations = env.observation_space.shape[0]
    if (posibles_actions == None):
        posibles_actions = env.action_space.shape[0]

    # Setup tensorboard
    writer = SummaryWriter('board/ppo')

    # Random
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    memory = Memory()
    timestep = 0
    ppo = PPO(num_of_observations=num_of_observations,
              environment_name=environment_name,
              posibles_actions=posibles_actions,
              actor_hidden_size=actor_hidden_size,
              critic_hidden_size=critic_hidden_size,
              pi_lr=pi_lr,
              vf_lr=vf_lr,
              gamma=gamma,
              k_epochs=k_epochs,
              clip_ratio=clip_ratio)  # use hyperparametres

    for i_episode in range(1, max_episode_length+1):
        state = env.reset()
        for t in range(1, max_steps + 1):
            timestep +=1
            # Running policy_old:
            action = ppo.actor_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_every_j_timestep == 0:
                ppo.update(memory)

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        writer.add_scalar('running_reward', int(
            (running_reward / log_interval)))
        writer.add_scalar('avg_length', int(avg_length/log_interval))

        if i_episode % log_interval == 0:
            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, int(avg_length/log_interval), int((running_reward/log_interval))))
            running_reward = 0
            avg_length = 0
    
        x = int((running_reward/log_interval))            
        if solved_reward < x:
            print("-----> goal archived, stop the training")
            break

    
    print("End training")
    torch.save(ppo.actor_old.state_dict(), './model/ppo_{}_latest.pth'.format(environment_name))
    return ppo
    
def play_latest(environment_name, size):
       

    env = gym.make(environment_name)
    num_of_observations = env.observation_space.shape[0]
    posibles_actions = env.action_space.n
    state = env.reset()
    
    actor = Actor(num_of_observations, posibles_actions, hidden_size=size).to(device)
    actor.load_state_dict(torch.load('./model/ppo_{}_latest.pth'.format(environment_name)))
    
    done = False
    total_reward = 0    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    while (not done):
        action = actor.play(state)
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
    
    env.close()
    print("total reward {}".format(total_reward))
    
        