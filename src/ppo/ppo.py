import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


# Hyper parameters
gamma = 0.95
clip_ratio = 0.2
pi_lr = 0.003  # Learning rate for policy optimizer.
vf_lr = 0.003  # Learning rate for value function optimizer.
k_epochs = 4
update_every_j_timestep = 50
max_episode_length = 1_000
max_steps = 500
critic_hidden_size = 256
actor_hidden_size = 16
render = False
mini_batch = 32


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inicializers


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)


def init_orthogonal(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def layer_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


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
    def __init__(self, num_of_observations, posibles_actions, hidden_size=actor_hidden_size, initialization=None):
        super(Actor, self).__init__()
        self.num_of_observations = num_of_observations
        self.fc1 = nn.Linear(num_of_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, posibles_actions)
        self.actor = nn.Sequential(
            self.fc1,
            nn.Tanh(),
            self.fc2,
            nn.Tanh(),
            self.fc3,
            nn.Softmax(dim=-1)
        )
        if initialization == "normal":
            self.actor.apply(init_normal)

        if initialization == "orthogonal":
            # self.actor.apply(init_orthogonal)
            layer_init(self.fc1)
            layer_init(self.fc2)
            layer_init(self.fc3, 1e-3)

    def forward(self, x):
        probs = self.actor(x)
        return probs

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)

        probs = self.forward(state.flatten())

        dist = Categorical(probs)
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
        return action.item(), probs.data


# Critic
class Critic(nn.Module):
    def __init__(self, num_of_observations, posibles_actions, hidden_size=critic_hidden_size, initialization=None):
        super(Critic, self).__init__()
        self.num_of_observations = num_of_observations
        self.fc1 = nn.Linear(num_of_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.critic = nn.Sequential(
            self.fc1,
            nn.Tanh(),
            self.fc2,
            nn.Tanh(),
            self.fc3,
        )

        if initialization == "normal":
            self.critic.apply(init_normal)
        if initialization == "orthogonal":
            # self.critic.apply(init_orthogonal)
            layer_init(self.fc1)
            layer_init(self.fc2)
            layer_init(self.fc3, 1e-3)

    def forward(self, x):
        probs = self.actor(x)
        return probs

    def evaluate(self, state, action, actor):
        state = state.reshape((-1, self.num_of_observations))
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
                 eps,
                 coeficient_entropy,
                 coeficient_value,
                 weight_decay,
                 lmbda,
                 actor_hidden_size=actor_hidden_size,
                 critic_hidden_size=critic_hidden_size,
                 pi_lr=pi_lr,
                 vf_lr=vf_lr,
                 gamma=gamma,
                 k_epochs=k_epochs,
                 clip_ratio=clip_ratio,
                 previousTrainedPolicy=None,
                 previousTrainedCritic=None,
                 actorGradientNormalization=0,
                 observationNormalization=False,
                 initialization=None,
                 advantageAlgorithm=None,
                 normalizeAdvantage=False,
                 ):
        # Current
        self.actor = Actor(num_of_observations=num_of_observations,
                           posibles_actions=posibles_actions, hidden_size=actor_hidden_size).to(device)
        self.critic = Critic(num_of_observations=num_of_observations,
                             posibles_actions=posibles_actions, hidden_size=critic_hidden_size).to(device)

        if previousTrainedPolicy:
            self.actor.load_state_dict(previousTrainedPolicy)
        if previousTrainedCritic:
            self.critic.load_state_dict(previousTrainedCritic)

        # Old
        self.actor_old = Actor(num_of_observations=num_of_observations,
                               posibles_actions=posibles_actions, hidden_size=actor_hidden_size, initialization=initialization).to(device)
        self.critic_old = Critic(num_of_observations=num_of_observations,
                                 posibles_actions=posibles_actions, hidden_size=critic_hidden_size, initialization=initialization).to(device)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=pi_lr, weight_decay=weight_decay,  eps=eps)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=vf_lr, weight_decay=weight_decay, eps=eps)

        # Save parameters
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.clip_ratio = clip_ratio
        self.environment_name = environment_name
        self.actorGradientNormalization = actorGradientNormalization
        self.observationNormalization = observationNormalization
        self.eps = eps
        self.coeficient_entropy = coeficient_entropy
        self.weight_decay = weight_decay
        self.coeficient_value = coeficient_value
        self.lmbda = lmbda
        self.advantageAlgorithm = advantageAlgorithm
        self.normalizeAdvantage = normalizeAdvantage

        # Loss
        self.mseLoss = nn.MSELoss()

    def update(self, memory, writer, num_mini_batch):

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for e in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, values, dist_entropy = self.critic.evaluate(
                old_states, old_actions, self.actor)

            # GAE Compute rewards
            if self.advantageAlgorithm == "GAE":
                rewards, advantages = self.gae(
                    self.gamma, memory, values, normalizeAdvantage=self.normalizeAdvantage)
            else:
                rewards, advantages = self.compute_returns(self.gamma, memory)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio,
                                1 + self.clip_ratio) * advantages
            policy_loss = torch.min(surr1, surr2)
            value_loss = self.coeficient_value * \
                F.smooth_l1_loss(values, rewards)
            # entropy del critic or actor?
            entropy = self.coeficient_entropy * dist_entropy
            loss = -policy_loss + value_loss - entropy

            writer.add_scalar("1 - loss/policy", policy_loss.mean())
            writer.add_scalar("1 - loss/value", value_loss.mean())
            writer.add_scalar("1 - loss/entropy", dist_entropy.mean())
            writer.add_scalar("1 - loss/loss", loss.mean())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            if (self.actorGradientNormalization != 0):
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.actorGradientNormalization)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

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
            R = reward + gamma * R * self.lmbda
            returns.insert(0, R)

        # Normalizing the returns:
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        v_target = returns.clone()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns, v_target

    def gae(self, gamma, memory, v, normalizeAdvantage=False):
        gae = 0
        returns = []
        adv = []
        values = (v.data).cpu().numpy()
        for reward, is_terminal, value in zip(reversed(memory.rewards), reversed(memory.is_terminals), reversed(values)):
            done_mask = not is_terminal
            delta = reward + gamma * value * done_mask - value
            gae = delta + gamma * self.lmbda * done_mask * gae
            returns.insert(0, gae + value)
            adv.insert(0, gae)

        if normalizeAdvantage:
            adv = (adv - np.mean(adv)) / (np.std(adv) + self.eps)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        adv = torch.tensor(adv, dtype=torch.float32).to(device)
        return returns, adv


def prodOfTupple(val):
    val = list(val)
    res = 1
    for ele in val:
        res *= ele
    return res


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalizeArray(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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
          critic_hidden_size=256,
          actor_hidden_size=16,
          render=False,
          random_seed=None,
          posibles_actions=None,
          pathForBasePolicyToTrain=None,
          pathForBaseCriticToTrain=None,
          observationNormalization=False,
          actorGradientNormalization=0,
          eps=None,
          coeficient_entropy=0.01,
          weight_decay=1e-4,
          coeficient_value=0.5,
          lmbda=0.95,
          initialization="orthogonal",
          advantageAlgorithm="GAE",
          saveModelsEvery=10_000,
          tensorboardName=None,
          normalizeAdvantage=False,
          num_mini_batch=32
          ):

    previousTrainedPolicy = None
    if pathForBasePolicyToTrain:
        previousTrainedPolicy = torch.load(pathForBasePolicyToTrain)

    previousTrainedCritic = None
    if pathForBaseCriticToTrain:
        previousTrainedCritic = torch.load(pathForBaseCriticToTrain)

    # logging variables
    running_reward = 0
    avg_length = 0
    log_interval = 10

    # Environment
    env = gym.make(environment_name)

    num_of_observations = prodOfTupple(env.observation_space.shape)
    if (posibles_actions == None):
        posibles_actions = env.action_space.n

    if eps == None:
        eps = np.finfo(np.float32).eps.item()

    # Setup tensorboard
    if tensorboardName == None:
        tensorboardName = datetime.now().strftime("%Y%m%d%H%M")
    writer = SummaryWriter(
        f'board/ppo/{environment_name}/{tensorboardName}')

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
              clip_ratio=clip_ratio,
              previousTrainedPolicy=previousTrainedPolicy,
              previousTrainedCritic=previousTrainedCritic,
              observationNormalization=observationNormalization,
              actorGradientNormalization=actorGradientNormalization,
              eps=eps,
              coeficient_entropy=coeficient_entropy,
              weight_decay=weight_decay,
              lmbda=lmbda,
              coeficient_value=coeficient_value,
              initialization=initialization,
              advantageAlgorithm=advantageAlgorithm,
              normalizeAdvantage=normalizeAdvantage
              )

    globlalStep = 0
    startTime = time.time()
    for i_episode in range(1, max_episode_length+1):
        state = env.reset()
        episode_reward = 0
        for t in range(1, max_steps + 1):
            timestep += 1
            globlalStep += 1
            # normalize state
            # https://arxiv.org/pdf/2006.05990.pdf
            if (observationNormalization):
                state = normalizeArray(state)
            # Running policy_old:
            action = ppo.actor_old.act(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_every_j_timestep == 0:
                ppo.update(memory, writer, num_mini_batch)

            running_reward += reward
            episode_reward += reward

            writer.add_scalar('2 - lerning rate/actor',
                              get_lr(ppo.actor_optimizer))
            writer.add_scalar('2 - lerning rate/critic',
                              get_lr(ppo.critic_optimizer))

            if render:
                env.render()
            if done:
                break

        avg_length += t
        writer.add_scalar('0 - reward/episode', int(episode_reward))
        writer.add_scalar('0 - reward/steps', int(globlalStep))

        if i_episode % log_interval == 0:
            elapsedTime = time.time() - startTime
            print('Episode {} \t avg length: {} \t avg reward: {} \t\t globalStep: {} \t elapse time (ms): {}'.format(
                i_episode, int(avg_length / log_interval), int((running_reward / log_interval)), globlalStep, elapsedTime * 1000))
            writer.add_scalar('0 - reward/running reward', int(
                (running_reward / log_interval)))
            writer.add_scalar('0 - reward/running length',
                              int(avg_length/log_interval))
            running_reward = 0
            avg_length = 0
            startTime = time.time()

        x = int((running_reward/log_interval))
        if solved_reward < x:
            print("-----> goal archived, stop the training")
            break
        if i_episode % saveModelsEvery == 0:
            numK = i_episode // 1_000
            torch.save(ppo.actor_old.state_dict(),
                       './model/ppo_{}_policy_{}K.pth'.format(environment_name, numK))
            torch.save(ppo.critic_old.state_dict(),
                       './model/ppo_{}_critic_{}K.pth'.format(environment_name, numK))
    print("End training")
    torch.save(ppo.actor_old.state_dict(),
               './model/ppo_{}_policy_latest.pth'.format(environment_name))
    torch.save(ppo.critic_old.state_dict(),
               './model/ppo_{}_critic_latest.pth'.format(environment_name))
    env.close()
    writer.close()
    return ppo


def play_latest(environment_name, size, plot=False):

    env = gym.make(environment_name)
    num_of_observations = prodOfTupple(env.observation_space.shape)

    posibles_actions = env.action_space.n
    state = env.reset()

    actor = Actor(num_of_observations, posibles_actions,
                  hidden_size=size).to(device)
    actor.load_state_dict(torch.load(
        './model/ppo_{}_policy_latest.pth'.format(environment_name)))

    done = False
    total_reward = 0
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    i = 0

    if plot:
        y_pos = np.arange(posibles_actions)
        probLegend = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.xticks(y_pos, y_pos)
        plt.ylabel('Probs')
        plt.xlabel('Action')

    while (not done):
        env.render()
        state = np.ndarray.flatten(state)
        action, probs = actor.play(state)
        i = i + 1
        if plot:
            probs = probs.detach().cpu().numpy()
            ax = plt.bar(y_pos, probs, align='center',
                         alpha=0.5, color=(0.2, 0.4, 0.6, 0.6))
            plt.title(f'Step - {i}, Action choosed: {action}')
            plt.pause(0.02)
            ax.remove()
        print(f'step {i} \t action {action}')
        state, reward, done, _ = env.step(action)

        total_reward += reward

    print("total reward {}".format(total_reward))
    if plot:
        plt.show()
        plt.close()
    env.close()
