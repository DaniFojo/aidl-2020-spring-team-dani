import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np


from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


# def prepro(img): return imresize(img[35:195].mean(2), (80, 80)).astype(
#     np.float32).reshape(1, 80, 80) / 255.


def prepro(img): return np.array(Image.fromarray(img[35:195].mean(2)).resize((80, int(80)))) .astype(
    np.float32).reshape(1, 1,  80, 80) / 255.


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
    def __init__(self, num_of_observations, posibles_actions, dropout, hidden_size, initialization=None,):
        super(Actor, self).__init__()
        self.num_of_observations = num_of_observations
        self.fc1 = nn.Linear(800, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, posibles_actions)
        if dropout == None:
            dropout = 0
        if initialization == "normal":
            self.actor.apply(init_normal)

        if initialization == "orthogonal":
            # self.actor.apply(init_orthogonal)
            layer_init(self.fc1)
            layer_init(self.fc2)
            layer_init(self.fc3, 1e-3)

        self.actor = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            Flatten(),
            self.fc1,
            nn.Dropout(p=dropout),
            nn.Tanh(),
            self.fc2,
            nn.Dropout(p=dropout),
            nn.Tanh(),
            self.fc3,
            nn.Softmax(dim=-1)
        )

        # self.actor = nn.Sequential(
        #     self.fc1,
        #     nn.Dropout(p=dropout),
        #     nn.Tanh(),
        #     self.fc2,
        #     nn.Dropout(p=dropout),
        #     nn.Tanh(),
        #     self.fc3,
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, x):
        probs = self.actor(x)
        return probs

    def act(self, state, memory):
        #state = torch.from_numpy(state).float().to(device)
        state = state.to(device)
        probs = self.forward(state)

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
    def __init__(self, num_of_observations, posibles_actions, dropout, hidden_size, initialization=None):
        super(Critic, self).__init__()
        self.num_of_observations = num_of_observations
        self.fc1 = nn.Linear(800, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        if dropout == None:
            dropout = 0
        self.critic = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            Flatten(),
            self.fc1,
            nn.Dropout(p=dropout),
            nn.Tanh(),
            self.fc2,
            nn.Dropout(p=dropout),
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
        probs = self.critic(x)
        return probs

    def evaluate(self, state, action, actor):
        state = state.reshape(-1, 1, 80, 80)
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
                 betas,
                 num_mini_batch,
                 update_every_j_timesteps,
                 actor_hidden_size,
                 critic_hidden_size,
                 pi_lr,
                 vf_lr,
                 gamma,
                 k_epochs,
                 clip_ratio,
                 previousTrainedPolicy=None,
                 previousTrainedCritic=None,
                 actorGradientNormalization=0,
                 observationNormalization=False,
                 initialization=None,
                 advantageAlgorithm=None,
                 normalizeAdvantage=False,
                 dropout=None,

                 ):
        # Current
        self.actor = Actor(num_of_observations=num_of_observations,
                           posibles_actions=posibles_actions, dropout=dropout, hidden_size=actor_hidden_size).to(device)
        self.critic = Critic(num_of_observations=num_of_observations,
                             posibles_actions=posibles_actions, dropout=dropout, hidden_size=critic_hidden_size).to(device)

        if previousTrainedPolicy:
            self.actor.load_state_dict(previousTrainedPolicy)
        if previousTrainedCritic:
            self.critic.load_state_dict(previousTrainedCritic)

        # Old
        self.actor_old = Actor(num_of_observations=num_of_observations,
                               posibles_actions=posibles_actions, dropout=dropout, hidden_size=actor_hidden_size, initialization=initialization).to(device)
        self.critic_old = Critic(num_of_observations=num_of_observations,
                                 posibles_actions=posibles_actions, dropout=dropout, hidden_size=critic_hidden_size, initialization=initialization).to(device)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        p = list(self.actor.parameters()) + list(self.critic.parameters())

        if betas:
            self.actor_optimizer = optim.Adam(
                p, lr=pi_lr, eps=eps, betas=betas)
        else:
            self.actor_optimizer = optim.Adam(
                p, lr=pi_lr, weight_decay=weight_decay,  eps=eps)
        # self.critic_optimizer = optim.Adam(self.critic.parameters()
        # , lr=vf_lr, weight_decay=weight_decay, eps=eps)

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
        self.num_of_observations = num_of_observations
        self.num_mini_batch = num_mini_batch
        self.posibles_actions = posibles_actions
        self.update_every_j_timesteps = update_every_j_timesteps

        # Loss
        self.mseLoss = nn.MSELoss()

    def update(self, memory, writer, mini_batch, i_episode, totalMiniBatch):

        z_logProbs = torch.zeros(1, mini_batch)
        z_values = torch.zeros(1, mini_batch)
        z_states = torch.zeros(mini_batch, self.num_of_observations)
        z_actions = torch.zeros(mini_batch, self.posibles_actions)
        z_rewards = torch.zeros(1, mini_batch)
        z_dones = torch.zeros(1, mini_batch)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        # old_states = torch.cat((torch.zeros(
        #     mini_batch, self.num_of_observations).to(device), old_states), 0)

        old_actions = torch.stack(memory.actions).to(device).detach()
        # x = torch.zeros(
        #     mini_batch, dtype=long).to(device)
        # print(x.shape, old_actions.shape)
        # old_actions = torch.cat((old_actions, x), 0)
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        # old_logprobs = torch.cat((torch.zeros(
        #     mini_batch).to(device), old_logprobs), 0)
        old_rewards = torch.tensor(
            memory.rewards, dtype=torch.float32).to(device).detach()
        # old_rewards = torch.cat((torch.zeros(
        #     mini_batch).to(device), old_rewards), 0)
        old_is_terminal = torch.tensor(
            memory.is_terminals, dtype=torch.float32).to(device).detach()
        # old_is_terminal = torch.cat((torch.zeros(
        #     mini_batch).to(device), old_is_terminal), 0)

        # print(old_states.shape, old_actions.shape, old_logprobs.shape,
        #       old_rewards.shape, old_is_terminal.shape)

        self.actor.train()
        self.critic.train()
        for e in range(self.k_epochs):
            for i in range(int((self.update_every_j_timesteps / mini_batch))):
                index = i * mini_batch
                totalMiniBatch += 1
                # Slice
                writer.add_scalar('2 - Training/mini batch',
                                  totalMiniBatch, global_step=i_episode)
                mini_old_states = old_states.narrow(0, i, mini_batch)
                mini_old_actions = old_actions.narrow(0, i, mini_batch)
                mini_old_logprobs = old_logprobs.narrow(0, i, mini_batch)
                mini_rewards = old_rewards.narrow(0, i, mini_batch)
                mini_is_terminal = old_is_terminal.narrow(0, i, mini_batch)

                # mini_old_states = torch.cat(
                #     (torch.zeros(4, 128).to(device), mini_old_states), 0)
                # mini_old_actions = mini_old_actions.cat(torch.zeros(mini_batch -1))
                # mini_old_logprobs = mini_old_logprobs.cat(torch.zeros(mini_batch, mini_batch -1))
                # mini_rewards = mini_rewards.cat(torch.zeros(mini_batch, mini_batch -1))
                # mini_is_terminal = mini_is_terminal.cat(torch.zeros(mini_batch, mini_batch -1))

                # print(mini_old_states.shape, mini_old_actions.shape, mini_old_logprobs.shape,
                #       mini_rewards.shape, mini_is_terminal.shape)

                # Evaluating old actions and values :
                logprobs, values, dist_entropy = self.critic.evaluate(
                    mini_old_states, mini_old_actions, self.actor)

                # print(logprobs.shape, values.shape, dist_entropy.shape)
                # GAE Compute rewards
                if self.advantageAlgorithm == "GAE":

                    rewards, advantages = self.gae(
                        self.gamma, mini_rewards, mini_is_terminal, values, normalizeAdvantage=self.normalizeAdvantage)
                else:
                    rewards, advantages = self.compute_returns(
                        self.gamma, mini_rewards, mini_is_terminal)

                ratios = torch.exp(logprobs - mini_old_logprobs)
                writer.add_scalar("1 - loss/ratios",
                                  ratios.mean(), global_step=i_episode)
                writer.add_scalar("1 - loss/advantages",
                                  advantages.mean(), global_step=i_episode)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio,
                                    1 + self.clip_ratio) * advantages
                policy_loss = torch.min(surr1, surr2)
                value_loss = self.coeficient_value * \
                    F.smooth_l1_loss(values, rewards)
                # entropy del critic or actor?
                entropy = self.coeficient_entropy * dist_entropy
                loss = -policy_loss + value_loss - entropy

                writer.add_scalar("1 - loss/policy",
                                  policy_loss.mean(), global_step=i_episode)
                writer.add_scalar("1 - loss/value",
                                  value_loss.mean(), global_step=i_episode)
                writer.add_scalar("1 - loss/entropy",
                                  dist_entropy.mean(), global_step=i_episode)
                writer.add_scalar("1 - loss/loss", loss.mean(),
                                  global_step=i_episode)

                self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()
                loss.sum().backward()
                if (self.actorGradientNormalization != 0):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.actorGradientNormalization)
                self.actor_optimizer.step()
                # self.critic_optimizer.step()

            # Copy new weights into old policy:
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.critic_old.load_state_dict(self.critic.state_dict())

            # clean memory
            memory.clear_memory()
        return totalMiniBatch

    def compute_returns(self, gamma, rewards, is_terminals):
        R = 0

        returns = []
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                R = 0
            R = reward + gamma * R * self.lmbda
            returns.insert(0, R)

        # Normalizing the returns:
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        v_target = returns.clone()
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns, v_target

    def gae(self, gamma, rewards, is_terminals, v, normalizeAdvantage=False):
        gae = 0
        returns = []
        adv = []
        values = (v.data).cpu().numpy()
        for reward, is_terminal, value in zip(reversed(rewards), reversed(is_terminals), reversed(values)):
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
    # return (data - np.min(data)) / (np.max(data) - np.min(data))
    return data / 255


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
          num_mini_batch=None,
          betas=None,
          dropout=0,
          saveWithTheSameNameThatTensorboard=False,
          atari=True,
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
    if num_mini_batch == None:
        num_mini_batch = update_every_j_timestep

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
              normalizeAdvantage=normalizeAdvantage,
              betas=betas,
              dropout=dropout,
              num_mini_batch=num_mini_batch,
              update_every_j_timesteps=update_every_j_timestep,
              )

    globlalStep = 0
    startTime = time.time()
    totalReward = 0
    totalMiniBatch = 0
    for i_episode in range(1, max_episode_length+1):
        state = torch.tensor(prepro(env.reset()))
        episode_reward = 0

        writer.add_scalar('2 - Training/episode',
                          i_episode, global_step=i_episode)
        for t in range(1, max_steps + 1):
            ppo.actor_old.eval()
            timestep += 1
            globlalStep += 1
            # normalize state
            # https://arxiv.org/pdf/2006.05990.pdf
            if (observationNormalization):
                state = normalizeArray(state)
            # Running policy_old:
            action = ppo.actor_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            state = torch.tensor(prepro(state))

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_every_j_timestep == 0:
                totalMiniBatch = ppo.update(
                    memory, writer, num_mini_batch, i_episode, totalMiniBatch)

            running_reward += reward
            episode_reward += reward
            totalReward += reward

            writer.add_scalar('2 - Training/lr',
                              get_lr(ppo.actor_optimizer), global_step=i_episode)
            # writer.add_scalar('2 - lerning rate/critic',
            #                   get_lr(ppo.critic_optimizer))

            if render:
                env.render()
            if done:
                break

        avg_length += t
        writer.add_scalar('0 - reward/episode',
                          int(episode_reward), global_step=i_episode)
        writer.add_scalar('2 - Training/steps',
                          int(globlalStep), global_step=i_episode)

        x = int((running_reward / log_interval))
        if solved_reward <= x:
            print(
                f"-----> goal archived, stop the training at episode:{i_episode}")
            break
        if i_episode % log_interval == 0:
            elapsedTime = time.time() - startTime
            t = getString(elapsedTime)
            print('Episode {} \t avg length: {} \t avg reward: {} \t\t globalStep: {} \t elapse time: {}'.format(
                i_episode, int(avg_length / log_interval), int((running_reward / log_interval)), globlalStep, t))
            writer.add_scalar('0 - reward/running reward', int(
                (running_reward / log_interval)), global_step=i_episode)
            writer.add_scalar('0 - reward/running length',
                              int(avg_length/log_interval), global_step=i_episode)
            running_reward = 0
            avg_length = 0
            startTime = time.time()

        if i_episode % saveModelsEvery == 0:
            numK = i_episode // 1_000
            torch.save(ppo.actor_old.state_dict(),
                       './model/ppo_{}_policy_{}K.pth'.format(environment_name, numK))
            torch.save(ppo.critic_old.state_dict(),
                       './model/ppo_{}_critic_{}K.pth'.format(environment_name, numK))

    writer.add_hparams({"gammma": gamma,
                        "clip_ratio": clip_ratio,
                        "lr": pi_lr,
                        "UE": update_every_j_timestep,
                        "ME": max_episode_length,
                        "MS": max_steps,
                        "AS": actor_hidden_size,
                        "CS": critic_hidden_size,
                        "ON": observationNormalization,
                        "AN": actorGradientNormalization,
                        "NA": normalizeAdvantage,
                        "CE": coeficient_entropy,
                        "CV": coeficient_value,
                        "L": lmbda,
                        "EPS": eps,
                        "DO": dropout,
                        },
                       {"TotalReward": totalReward})
    print("End training")
    if saveWithTheSameNameThatTensorboard:
        torch.save(ppo.actor_old.state_dict(),
                   './model/ppo_{}_policy_{}.pth'.format(environment_name, tensorboardName))
        torch.save(ppo.critic_old.state_dict(),
                   './model/ppo_{}_critic_{}.pth'.format(environment_name, tensorboardName))
    else:
        torch.save(ppo.actor_old.state_dict(),
                   './model/ppo_{}_policy_latest.pth'.format(environment_name))
        torch.save(ppo.critic_old.state_dict(),
                   './model/ppo_{}_critic_latest.pth'.format(environment_name))

    env.close()
    writer.close()
    return ppo


def getString(date):
    hours, rem = divmod(
        date, 3600)
    minutes, seconds = divmod(
        rem, 60)
    t = "{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds)
    return t


def multitrain(environment_name,
               lrs,
               k_epochss,
               dropouts,
               observationNormalizations,
               actorGradientNormalizations,
               normalizeAdvantages,
               coeficient_entropys,
               coeficient_values,
               lmbdas,
               actor_sizes,
               critic_sizes,
               gammas,
               clip_ratios,
               update_every_j_timesteps,
               max_episode_lengths,
               max_steps,
               epss,
               betass,
               solved_reward,
               num_mini_batchs):
    for lr in lrs:
        for dropout in dropouts:
            for observationNormalization in observationNormalizations:
                for actorGradientNormalization in actorGradientNormalizations:
                    for normalizeAdvantage in normalizeAdvantages:
                        for coeficient_entropy in coeficient_entropys:
                            for coeficient_value in coeficient_values:
                                for lmbda in lmbdas:
                                    for actor_size in actor_sizes:
                                        for critic_size in critic_sizes:
                                            for gamma in gammas:
                                                for clip_ratio in clip_ratios:
                                                    for update_every_j_timestep in update_every_j_timesteps:
                                                        for max_episode_length in max_episode_lengths:
                                                            for max_step in max_steps:
                                                                for eps in epss:
                                                                    for betas in betass:
                                                                        for num_mini_batch in num_mini_batchs:
                                                                            for k_epochs in k_epochss:
                                                                                tensorboardName = f"S={actor_size}-{critic_size},\={lmbda},CV={coeficient_value},CE={coeficient_entropy},NA={normalizeAdvantage},AN={actorGradientNormalization},ON={observationNormalization},DO={dropout},LR={lr},MS={max_step},US={update_every_j_timestep},MB={num_mini_batch}"
                                                                                startTime = time.time()
                                                                                train(environment_name,
                                                                                      solved_reward,
                                                                                      gamma=gamma,
                                                                                      clip_ratio=clip_ratio,
                                                                                      # Learning rate for policy optimizer.
                                                                                      pi_lr=lr,
                                                                                      k_epochs=k_epochs,
                                                                                      update_every_j_timestep=update_every_j_timestep,
                                                                                      max_episode_length=max_episode_length,
                                                                                      max_steps=max_step,
                                                                                      critic_hidden_size=critic_size,
                                                                                      actor_hidden_size=actor_size,
                                                                                      render=False,
                                                                                      random_seed=1,
                                                                                      posibles_actions=None,
                                                                                      pathForBasePolicyToTrain=None,
                                                                                      pathForBaseCriticToTrain=None,
                                                                                      observationNormalization=observationNormalization,
                                                                                      actorGradientNormalization=actorGradientNormalization,
                                                                                      eps=eps,
                                                                                      coeficient_entropy=coeficient_entropy,
                                                                                      weight_decay=1e-4,
                                                                                      coeficient_value=coeficient_value,
                                                                                      lmbda=lmbda,
                                                                                      initialization="orthogonal",
                                                                                      advantageAlgorithm="GAE",
                                                                                      saveModelsEvery=100,
                                                                                      tensorboardName=tensorboardName,
                                                                                      normalizeAdvantage=normalizeAdvantage,
                                                                                      num_mini_batch=num_mini_batch,
                                                                                      betas=None,
                                                                                      dropout=dropout,
                                                                                      saveWithTheSameNameThatTensorboard=True
                                                                                      )
                                                                                endTime = time.time()
                                                                                t = getString(
                                                                                    endTime - startTime)
                                                                                print(
                                                                                    f"{t}--> done: {tensorboardName}")
    print("Training done!!!")


def play_name(size, environment_name, name, plot=False, observationNormalization=False):

    env = gym.make(environment_name)
    num_of_observations = prodOfTupple(env.observation_space.shape)

    posibles_actions = env.action_space.n
    state = env.reset()

    if (observationNormalization):
        state = normalizeArray(state)

    actor = Actor(num_of_observations, posibles_actions, 0,
                  hidden_size=size).to(device)
    actor.load_state_dict(torch.load(name))

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
        if (observationNormalization):
            state = normalizeArray(state)
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


def play_latest(environment_name, size, plot=False, observationNormalization=False):

    name = './model/ppo_{}_policy_latest.pth'.format(environment_name)
    play_name(size, environment_name, name, plot=plot,
              observationNormalization=observationNormalization)
