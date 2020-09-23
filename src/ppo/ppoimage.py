import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torchsummary import summary

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from PIL import Image
import numpy as np
import cv2
from skimage.color import rgb2grey
from skimage.transform import rescale

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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# def prepro(img): return np.array(Image.fromarray(img[35:195].mean(2)).resize((80, int(80)))) .astype(
#     np.float32).reshape(1, 1, 8, 80) / 255
def prepro(image):
    image = rgb2grey(image[35:195, :, :])
    image = rescale(image, scale=0.5)
    image = image[np.newaxis, np.newaxis, :, :]
    return torch.tensor(image, device=device, dtype=torch.float)

    # image_data = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    # image_data[image_data > 0] = 255
    # image_data = np.reshape(image_data, (80, 80, 1))
    # image_tensor = image_data.transpose(2, 0, 1)
    # image_tensor = image_tensor.astype(np.float32)
    # image_tensor = torch.from_numpy(image_tensor)
    # return image_tensor.reshape(1, 1, 80, 80)


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
    def __init__(self, num_of_observations, posibles_actions, dropout, hidden_size=actor_hidden_size, initialization=None,):
        super(Actor, self).__init__()
        self.num_of_observations = num_of_observations
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(6 * 6 * 64, 512)
        self.actor = nn.Sequential(
            nn.Linear(512, posibles_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(512, 1)

        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)

        #self.gru = nn.GRUCell(32 * 5 * 5, hidden_size)

        if dropout == None:
            dropout = 0
        if initialization == "normal":
            self.actor.apply(init_normal)

        if initialization == "orthogonal":
            # self.actor.apply(init_orthogonal)
            # layer_init(self.fc1)
            # layer_init(self.fc2)
            layer_init(self.fc2, 1e-3)

        self.seq = nn.Sequential(
            # self.fc1,
            # nn.Dropout(p=dropout),
            # nn.Tanh(),
            self.fc2,
            nn.Dropout(p=dropout),
            nn.Tanh()
        )

    def forward(self, x, hx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(f"forward {x.shape}")
        x = x.reshape((-1, 6 * 6 * 64))
        #print(f"forward ---> {x.shape}")
        #hx = self.gru(x, (hx))
        x = self.seq(x)
        #print(f"forward2 ---> {hx.shape}")
        probs = self.actor(x)
        return probs, hx, x

    def act(self, state, memory, hx, deterministic=False):
        self.eval()
        with torch.no_grad():
            state = state.to(device)
            # print(state.shape)
            probs, hx, _ = self.forward(state, hx)
            #print(state.shape, probs.shape)
            probs = probs.squeeze()

            dist = Categorical(probs)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            logprobs = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprobs)
        # print(action.shape, action.item())
        return action.item(), hx

    def play(self, state, hx):
        self.eval()
        with torch.no_grad():
            state = prepro(state).to(device)
            probs, hx = self.forward(state, hx)
            probs = probs.squeeze()
            _, action = probs.max(0)
        return (action.item(), probs, hx)

    def evaluate(self, state, action, hx):
        self.train()
        with torch.enable_grad():
            # print(state.shape)
            state = state.squeeze()
            #state = state.reshape(-1, 4, 80, 80)
            # print(state.shape)
            #print(f"**** ----->{state.shape}{action.shape}")
            action_probs, hx, inter = self.forward(state, hx)
            #print(f"**** ----->{state.shape}")
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            #print(f"**** 3 ----->{state.shape} {hxcritic.shape}")
            state_value = self.critic(inter)
            #print(f"**** 4 ----->{state.shape}")
            state_value = state_value.squeeze()
            # print(state_value.shape)

        return action_logprobs, state_value, dist_entropy, hx


# Critic
# class Critic(nn.Module):
#     def __init__(self, num_of_observations, posibles_actions, dropout, hidden_size=critic_hidden_size, initialization=None):
#         super(Critic, self).__init__()
#         self.num_of_observations = num_of_observations
#         #self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size//2)
#         self.fc3 = nn.Linear(hidden_size // 2, 1)

#         self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#         self.gru = nn.GRUCell(32 * 5 * 5, hidden_size)

#         if dropout == None:
#             dropout = 0
#         # self.critic = nn.Sequential(
#         #     nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
#         #     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
#         #     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
#         #     nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
#         #     Flatten(),
#         #     self.fc1,
#         #     nn.Dropout(p=dropout),
#         #     nn.Tanh(),
#         #     self.fc2,
#         #     nn.Dropout(p=dropout),
#         #     nn.Tanh(),
#         #     self.fc3,
#         # )

#         self.critic = nn.Sequential(
#             # self.fc1,
#             # nn.Dropout(p=dropout),
#             # nn.Tanh(),
#             self.fc2,
#             nn.Dropout(p=dropout),
#             nn.Tanh(),
#             self.fc3,
#         )

#         if initialization == "normal":
#             self.critic.apply(init_normal)
#         if initialization == "orthogonal":
#             # self.critic.apply(init_orthogonal)
#             # layer_init(self.fc1)
#             layer_init(self.fc2)
#             layer_init(self.fc3, 1e-3)

#     def forward(self, x, hx):
#         x = F.elu(self.conv1(x))
#         x = F.elu(self.conv2(x))
#         x = F.elu(self.conv3(x))
#         x = F.elu(self.conv4(x))
#         x = x.view(-1, 32 * 5 * 5)
#         hx = self.gru(x, (hx))
#         probs = self.critic(hx)
#         return probs, hx

#     def evaluate(self, state, action, actor, hxactor, hxcritic):

#         state = state.reshape(-1, 1, 80, 80)
#         #print(f"**** ----->{state.shape}")
#         action_probs, hxactor = actor.forward(state, hxactor)
#         #print(f"**** ----->{state.shape}")
#         dist = Categorical(action_probs)

#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         #print(f"**** 3 ----->{state.shape} {hxcritic.shape}")
#         state_value, hxcritic = self.forward(state, hxcritic)
#         #print(f"**** 4 ----->{state.shape}")

#         return action_logprobs, torch.squeeze(state_value), dist_entropy, hxactor, hxcritic


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
                 update_every_j_timestep,
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
                 dropout=None
                 ):
        # Current
        self.actor = Actor(num_of_observations=num_of_observations,
                           posibles_actions=posibles_actions, dropout=dropout, hidden_size=actor_hidden_size).to(device)
        # self.critic = Critic(num_of_observations=num_of_observations,
        #                      posibles_actions=posibles_actions, dropout=dropout, hidden_size=critic_hidden_size).to(device)

        if previousTrainedPolicy:
            self.actor.load_sta32te_dict(previousTrainedPolicy)
        if previousTrainedCritic:
            self.critic.load_state_dict(previousTrainedCritic)

        # Old
        self.actor_old = Actor(num_of_observations=num_of_observations,
                               posibles_actions=posibles_actions, dropout=dropout, hidden_size=actor_hidden_size, initialization=initialization).to(device)
        # self.critic_old = Critic(num_of_observations=num_of_observations,
        #                          posibles_actions=posibles_actions, dropout=dropout, hidden_size=critic_hidden_size, initialization=initialization).to(device)

        self.actor_old.load_state_dict(self.actor.state_dict())
        # self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        p = self.actor.parameters()
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
        self.update_every_j_timestep = update_every_j_timestep

        # Loss
        self.mseLoss = nn.MSELoss()

    def update(self, memory, writer, mini_batch, i_episode, hx):

        # print(memory.states[0].shape)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # print(old_states.shape, old_actions.shape, old_logprobs.shape)

        # print(old_states.shape)
        # print(old_states.shape, old_actions.shape, old_logprobs.shape)

        for e in range(self.k_epochs):

            # Evaluating old actions and values :
            hx = hx.detach()
            logprobs, values, dist_entropy, hx = self.actor.evaluate(
                old_states, old_actions, hx)
            # print(old_states.shape,
            #       old_actions.shape, old_logprobs.shape)
            # GAE Compute rewards
            if self.advantageAlgorithm == "GAE":
                rewards, advantages = self.gae(
                    self.gamma, memory.rewards, memory.is_terminals, values, normalizeAdvantage=self.normalizeAdvantage)
            else:
                rewards, advantages = self.compute_returns(
                    self.gamma,  memory.rewards, memory.is_terminals)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)
            writer.add_scalar("1 - loss/ratios",
                              ratios.mean(), global_step=i_episode)
            writer.add_scalar("1 - loss/advantages",
                              advantages.mean(), global_step=i_episode)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio,
                                1 + self.clip_ratio) * advantages
            policy_loss = torch.min(surr1, surr2)
            value_loss = self.coeficient_value * \
                self.mseLoss(values, rewards)
            #F.smooth_l1_loss(values, rewards)
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
            loss.mean().backward()
            if (self.actorGradientNormalization != 0):
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.actorGradientNormalization)
            self.actor_optimizer.step()
            # self.critic_optimizer.step()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        # self.critic_old.load_state_dict(self.critic.state_dict())

        # clean memory
        memory.clear_memory()
        return hx

    def compute_returns(self, gamma, rewards, is_terminals,):
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

    def gae(self, gamma, rewards, is_terminals,  v, normalizeAdvantage=False):
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
          mini_batch=4,
          betas=None,
          dropout=0,
          saveWithTheSameNameThatTensorboard=False
          ):

    previousTrainedPolicy = None
    if pathForBasePolicyToTrain:
        previousTrainedPolicy = torch.load(pathForBasePolicyToTrain)

    previousTrainedCritic = None
    if pathForBaseCriticToTrain:
        previousTrainedCritic = torch.load(pathForBaseCriticToTrain)

    if mini_batch == None:
        mini_batch = update_every_j_timestep
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

    print(f"start test {tensorboardName}")

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
              update_every_j_timestep=update_every_j_timestep
              )

#    summary(ppo.actor, input_size=(255, 255, 3))
    globlalStep = 0
    startTime = time.time()
    globalStartTime = time.time()
    totalReward = 0

    for i_episode in range(1, max_episode_length+1):
        state = prepro(env.reset()).clone().detach()
        state_4 = []
        reword_4 = []
        for i in range(4):
            state_4.append(state)
            reword_4.append(0)
        episode_reward = 0
        lives = env.unwrapped.ale.lives()
        # print(f"{lives}")
        # if done else hx.detach()  # rnn activation vector
        hx = torch.zeros(1, actor_hidden_size).to(device)
        hxactor = torch.zeros(1, actor_hidden_size).to(device)

        for t in range(1, max_steps + 1):
            timestep += 1
            globlalStep += 1
            # normalize state
            # https://arxiv.org/pdf/2006.05990.pdf
            # if (observationNormalization):
            #     state = normalizeArray(state)
            # Running policy_old:
            x = torch.cat(state_4, 1)
            # print(x.shape)
            action, hx = ppo.actor_old.act(x, memory, hx)
            state, reward, done, _ = env.step(action)
            state = prepro(state).clone().detach()
            state_4.pop(0)
            state_4.append(state)
            reword_4.pop(0)
            reword_4.append(reward)

            newlives = env.unwrapped.ale.lives()
            # print(f"{lives}-{newlives}")
            if newlives < lives:
                done = True
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_every_j_timestep == 0:
                # if done else hx.detach()  # rnn activation vector
                hxactor = ppo.update(memory, writer, mini_batch,
                                     i_episode, hx)

            running_reward += reward
            episode_reward += reward
            totalReward += reward

            writer.add_scalar('2 - lerning rate/lr',
                              get_lr(ppo.actor_optimizer), global_step=i_episode)
            # writer.add_scalar('2 - lerning rate/critic',
            #                   get_lr(ppo.critic_optimizer))

            if render:
                env.render()
            if done:
                hx = hx.detach()
                hxactor = hxactor.detach()
                break

        avg_length += t
        writer.add_scalar('0 - reward/episode',
                          int(episode_reward), global_step=i_episode)
        writer.add_scalar('0 - reward/s_process_obsps',
                          int(globlalStep), global_step=i_episode)

        x = int((running_reward / log_interval))
        if solved_reward <= x:
            print(
                f"-----> goal archived, stop the training at episode:{i_episode}")
            break
        if i_episode % log_interval == 0:
            elapsedTime = time.time() - startTime
            t = getString(elapsedTime)
            globalElapsedTime = time.time() - globalStartTime
            gt = getString(globalElapsedTime)

            print('Episode {} \t avg length: {} \t avg reward: {} \t\t globalStep: {} \t elapse time: {} \t global time: {}'.format(
                i_episode, int(avg_length / log_interval), (running_reward / log_interval), globlalStep, t, gt))
            writer.add_scalar('0 - reward/running reward',
                              (running_reward / log_interval), global_step=i_episode)
            writer.add_scalar('0 - reward/running length',
                              avg_length/log_interval, global_step=i_episode)
            running_reward = 0
            avg_length = 0
            startTime = time.time()

        if i_episode % saveModelsEvery == 0:
            numK = i_episode // 1_000
            torch.save(ppo.actor_old.state_dict(),
                       './model/ppo_{}_policy_{}K.pth'.format(environment_name, numK))
            # torch.save(ppo.critic_old.state_dict(),
            #            './model/ppo_{}_critic_{}K.pth'.format(environment_name, numK))

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
        # torch.save(ppo.critic_old.state_dict(),
        #            './model/ppo_{}_critic_{}.pth'.format(environment_name, tensorboardName))
    else:
        torch.save(ppo.actor_old.state_dict(),
                   './model/ppo_{}_policy_latest.pth'.format(environment_name))
        # torch.save(ppo.critic_old.state_dict(),
        #            './model/ppo_{}_critic_latest.pth'.format(environment_name))

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
                                                                                      advantageAlgorithm=None,
                                                                                      saveModelsEvery=100,
                                                                                      tensorboardName=tensorboardName,
                                                                                      normalizeAdvantage=normalizeAdvantage,
                                                                                      mini_batch=num_mini_batch,
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


def play_latest(environment_name, size, plot=False, observationNormalization=False):

    env = gym.make(environment_name)
    num_of_observations = prodOfTupple(env.observation_space.shape)
    hx = torch.zeros(1, size).to(device)

    posibles_actions = env.action_space.n
    state = env.reset()

    if (observationNormalization):
        state = normalizeArray(state)

    actor = Actor(num_of_observations, posibles_actions, 0,
                  hidden_size=size).to(device)
    actor.load_state_dict(torch.load(
        './model/ppo_BreakoutNoFrameskip-v4_policy_S=128-256,\=0.95,CV=0.5,CE=0.01,NA=True,AN=0,ON=False,DO=0,LR=0.0002,MS=10240,US=512,MB=512.pth'.format(environment_name)))

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
        #state = np.ndarray.flatten(state)
        if (observationNormalization):
            state = normalizeArray(state)
        action, probs, hx = actor.play(state, hx)
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
