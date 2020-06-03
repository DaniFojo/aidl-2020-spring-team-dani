import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('board/cart-pole')

env = gym.make('CartPole-v1')
device = torch.device("cuda")

num_of_observations = 4
hidden_perceptrons = 16
posibles_actions = 2

lr = 0.000_1
weight_decay = 0.000_05
number_of_episodes = 150_000
max_episode_length = 100
success_score = 195

layers = collections.OrderedDict([
    ("inputs", nn.Linear(num_of_observations, hidden_perceptrons)),
    ("relu_i", nn.ReLU()),
    ("hidden_1", nn.Linear(hidden_perceptrons, hidden_perceptrons)),
    ("relu_h1", nn.ReLU()),
    ("classification", nn.Linear(hidden_perceptrons, posibles_actions))
])
policy = nn.Sequential(layers).to(device)

"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
"""

optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
number_of_consecutive_success = 0
scores = []

for i_episode in range(number_of_episodes):
    observation = env.reset()
    ep_rews = []
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths
    score = 0
    for t in range(max_episode_length):
        # env.render()

        # collect trajectory
        batch_obs.append(observation.copy())
        input = F.Tensor(observation).to(device)

        predictionTorch = policy(input)
        _, pred = predictionTorch.max(0)
        action = pred.item()
        observation, reward, done, info = env.step(action)
        batch_acts.append(predictionTorch.tolist())
        ep_rews.append(reward)
        score = max(sum(ep_rews), score)
        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_weights += [ep_ret] * ep_len

            ep_rews = []

    if i_episode % 100 == 0:
        print("Episode: %i, score: %i" % (i_episode, score))
    policy.train()

    l = F.Tensor(batch_acts).to(device)
    # l = policy(F.Tensor(batch_obs).to(device))

    # which gives the expected return if you start in state s, take an arbitrary action a
    Q_pi = 0.5
    # which gives the expected return if you start in state s and always act according to policy \pi:
    # V^{\pi}(s) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}
    V_pi = sum(np.asarray(batch_weights)) / len(batch_weights)
    # A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
    A_pi = Q_pi - V_pi

    z = (1/max_episode_length) * sum(np.asarray(batch_weights))
    # z aumenta si el score es mayour

    A_hat = F.Tensor(np.asarray(batch_weights) *
                     A_pi * z).repeat(2).view(100, 2).to(device)
    A_hat.requires_grad = True

    # loss = criterion(l, target)
    loss = l.log_softmax(dim=1).mul(A_hat)
    loss = criterion(loss, A_hat)
    # weight = F.Tensor(batch_weights).to(device).repeat(2).reshape(-1, 2)
    # loss = -(l * weight).mean()
    # c = F.Tensor((Q_pi - np.asarray(batch_weights)) **                 2).to(device).repeat(2).reshape(-1, 2)
    # target = l.mul(c).log()
    # loss = -(target * weight).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    policy.eval()

    writer.add_scalar('score', score)
    scores.append(score)

    if (number_of_consecutive_success > 99):
        print("Episode %i: We have a model!!!" % (i_episode, ))
        torch.save(policy.state_dict(), "./model/cart-pole.pth")
        break
writer.close()
env.close()
