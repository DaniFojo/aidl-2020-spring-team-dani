import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
writer = SummaryWriter('board/cart-pole')

env = gym.make('CartPole-v0')
device = torch.device("cuda")

num_of_observations = 4
hidden_perceptrons = 256
posibles_actions = 2

gamma = 0.99
lr = 0.000_1
weight_decay = 0.000_05
number_of_episodes = 1_000
max_episode_length = 500
success_score = 195

layers = collections.OrderedDict([
    ("inputs", nn.Linear(num_of_observations, hidden_perceptrons)),
    ("relu_i", nn.ReLU()),
    ("hidden_1", nn.Linear(hidden_perceptrons, hidden_perceptrons)),
    ("relu_h1", nn.ReLU()),
    ("dropout", nn.Dropout(0.5)),
    ("classification", nn.Linear(hidden_perceptrons, posibles_actions)),
    ("softmax", nn.Softmax())
])
policy = nn.Sequential(layers).to(device)


"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart'shttps://meet.google.com/muk-hgdy-zvy?authuser=0
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
#criterion = nn.MSELoss()
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
    batch_logprob = []
    score = 0
    done = False
    for t in range(max_episode_length):
        if (i_episode > 900):
            env.render()

        # collect trajectory
        batch_obs.append(observation.copy())
        input = F.Tensor(observation).to(device)
        input = input.unsqueeze(0)

        p = policy(input)
        dist = Categorical(probs=p)
        predictionTorch = dist.sample()
        batch_logprob.append(dist.log_prob(predictionTorch))

        action = predictionTorch.item()

        batch_acts.append(action)
        observation, reward, done, info = env.step(action)
        ep_rews.append(reward)

        score = sum(ep_rews)
        scores.append(score)
        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_weights += [ep_ret] * ep_len
            break

    if i_episode % 20 == 0:
        av = sum(scores)/len(scores)
        writer.add_scalar('Score every 20 episodes ', av)
        print("Episode: %i, score: %i, average: %f" % (
            i_episode, score, av))
        scores = []
    else:
        scores.append(score)
    writer.add_scalar('score', score)

    policy.train()

    # Reward total
    G = 0
    policy_loss = []
    returns = []

    for r in ep_rews[::-1]:  # dirrecio contraria
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    # Normalize the rewards
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    obs = F.Tensor(batch_obs).to(device)
    acts = F.Tensor(batch_acts).to(device)
    weights = F.Tensor(batch_weights).to(device)

    batch_loss = []
    for log_prob, r in zip(batch_logprob, returns):
        batch_loss.append(-log_prob * r)

    logp = F.Tensor(batch_logprob).to(device)
    loss = torch.stack(batch_loss).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    policy.eval()
    scores.append(score)

    if (number_of_consecutive_success > 99):
        print("Episode %i: We have a model!!!" % (i_episode, ))
        torch.save(policy.state_dict(), "../model/cart-pole.pth")
        break
writer.close()
env.close()

torch.save(policy.state_dict(), "./model/cart-pole.pth")
