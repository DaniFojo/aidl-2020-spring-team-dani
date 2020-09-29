import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import collections
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
device = torch.device("cuda")

num_of_observations = 4
hidden_perceptrons = 16
posibles_actions = 2

lr = 0.001
weight_decay = 0.000_5
number_of_episodes = 1_000
max_episode_length = 200
success_score = 195

layers = collections.OrderedDict([
    ("inputs", nn.Linear(num_of_observations, hidden_perceptrons)),
    ("relu_i", nn.ReLU()),
    ("hidden_1", nn.Linear(hidden_perceptrons, hidden_perceptrons)),
    ("relu_h1", nn.ReLU()),
    ("hidden_2", nn.Linear(hidden_perceptrons, posibles_actions)),
    ("relu_h2", nn.ReLU()),
    ("classification", nn.Linear(posibles_actions, posibles_actions)),
    ("softmax", nn.Softmax()),
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
criterion = nn.CrossEntropyLoss()
number_of_consecutive_success = 0

losses = []
accs = []

for i_episode in range(number_of_episodes):
    observation = env.reset()
    score = 0
    for t in range(max_episode_length):
        # env.render()

        input = F.Tensor(observation).to(device)

        policy.eval()
        predictionTorch = policy(input)
        _, pred = predictionTorch.max(0)
        action = pred.item()
        observation, reward, done, info = env.step(action)
        score += reward

        # TODO rething the target function
        if (reward == 1.0):
            if (action == 0):
                target = torch.LongTensor([1, 0])
            else:
                target = torch.LongTensor([0, 1])
        else:
            if (action == 0):
                target = torch.LongTensor([0, 1])
            else:
                target = torch.LongTensor([1, 0])

        policy.train()
        optimizer.zero_grad()
        # TODO REVIEW this loss
        predictionTorch = predictionTorch.repeat(2).reshape(2, 2)
        loss = criterion(predictionTorch, target.to(device))
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if done:

            print("Episode finished after {} timestepms. Loss: {:.6f}".format(
                t+1, loss.item()))
            break

    if score > success_score:
        print("Challenge archived")
        number_of_consecutive_success += 1
    else:
        number_of_consecutive_success = 0
    if (number_of_consecutive_success > 99):
        print("Episode %i: We have a model!!!" % (i_episode, ))
        torch.save(policy.state_dict(), "./model/cart-pole.pth")
        break

plt.plot(losses)
plt.show()
env.close()
