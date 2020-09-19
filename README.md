# aidl-2020-spring-team-dani

# Our journey through Reinforcement Learning: From Vanilla Policy Gradients to PPO.

### UPC School

### Authors: Abraham Alcaina, Omar Brid, Bernat Martínez

### Advisor: Dani Fojo

## Index:
  - Introduction
  - Hypothesis
  - Experiment Setup
  - Results
  - Next Steps
  - Conclusions


## Introduction
In this project we will implement different Reinforcement Learning (RL) algorithms in order to solve different environments from [OpenAI Gym](https://gym.openai.com/). We will start implementing Vanilla Policy Gradient algorithm. Then we will implement a version of the Advantadge Actor Critic. Finally we will implement the Policy Proximal Optimization (PPO) algorithm in order to solve the Atari game 'Breakout'.

## Hypothesis
In Reinforcement Learning we have an agent in an unknown environment and this agent can obtain some rewards by interacting with the environment. The agent ought to take actions so as to maximize cumulative rewards. In our case the agent will try to solve an environment from OpenAI Gym. Below we list the ones which we have solved:
- *Cart Pole.* A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
- *Atari Lunar Lander.* Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
- *Atari Breakout.* Maximize your score in the Atari 2600 game Breakout. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k frames, where k is uniformly sampled from {2,3,4}.

<p align="center">
  <img src="https://lilianweng.github.io/lil-log/assets/images/RL_illustration.png">
</p>

Below we find some key concepts in Reinforcement Learning we will be using in this document:
- *Model.* The model is a descriptor of the environment. With the model, we can learn or infer how the environment would interact with and provide feedback to the agent.
- *Policy.* Policy, as the agent’s behavior function π, tells us which action to take in state s. It is a mapping from state s to action a and can be either deterministic or stochastic.
- *Value Function.* Value function measures the goodness of a state or how rewarding a state or an action is by a prediction of future reward. The future reward, also known as return, is a total sum of discounted rewards going forward.

All the algorithms we will be reviewing in this project are on-policy. This means that they se the deterministic outcomes or samples from the target policy to train the algorithm.

The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relative simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards. Let's review the three RL algorithms we implemented for this project.

### Vanilla Policy Gradient
The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.

VPG explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found.

The gradient is computed as follows:
<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/math/ada1266646d71c941e77e3fd41bba9d92d06b7c2.svg">
</p>

where A is the advantage function for the current policy, π is the policy, 'a' is the action and 's' is the state.


### Advantage Actor Critic (A2C)

### Policy Proximal Optimization (PPO)


## Experiment Setup
PyTorch, OpenAI Gym, TensorBoard, GitHub, 

## Results
Results for the different envs/algorithms.

## Next Steps


## Conclusions
