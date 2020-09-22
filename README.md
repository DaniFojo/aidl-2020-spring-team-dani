# Our journey through Reinforcement Learning: From Vanilla Policy Gradients to PPO.

### UPC School

### Authors: Abraham Alcaina, Omar Brid, Bernat Martínez

### Advisor: Dani Fojo

## Index:
  - Introduction
  - Hypothesis
  - Experiment Setup
  - Results
  - Execution Instructions
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
The idea of Actor Critic is to use two neural networks. The actor is is a policy function that controls how our agent acts. The critic is a value function that measures how good these actions are.

Both Actor and Critic run in parallel. As we have two models that need to be trained, we have two set of weights:
<p align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/1*KlX2-kNXRYLAYpdnI8VPiA.png">
</p>

In Advantage Actor Critic (A2C) we introduce an advantage function, which will tell us the improvement compared to the average the action taken at that state is. In other words, this function calculates the extra reward the agent gets if I take this action. The extra reward is that beyond the expected value of that state. The Advantage functions is as follows **Aπ(s,a)=Qπ(s,a)−Vπ(s)**.

Recall the new update equation, replacing the discounted cumulative award from vanilla policy gradients with the Advantage function:

<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/math/ada1266646d71c941e77e3fd41bba9d92d06b7c2.svg">
</p>

On each learning step, we update both the Actor parameter and the Critic parameter.

### Policy Proximal Optimization (PPO)
In many Policy Gradient Methods, policy updates are unstable because of larger step size, which leads to bad policy updates and when this new bad policy is used for learning then it leads to even worse policy. Moreover, many learning methods learn from current experience and discard the experiences after gradient updates. This makes the learning process slow as a neural net takes lots of data to learn, which is inneficient regardding data. PPO overcomes those issues.

Now instead of the log of current policy, we will be taking the ratio of current policy and old policy.

<p align="center">
  <img src="https://miro.medium.com/max/377/1*Zgk2-ZhrDPzFI0ZYCxHc4A.png">
</p>

We will be also clipping the ratio and will the minimum of the two i.e b/w clipped and unclipped.

<p align="center">
  <img src="https://miro.medium.com/max/627/1*2-rWCA-oqVxsw-MnVd_lKQ.png">
</p>

where epsilon is a hyperparameter. The motivation for this objective is as follows. The first term inside the min is L^CPI . The second term, modifies the surrogate objective by clipping the probability ratio, which removes the incentive for moving rt outside of the interval [1 − eps, 1 + eps]. Finally, we take the minimum of the clipped and unclipped objective, so the final objective is a lower bound (i.e., a pessimistic bound) on the unclipped objective. With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse.


<p align="center">
  <img src="https://miro.medium.com/max/788/1*VN01Obh5VyJ6QuA0qfyq6w.png">
</p>

## Experiment Setup
We developed our experiments using Python3 and the PyTorch library for the Deep Learning models (among others). 

In our experiments we developed different algorithms in order to solve the following environments from OpenAI:
- [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)
- [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)
- [Breakout-ram-v0](https://gym.openai.com/envs/Breakout-ram-v0/)
- [Breakout-v0](https://gym.openai.com/envs/Breakout-v0/)

We used [TensorBoard](https://www.tensorflow.org/tensorboard/) in order to see the performance of our models.

The specifications of the machine where we run our experiments are as follows:

GPU:



CPU:



PyTorch, OpenAI Gym, TensorBoard, GitHub, 

## Results
Results for the different envs/algorithms.

## Execution Instructions

## Next Steps


## Conclusions

## References
https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
https://arxiv.org/pdf/1707.06347.pdf
https://openai.com/blog/openai-baselines-ppo/
https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
https://gym.openai.com/envs/
