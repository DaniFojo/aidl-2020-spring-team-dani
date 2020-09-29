# Our journey through Reinforcement Learning: From Vanilla Policy Gradients to PPO.

### UPC School

### Authors: Abraham Alcaina, Omar Brid, Bernat Martínez

### Advisor: Dani Fojo

## Index

- Introduction
- Hypothesis
- Experiment Setup
- Results
- Execution Instructions
- Conclusions

## Introduction

In this project we will implement different Reinforcement Learning (RL) algorithms in order to solve different environments from [OpenAI Gym](https://gym.openai.com/). We will start implementing Vanilla Policy Gradient algorithm. Then we will implement a version of the Advantadge Actor Critic. Finally we will implement the Policy Proximal Optimization (PPO) algorithm in order to solve the Atari game 'Breakout'.

## Hypothesis

In Reinforcement Learning we have an agent in an unknown environment and this agent can obtain some rewards by interacting with the environment. The agent ought to take actions so as to maximize cumulative rewards. In our case the agent will try to solve an environment from OpenAI Gym. Below we list the ones which we have solved:

- _Cart Pole._ A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
- _Lunar Lander._ Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
- _Atari Breakout._ Maximize your score in the Atari 2600 game Breakout. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes. Each action is repeatedly performed for a duration of k frames, where k is uniformly sampled from {2,3,4}.

<p align="center">
  <img src="https://lilianweng.github.io/lil-log/assets/images/RL_illustration.png">
</p>

Below we find some key concepts in Reinforcement Learning we will be using in this document:

- _Model._ The model is a descriptor of the environment. With the model, we can learn or infer how the environment would interact with and provide feedback to the agent.
- _Policy._ Policy, as the agent’s behavior function π, tells us which action to take in state s. It is a mapping from state s to action a and can be either deterministic or stochastic.
- _Value Function._ Value function measures the goodness of a state or how rewarding a state or an action is by a prediction of future reward. The future reward, also known as return, is a total sum of discounted rewards going forward.

All the algorithms we will be reviewing in this project are on-policy. This means that they se the deterministic outcomes or samples from the target policy to train the algorithm.

The goal of Reinforcement Learning (RL) is to learn a good strategy for the agent from experimental trials and relative simple feedback received. With the optimal strategy, the agent is capable to actively adapt to the environment to maximize future rewards.

Note that all the environments that we are going to solve are discrete, which means that the action space allows a fixed range of non-negative numbers. In case we wanted to develop algorithms to solve continuous environments, we should take a look to the [Bellman Equations](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#bellman-equations).

Let's review the three RL algorithms we implemented for this project.

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

In Advantage Actor Critic (A2C) we introduce an advantage function, which will tell us the improvement compared to the average the action taken at that state is. In other words, this function calculates the extra reward the agent gets if I take this action. The extra reward is that beyond the expected value of that state. The Advantage functions is as follows: **Aπ(s,a)=Qπ(s,a)−Vπ(s)**.

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

### GPU

![GPU](/images/GPU.png)

### CPU

![CPU](/images/CPU.JPG)

## Results

The goal of this project is to achieve solving Atari games from OpenAI Gym applying the PPO algorithm. In order to do that, we first implemented VPG, then A2C and finally PPO in order to understand how RL algorithms work. You can find the different implementations in this repository.

In this section we will analyze the results of our PPO implementation for different OpenAI Gym environments.

### Cart Pole

In this implementation we used an Actor-Critic architecture as the agent for the PPO. "Considered solved when the average return is greater than or equal to 195.0 over 100 consecutive trials."

![Cart Pole](/images/tb_cart_pole_ppo.png)
![Cart Pole](/images/cartpole.gif)

### Lunar Lander

In this implementation we used an Actor-Critic architecture as the agent for the PPO. "Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt."

![Lunar Lander](/images/tb_lunar_lander_ppo.png)
![Lunar Lander](/images/lunar.gif)

### Atari Breakout

In this case we are just using an Actor. The input to the model are four stacked frames (in order to get the movement of the screen) and the actor is built with convolutional networks.

![Breakout](/images/tb_breakout_ppo_100k.png)
![Breakout](/images/tb_breakout_ppo_200k.png)
![Breakout](/images/tb_breakout_ppo_300k.png)
![Breakout](/images/breakout.gif)

### Other solved Atari environments

With our PPO implementation we were able to solve other atari environments as Pong or Space Invaders.

![Pong](/images/pong.gif)

![Space Invaders](/images/tb_space_invaders_ppo.png)
![Space Invaders](/images/invaders.gif)

## Execution Instructions

### Dependencies

In order to run experiments with our PPO implementation, the needed dependencies need to be installed. Running the shell script [install.sh](https://github.com/DaniFojo/aidl-2020-spring-team-dani/blob/master/install.sh) all dependencies will be installed.

### Running Experiments

In this repository we will find different different execution scripts, depending on the PPO implementation and the environment we want to try. Our final results are based on the async implementation, so we will focus on the scripts located in [this folder](https://github.com/DaniFojo/aidl-2020-spring-team-dani/tree/master/src/ppo/async).

We have different execution scripts for each environment. In order to test different hyperparameters for training, the 'train' flag has to be 'True' and then we can change the hyperparameters. In case we want to test a trained model, we will execute the script with 'train=False' and in the 'model_file' parameter we will locate the model that we want to test.
If you want to use a pre-trained model you can use the models located in [this folder](https://github.com/DaniFojo/aidl-2020-spring-team-dani/tree/master/model)

In case we want to train a model to play in a different Atari environment, we can change it in the 'game' parameters, where we will put the name of the environment that we want to test.

## Conclusions

It has been a very challenging project and we have learned a lot. One of the first issues we faced is the hardware limitation. It took a lot of time to train the models, and the batch size was also limited to the hardware compute power.

We have observed that in PPO with A2C the models were training successfully in simple environments, but when complexity was increased (Atari) we needed more episodes and we didn't get a model avble to solve the environments. This was solved using the environments frames (images) instead of the RAM as input data. Using this approach we observed that training finishes for less episodes in Pong and Space Invaders, but for Breakout the model is not able to finish the game. We suspect that this could be overfitting, and a way to solve that could be increasing the batch size, but this takes us back to the hardware limitation.

We have seen that it is very important to monitor the performance of the models with tools like TensorBoard. All the tests we made took us to other monitoring techniques like observing how each hyperparameter affected the result. Both monitoring and testing different techniques led us to the final algorithm, with which we would be able to train models able to solve Atari environments.

During this project we have had the opportunity of learning how to implement Reinforcement Learning algorithms gradually, starting with Vanilla Policy Gradients, then Advantage Actor Critic and finallty PPO, being able to train models that are able to solve different Atari games. It has been very challenging, specially with PPO, as there are a lot of factors that have to be taken into account :hyperparameters, different architectures, read different papers in order to learn techniques that could improve the performance of our model...

Deep learning has a huge community behind it, which makes it easier to find a solution to your problem. However, bulding deep learning models can lead you to a very specific problem to your case and it won't be that easy to solve it.

## References

[Learning from the memory of Atari 2600](https://arxiv.org/pdf/1605.01335.pdf)

[High-dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)

[What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf)

[Stabilizing Transformers For Reinforcement Learning](https://arxiv.org/pdf/1910.06764.pdf)

[RL — Proximal Policy Optimization (PPO) Explained | by Jonathan Hui](https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)

[A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)

[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/)

[Proximal Policy Optimization (PPO)](https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26)

[RL — Proximal Policy Optimization (PPO) Explained](https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)

[Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)

[Cartpole - Introduction to Reinforcement Learning (DQN - Deep Q-Learning)](https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288)

[Docs » Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html#documentation-pytorch-version)

[RL — Policy Gradient Explained](https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146)

[RL — Policy Gradients Explained (Part 2)](https://medium.com/@jonathan_hui/rl-policy-gradients-explained-advanced-topic-20c2b81a9a8b)

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

[Reinforcement learning algorithms with Generalized Advantage Estimation](https://github.com/bsivanantham/GAE)

[Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6)

[Proximal Policy Optimization Tutorial (Part 2/2: GAE and PPO loss)](https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8)

[Applications of Reinforcement Learning in Real World](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)

[Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

[Playing Atari with Deep Reinforcement Learning](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning)

[UC Berkeley Reward-Free RL Beats SOTA Reward-Based RL](https://syncedreview-com.cdn.ampproject.org/v/s/syncedreview.com/2020/09/21/uc-berkeley-reward-free-rl-beats-sota-reward-based-rl/amp/?usqp=mq331AQFKAGwASA%3D&amp_js_v=01#referrer=https%3A%2F%2Fwww.google.com&amp_tf=De%20%251%24s&ampshare=https%3A%2F%2Fsyncedreview.com%2F2020%2F09%2F21%2Fuc-berkeley-reward-free-rl-beats-sota-reward-based-rl%2F)

[From 0 to 200 - lessons learned from solving Atari Breakout with Reinforcement Learning](http://blog.jzhanson.com/blog/rl/project/2018/05/28/breakout.html)

[ACCELERATED METHODS FOR DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1803.02811.pdf)

## Presentation

[Presentation (PPTX 2020-9-29)](/presentation/aidl_presentation_breakout.pptx)

[Presentation (PDF 2020-9-29)](/presentation/aidl_presentation_breakout.pdf)

[Presentation (latest)](https://docs.google.com/presentation/d/15Fdu86SqXk07pGEvxz5FZPpUEyuPUffezvqd6eWLsdA/edit#slide=id.p)
