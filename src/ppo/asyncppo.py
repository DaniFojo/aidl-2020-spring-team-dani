import multiprocessing
import multiprocessing.connection
import random
import cv2
import gym
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from typing import Dict, List
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getString(date):
    hours, rem = divmod(
        date, 3600)
    minutes, seconds = divmod(
        rem, 60)
    t = "{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds)
    return t


class Game:

    def __init__(self, seed: int, game):
        self.env = gym.make(game)
        self.env.seed(seed)

        self.obs_4 = np.zeros((4, 84, 84))
        self.rewards = []
        self.lives = 0

    def step(self, action):

        reward = 0.
        done = None

        # run for 4 steps
        for i in range(4):
            obs, r, done, info = self.env.step(action)

            reward += r

            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives:
                done = True
                break

        obs = self._process_obs(obs)
        self.rewards.append(reward)

        if done:
            episode_info = {"reward": sum(
                self.rewards), "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None

            # push it to the stack of 4 frames
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = obs

        return self.obs_4, reward, done, episode_info

    def reset(self):
        obs = self.env.reset()

        obs = self._process_obs(obs)
        for i in range(4):
            self.obs_4[i] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4

    def play(self, action):
        self.env.render()
        obs, r, done, info = self.env.step(action)
        obs = self._process_obs(obs)
        self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
        self.obs_4[-1] = obs

        obsT = obs_to_torch(self.obs_4).reshape(-1, 4, 84, 84).to(device)
        return obsT, r, done

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


def worker_process(remote: multiprocessing.connection.Connection, seed: int, game):
    game = Game(seed, game)

    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, seed, game):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=worker_process, args=(parent, seed, game))
        self.process.start()


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=8, stride=4)
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=512, out_features=4)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = F.relu(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value

    def play(self, obs):
        with torch.no_grad():
            h = F.relu(self.conv1(obs))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            h = h.reshape((-1, 7 * 7 * 64))
            h = F.relu(self.lin(h))
            pi = self.pi_logits(h).squeeze()
            _, action = pi.max(0)
        return action.item(), pi


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    # scale to `[0, 1]`
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class Main:
    def __init__(self,
                 gamma,
                 lmda,
                 n_update,
                 n_epochs,
                 n_workers,
                 worker_steps,
                 n_mini_batch,
                 lr,
                 coeficient_entropy,
                 coeficient_vf,
                 actorGradientNormalizations,
                 game,
                 log_interval,
                 ):

        self.game = game
        self.gamma = gamma
        self.lamda = lmda
        self.updates = n_update
        self.epochs = n_epochs
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_mini_batch = n_mini_batch
        self.lr = lr
        self.coeficient_entropy = coeficient_entropy
        self.coeficient_vf = coeficient_vf
        self.actorGradientNormalizations = actorGradientNormalizations
        self.log_interval = log_interval
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)

        # create workers
        self.workers = [Worker(47 + i, self.game)
                        for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        self.model = Model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def sample(self) -> (Dict[str, np.ndarray], List):

        rewards = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps,
                        4, 84, 84), dtype=np.uint8)
        log_pis = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)

        # sample `worker_steps` from each worker
        for t in range(self.worker_steps):
            with torch.no_grad():
                obs[:, t] = self.obs
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                self.obs[w], rewards[w, t], done[w,
                                                 t], info = worker.child.recv()
                if info:
                    self.writer.add_scalar(
                        '0 - reward/reward', info['reward'], global_step=self.i_episode)
                    self.writer.add_scalar(
                        '0 - reward/length', info['length'], global_step=self.i_episode)

        # calculate advantages
        advantages = self._calc_advantages(done, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'advantages': advantages
        }

        #  flatten
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:

        advantages = np.zeros(
            (self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage
            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages

    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                loss = self._calc_loss(clip_range=clip_range,
                                       samples=mini_batch)

                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.actorGradientNormalizations)
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        sampled_return = samples['values'] + samples['advantages']
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        pi, value = self.model(samples['obs'])

        log_pi = pi.log_prob(samples['actions'])
        ratio = torch.exp(log_pi - samples['log_pis'])

        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                              max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2,
                            (clipped_value - sampled_return) ** 2)
        vf_loss = self.coeficient_vf * vf_loss.mean()

        loss = -(policy_reward - self.coeficient_vf * vf_loss +
                 self.coeficient_entropy * entropy_bonus)

        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) >
                         clip_range).to(torch.float).mean()

        self.writer.add_scalar('1 - loss/policy_reward', policy_reward,
                               global_step=self.i_episode)
        self.writer.add_scalar('1 - loss/vf_loss', vf_loss,
                               global_step=self.i_episode)
        self.writer.add_scalar('1 - loss/entropy_bonus', entropy_bonus,
                               global_step=self.i_episode)
        self.writer.add_scalar('1 - loss/kl_div', approx_kl_divergence,
                               global_step=self.i_episode)
        self.writer.add_scalar('1 - loss/clip_fraction', clip_fraction,
                               global_step=self.i_episode)

        return loss

    def run_training_loop(self):
        self.tensorboardName = f"G={self.game}=,gamma={self.gamma},\={self.lamda},NU={self.updates},NE={self.epochs},NW={self.n_workers},WS={self.worker_steps},NMB={self.n_mini_batch},LR={self.lr},CE={self.coeficient_entropy},CVF={self.coeficient_vf},AGN={self.actorGradientNormalizations}"
        globalStartTime = time.time()
        startTime = time.time()
        self.i_episode = 0
        self.writer = SummaryWriter(
            f'board/ppo/async/{self.game}/{self.tensorboardName}')
        for update in range(self.updates):

            progress = update / self.updates
            self.i_episode = self.i_episode + 1

            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)
            samples = self.sample()

            self.train(samples, learning_rate, clip_range)

            if (update + 1) % 1_000 == 0:
                d = datetime.now().strftime("%Y%m%d%H%M")
                torch.save(self.model.state_dict(),
                           './model/ppo_{}_{}K_{}.pth'.format(self.game, ((update + 1) // 1_000), d))
            if update % self.log_interval == 0:
                elapsedTime = time.time() - startTime
                t = getString(elapsedTime)
                globalElapsedTime = time.time() - globalStartTime
                gt = getString(globalElapsedTime)
                print('Episode {}  \t elapse time: {} \t global time: {}'.format(
                    self.i_episode, t, gt))
                startTime = time.time()

        torch.save(self.model.state_dict(),
                   './model/ppo_{}_final.pth'.format(self.game))

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))


def play_latest(game, plot=False):
    play(game, f"ppo_{game}_final.pth", plot=plot)


def play(game, model_file, plot=False):
    seed = random.randint(0, 1e10)
    game = Game(seed, game)
    model = Model().to(device)
    model.load_state_dict(torch.load(
        f"./model/{model_file}"))
    obs = obs_to_torch(game.reset()).reshape(-1, 4, 84, 84).to(device)
    done = False
    i = 0
    total_reward = 0
    posibles_actions = game.env.action_space.n
    if plot:
        y_pos = np.arange(posibles_actions)
        probLegend = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.xticks(y_pos, y_pos)
        plt.ylabel('Probs')
        plt.xlabel('Action')

    while (not done):
        action, probs = model.play(obs)
        obs, reward, done = game.play(action)
        i = i + 1
        if plot:
            probs = probs.detach().cpu().numpy()
            ax = plt.bar(y_pos, probs, align='center',
                         alpha=0.5, color=(0.2, 0.4, 0.6, 0.6))
            plt.title(f'Step - {i}, Action choosed: {action}')
            plt.pause(0.002)
            ax.remove()
        else:
            print(f'step {i} \t action {action}')
        total_reward += reward

    print("total reward {}".format(total_reward))
    if plot:
        plt.show()
        plt.close()
    env.close()
