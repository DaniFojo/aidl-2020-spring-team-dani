import ppo
if __name__ == "__main__":
    train = True
    actor_size = 64
    critic_size = 256
    name = "BreakoutNoFrameskip-v4"
    observationNormalization = True
    if train:
        env = ppo.multitrain(name,
                             gammas=[0.99],
                             clip_ratios=[0.1],
                             lrs=[2e-4],
                             # vf_lr=[2e-3],
                             k_epochss=[4],
                             update_every_j_timesteps=[128],
                             # 128 * 80_000 ~ 10 Millions 120K ~ 15M
                             max_episode_lengths=[10_000],
                             max_steps=[1024],
                             critic_sizes=[256],
                             actor_sizes=[128],
                             # render=False,
                             # random_seed=1,
                             solved_reward=200,
                             # Normalize observations of the environment
                             observationNormalizations=[False],
                             # Clips gradient norm of an iterable of parameters. max norm of the gradients
                             actorGradientNormalizations=[0.5],
                             normalizeAdvantages=[True],
                             # initialization="orthogonal",  # normal = normal distribution, None
                             # advantageAlgorithm="GAE",  # None = use A2C reward calculation
                             # pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
                             # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
                             coeficient_entropys=[0.001],  # 0.001
                             coeficient_values=[0.5],  # 0.5
                             lmbdas=[0.95],
                             epss=[1e-5],
                             # weight_decay=5e-4,
                             # saveModelsEvery=5_000,
                             betass=[(0.9, 0.999)],
                             dropouts=[0],
                             num_mini_batchs=[4],
                             )
    ppo.play_latest(name, 128, plot=False,
                    observationNormalization=observationNormalization)
