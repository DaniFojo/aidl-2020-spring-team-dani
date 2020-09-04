import ppo
if __name__ == "__main__":
    train = True
    actor_size = 64
    critic_size = 256
    name = "Breakout-ram-v4"
    observationNormalization = True
    if train:
        env = ppo.multitrain(name,
                             gammas=[0.99],
                             clip_ratios=[0.1],
                             lrs=[1e-3],
                             # vf_lr=[2e-3],
                             # k_epochs=4,
                             update_every_j_timesteps=[1000],
                             # 128 * 80_000 ~ 10 Millions 120K ~ 15M
                             max_episode_lengths=[5_000],
                             max_steps=[500],
                             critic_sizes=[256],
                             actor_sizes=[32, 128],
                             # render=False,
                             # random_seed=1,
                             solved_reward=249,
                             # Normalize observations of the environment
                             observationNormalizations=[True],
                             # Clips gradient norm of an iterable of parameters. max norm of the gradients
                             actorGradientNormalizations=[0, 5],
                             normalizeAdvantages=[False],
                             # initialization="orthogonal",  # normal = normal distribution, None
                             # advantageAlgorithm="GAE",  # None = use A2C reward calculation
                             # pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
                             # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
                             coeficient_entropys=[0.01],  # 0.001
                             coeficient_values=[0.5],  # 0.5
                             lmbdas=[0.99],
                             epss=[1e-5],
                             # weight_decay=5e-4,
                             # saveModelsEvery=5_000,
                             betass=[(0.9, 0.999)],
                             dropouts=[0, 0.5],
                             )

# Train only one model

    #     env = ppo.train(environment_name=name,
    #                     gamma=0.99,
    #                     clip_ratio=0.2,
    #                     pi_lr=2e-3,
    #                     vf_lr=2e-3,
    #                     k_epochs=4,
    #                     update_every_j_timestep=600,
    #                     max_episode_length=15_000,  # 128 * 80_000 ~ 10 Millions 120K ~ 15M
    #                     max_steps=300,
    #                     critic_hidden_size=critic_size,
    #                     actor_hidden_size=actor_size,
    #                     render=False,
    #                     random_seed=1,
    #                     solved_reward=249,
    #                     # Normalize observations of the environment
    #                     observationNormalization=observationNormalization,
    #                     # Clips gradient norm of an iterable of parameters. max norm of the gradients
    #                     actorGradientNormalization=0,
    #                     normalizeAdvantage=False,
    #                     initialization="orthogonal",  # normal = normal distribution, None
    #                     advantageAlgorithm="GAE",  # None = use A2C reward calculation
    #                     # pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
    #                     # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
    #                     coeficient_entropy=0.01,  # 0.001
    #                     coeficient_value=0.5,  # 0.5
    #                     lmbda=0.996,
    #                     eps=1e-5,
    #                     weight_decay=5e-4,
    #                     saveModelsEvery=5_000,
    #                     # betas=(0.9, 0.999),
    #                     dropout=0.5,
    #                     tensorboardName="Base-32,DO=0.5,ON=True,AN=0,Upd=600,LR=2e-3,\=.996,WD=5e-4,CP=.2,15K",)

    # ppo.play_latest(name, actor_size, plot=False,
    #                 observationNormalization=observationNormalization)
