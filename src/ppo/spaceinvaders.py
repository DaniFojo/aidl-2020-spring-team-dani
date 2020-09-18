import ppoimage
import ppo
if __name__ == "__main__":
    train = True
    actor_size = 128
    critic_size = 256
    name = "SpaceInvaders-v4"
    observationNormalization = True
    if train:
        env = ppoimage.multitrain(name,
                                  gammas=[0.99],
                                  clip_ratios=[0.1],
                                  lrs=[2e-4],
                                  # vf_lr=[2e-3],
                                  k_epochss=[4],
                                  update_every_j_timesteps=[128],
                                  # 128 * 80_000 ~ 10 Millions 120K ~ 15M
                                  max_episode_lengths=[10_000],
                                  max_steps=[1024*10],
                                  critic_sizes=[256],
                                  actor_sizes=[256],
                                  # render=False,
                                  # random_seed=1,
                                  solved_reward=500,
                                  # Normalize observations of the environment
                                  observationNormalizations=[False],
                                  # Clips gradient norm of an iterable of parameters. max norm of the gradients
                                  actorGradientNormalizations=[0],
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

ppoimage.play_latest(name, 128, plot=True,
                     observationNormalization=False)
