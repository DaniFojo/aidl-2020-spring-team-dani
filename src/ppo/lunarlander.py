import ppo
if __name__ == "__main__":
    train = True
    name = "LunarLander-v2"
    actor_size = 32
    critic_size = 256
    if train:
        env = ppo.train(environment_name=name,
                        gamma=0.99,
                        clip_ratio=0.1,
                        pi_lr=2e-3,
                        vf_lr=2e-3,
                        k_epochs=4,
                        update_every_j_timestep=32,
                        max_episode_length=2_000,
                        max_steps=500,
                        critic_hidden_size=critic_size,
                        actor_hidden_size=actor_size,
                        render=False,
                        random_seed=1,
                        solved_reward=249,
                        observationNormalization=False,
                        actorGradientNormalization=0,
                        normalizeAdvantage=False,
                        initialization="orthogonal",  # normal = normal distribution, None
                        advantageAlgorithm="GAE",  # None = use A2C reward calculation o "GAE"
                        # pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
                        # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
                        coeficient_entropy=0.01,
                        coeficient_value=0.5,
                        lmbda=0.99,
                        eps=1e-5,
                        weight_decay=5e-4,
                        tensorboardName="base-32,lamda=0.99,LR=2e-3,Upd=1K,ON=False,AN=False",
                        mini_batch=4
                        )

    ppo.play_latest(name, actor_size, plot=True)

    #    env = ppo.train(environment_name=name,
    # gamma=0.99,
    # clip_ratio=0.1,
    # pi_lr=2e-3,
    # vf_lr=2e-3,
    # k_epochs=4,
    # update_every_j_timestep=1000,
    # max_episode_length=2_000,
    # max_steps=500,
    # critic_hidden_size=critic_size,
    # actor_hidden_size=actor_size,
    # render=False,
    # random_seed=1,
    # solved_reward=249,
    # observationNormalization=False,
    # actorGradientNormalization=0,
    # normalizeAdvantage=False,
    # initialization="orthogonal",  # normal = normal distribution, None
    # advantageAlgorithm="GAE",  # None = use A2C reward calculation o "GAE"
    # # pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
    # # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
    # coeficient_entropy=0.01,
    # coeficient_value=0.5,
    # lmbda=0.99,
    # eps=1e-5,
    # weight_decay=5e-4,
    # tensorboardName="base-32,lamda=0.99,LR=2e-3,Upd=1K,ON=False,AN=False"
    # )
