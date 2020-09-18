import ppo
if __name__ == "__main__":
    train = True
    actor_size = 32
    critic_size = 256
    name = "MountainCar-v0"
    if train:
        env = ppo.train(environment_name=name,
                        gamma=0.99,
                        clip_ratio=0.2,
                        pi_lr=2e-4,
                        vf_lr=2e-4,
                        k_epochs=4,
                        update_every_j_timestep=400,
                        max_episode_length=500,  # 128 * 80_000 ~ 10 Millions 120K ~ 15M
                        max_steps=200,
                        critic_hidden_size=critic_size,
                        actor_hidden_size=actor_size,
                        render=False,
                        random_seed=1,
                        solved_reward=249,
                        observationNormalization=True,  # Normalize observations of the environment
                        # Clips gradient norm of an iterable of parameters. max norm of the gradients
                        actorGradientNormalization=0,
                        normalizeAdvantage=False,
                        initialization="orthogonal",  # normal = normal distribution, None
                        advantageAlgorithm="GAE",  # None = use A2C reward calculation
                        # pathForBasePolicyToTrain=f"./model/ppo_{name}-v4_policy_latest.pth",
                        # pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
                        coeficient_entropy=0.001,  # 0.001
                        coeficient_value=0.5,  # 0.5
                        lmbda=0.95,
                        eps=1e-5,
                        weight_decay=5e-4,
                        saveModelsEvery=5_000,
                        num_mini_batch=4)

    ppo.play_latest(name, actor_size, plot=True)
