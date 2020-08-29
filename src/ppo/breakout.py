import ppo
if __name__ == "__main__":
    train = True
    actor_size = 64
    critic_size = 256
    if train:
        env = ppo.train(environment_name="Breakout-ram-v4",
                        gamma=0.99,
                        clip_ratio=0.25,
                        pi_lr=0.000_3,
                        vf_lr=0.000_3,
                        k_epochs=4,
                        update_every_j_timestep=4,
                        max_episode_length=1000,
                        max_steps=128,
                        critic_hidden_size=critic_size,
                        actor_hidden_size=actor_size,
                        render=False,
                        random_seed=1,
                        solved_reward=300,
                        observationNormalization=True,  # Normalize observations of the environment
                        # Clips gradient norm of an iterable of parameters. max norm of the gradients
                        actorGradientNormalization=5,
                        # pathForBasePolicyToTrain="./model/ppo_Breakout-ram-v0_policy_latest.pth",
                        # pathForBaseCriticToTrain="./model/ppo_Breakout-ram-v0_critic_latest.pth",
                        eps=1e-5,
                        coeficient_entropy=0.1,
                        coeficient_value=0.5,
                        lmbda=0.95,
                        initialization="orthogonal",  # normal = normal distribution, None
                        advantageAlgorithm="GAE",  # None = use A2C reward calculation
                        weight_decay=0.000_05,
                        saveModelsEvery=10_000)

    ppo.play_latest("Breakout-ram-v4", actor_size)
