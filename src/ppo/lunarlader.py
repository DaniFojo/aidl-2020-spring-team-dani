import ppo
if __name__ == "__main__":
    train = True
    size = 16
    if train:
        env = ppo.train(environment_name="LunarLander-v2",
                        gamma=0.99,
                        clip_ratio=0.1,
                        pi_lr=1e-3,
                        vf_lr=1e-3,
                        k_epochs=4,
                        update_every_j_timestep=32,
                        max_episode_length=1_000,
                        max_steps=300,
                        critic_hidden_size=256,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=250,
                        observationNormalization=False,
                        actorGradientNormalization=0,
                        initialization="orthogonal",  # normal = normal distribution, None
                        advantageAlgorithm="GAE",  # None = use A2C reward calculation
                        # pathForBasePolicyToTrain="./model/ppo_LunarLander-v2_policy_latest.pth",
                        # pathForBaseCriticToTrain="./model/ppo_LunarLander-v2_critic_latest.pth",
                        coeficient_entropy=0.001,
                        coeficient_value=0.5,
                        lmbda=0.95,
                        weight_decay=0.000_5,
                        tensorboardName="ClipTo0.1"
                        )

    ppo.play_latest("LunarLander-v2", size)
