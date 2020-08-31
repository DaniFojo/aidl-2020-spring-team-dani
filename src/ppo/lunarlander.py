import ppo
if __name__ == "__main__":
    train = True
    name = "LunarLander-v2"
    actor_size = 16
    critic_size = 256
    if train:
        env = ppo.train(environment_name=name,
                        gamma=0.99,
                        clip_ratio=0.1,
                        pi_lr=2e-3,
                        vf_lr=2e-3,
                        k_epochs=4,
                        update_every_j_timestep=32,
                        max_episode_length=10_000,
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
                        advantageAlgorithm="GAE",  # None = use A2C reward calculation
                        pathForBasePolicyToTrain=f"./model/ppo_{name}_policy_latest.pth",
                        pathForBaseCriticToTrain=f"./model/ppo_{name}_critic_latest.pth",
                        coeficient_entropy=0.001,
                        coeficient_value=0.5,
                        lmbda=0.95,
                        eps=1e-5,
                        weight_decay=5e-4,
                        tensorboardName="base-16,update=1k,clip=0.1,lr=2e-3,entropy=0.1,eps=1"
                        )

    ppo.play_latest(name, actor_size, plot=True)
