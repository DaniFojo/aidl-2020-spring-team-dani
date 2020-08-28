import ppo
if __name__ == "__main__":
    train = False
    size = 32
    if train:
        env = ppo.train(environment_name="LunarLander-v2",
                        gamma=0.99,
                        clip_ratio=0.25,
                        pi_lr=1e-4,
                        vf_lr=1e-4,
                        k_epochs=4,
                        update_every_j_timestep=4,
                        max_episode_length=5_000,
                        max_steps=300,
                        critic_hidden_size=256,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=250,
                        observationNormalization=False,
                        actorGradientNormalization=0,
                        )

    ppo.play_latest("LunarLander-v2", size)
