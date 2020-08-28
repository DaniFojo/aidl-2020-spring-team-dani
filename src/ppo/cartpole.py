import ppo
if __name__ == "__main__":
    train = True
    if train:
        env = ppo.train(environment_name="CartPole-v0",
                        solved_reward=199,
                        gamma=0.99,
                        clip_ratio=0.25,
                        pi_lr=0.000_3,
                        vf_lr=0.000_3,
                        k_epochs=4,
                        update_every_j_timestep=100,
                        max_episode_length=500,
                        max_steps=300,
                        critic_hidden_size=256,
                        actor_hidden_size=16,
                        render=False,
                        random_seed=1,
                        observationNormalization=False,
                        actorGradientNormalization=5,
                        )

    ppo.play_latest("CartPole-v0", 16)
