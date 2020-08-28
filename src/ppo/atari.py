import ppo
if __name__ == "__main__":
    train = True
    actor_size = 32
    critic_size = 256
    if train:
        env = ppo.train(environment_name="Breakout-ram-v0",
                        gamma=0.99,
                        clip_ratio=0.2,
                        pi_lr=1e-4,
                        vf_lr=1e-4,
                        k_epochs=4,
                        update_every_j_timestep=4,
                        max_episode_length=10_000,
                        max_steps=128,
                        critic_hidden_size=critic_size,
                        actor_hidden_size=actor_size,
                        render=True,
                        random_seed=1,
                        solved_reward=160 * 3,
                        observationNormalization=False,
                        actorGradientNormalization=0,
                        # pathForBasePolicyToTrain="./model/ppo_Breakout-ram-v0_policy_latest.pth",
                        # pathForBaseCriticToTrain="./model/ppo_Breakout-ram-v0_critic_latest.pth"
                        )

    ppo.play_latest("Breakout-ram-v0", actor_size)
