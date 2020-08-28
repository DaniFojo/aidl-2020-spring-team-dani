import ppo
if __name__ == "__main__":
    train = False
    size = 256
    if train:
        env = ppo.train(environment_name="Breakout-v0",
                        gamma=0.99,
                        clip_ratio=0.2,
                        pi_lr=0.002,  # Learning rate for policy optimizer.                        
                        vf_lr=0.002, # Learningrate for value function optimizer.
                        k_epochs=4,
                        update_every_j_timestep=300,
                        max_episode_length=50_000,
                        max_steps=50_000,
                        critic_hidden_size=size,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=250
                        )

    ppo.play_latest("Breakout-v0", size, atari=True)
