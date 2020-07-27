import ppo
if __name__ == "__main__":
    train = False
    size = 64
    if train:
        env = ppo.train(environment_name="LunarLander-v2",
                        gamma=0.99,
                        clip_ratio=0.2,
                        pi_lr=0.002,  # Learning rate for policy optimizer.                        
                        vf_lr=0.002, # Learningrate for value function optimizer.
                        k_epochs=4,
                        update_every_j_timestep=1000,
                        max_episode_length=5_000,
                        max_steps=300,
                        critic_hidden_size=size,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=250
                        )

    ppo.play_latest("LunarLander-v2", size)
