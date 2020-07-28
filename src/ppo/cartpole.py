import ppo
if __name__ == "__main__":
    train = True
    if train :
        env = ppo.train(environment_name="CartPole-v0",
                solved_reward=199,
                gamma=0.99,
                clip_ratio=0.2,
                pi_lr=0.000_3,  # Learning rate for policy optimizer.
                vf_lr=0.000_3,  # Learning rate for value function optimizer.
                k_epochs=4,
                update_every_j_timestep=100,
                max_episode_length=500,
                max_steps=300,
                critic_hidden_size=64,
                actor_hidden_size=64,
                render=False,
                random_seed=1
        )
                  
    ppo.play_latest("CartPole-v0", 64)