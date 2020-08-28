import ppocontinuous as ppo
if __name__ == "__main__":
    ppo.train(environment_name="BipedalWalker-v3",
              solved_reward=2_000,
              gamma=0.99,
              clip_ratio=0.2,
              pi_lr=0.000_3,  # Learning rate for policy optimizer.
              vf_lr=0.000_3,  # Learning rate for value function optimizer.
              k_epochs=10,
              update_every_j_timestep=10,
              max_episode_length=500,
              max_steps=500,
              critic_hidden_size=64,
              actor_hidden_size=64,
              render=True,
              random_seed=1,
              posibles_actions=4)
    ppo.play_latest("BipedalWalker-v3", 64)