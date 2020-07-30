import ppo
if __name__ == "__main__":
    train = True
    size = 64
    if train:
        env = ppo.train(environment_name="Assault-ram-v0",
                        gamma=0.99,
                        clip_ratio=0.1,
                        pi_lr=0.002,  # Learning rate for policy optimizer.                        
                        vf_lr=0.002, # Learningrate for value function optimizer.
                        k_epochs=10,
                        update_every_j_timestep=2_000,
                        max_episode_length=11,
                        max_steps=200,
                        critic_hidden_size=size,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=160*5,
                        pathForBasePolicyToTrain="./model/ppo_Assault-ram-v0_policy_19.0K.pth",
                        pathForBaseCriticToTrain="./model/ppo_Assault-ram-v0_critic_19.0K.pth"
                        )

    ppo.play_latest("Assault-ram-v0", size)
