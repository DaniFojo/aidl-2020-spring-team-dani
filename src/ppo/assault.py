import ppo
if __name__ == "__main__":
    train = True
    size = 64
    if train:
        env = ppo.train(environment_name="Assault-ram-v0",
                        gamma=0.99,
                        clip_ratio=0.3,
                        pi_lr=1e-3,  # Learning rate for policy optimizer.                        
                        vf_lr=1e-3, # Learningrate for value function optimizer.
                        k_epochs=3,
                        update_every_j_timestep=32,
                        max_episode_length=500,
                        max_steps=500,
                        critic_hidden_size=size,
                        actor_hidden_size=size,
                        render=False,
                        random_seed=1,
                        solved_reward=160*5,
                        #pathForBasePolicyToTrain="./model/ppo_Assault-ram-v0_policy_19.0K.pth",
                        #pathForBaseCriticToTrain="./model/ppo_Assault-ram-v0_critic_19.0K.pth"
                        )

    ppo.play_latest("Assault-ram-v0", size)
