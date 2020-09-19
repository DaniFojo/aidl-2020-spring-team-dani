import asyncppo
if __name__ == "__main__":
    train = True
    if train:
        m = asyncppo.Main(
            gamma=0.99,
            lmda=0.95,
            n_update=10_000 * 10,
            n_epochs=4,
            n_workers=12,
            worker_steps=32,
            n_mini_batch=4,
            lr=2.5e-4,
            coeficient_entropy=0.01,
            coeficient_vf=0.5,
            actorGradientNormalizations=0.5,
            game='BreakoutNoFrameskip-v4',
            log_interval=10,
            solved_reward=250,
            base_model_path="./model/ppo_BreakoutNoFrameskip-v4_100k_F_.pth"
        )
        m.run_training_loop()
        m.destroy()
    asyncppo.play_latest(game='BreakoutNoFrameskip-v4', plot=True)
    # asyncppo.play(game='BreakoutNoFrameskip-v4',
    #               model_file="ppo_BreakoutNoFrameskip-v4_100k_F_.pth", plot=True)
