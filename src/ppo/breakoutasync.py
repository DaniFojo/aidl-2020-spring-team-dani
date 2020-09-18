import asyncppo
if __name__ == "__main__":
    train = True
    if train:
        m = asyncppo.Main(
            gamma=0.99,
            lmda=0.95,
            n_update=5_000,
            n_epochs=4,
            n_workers=10,
            worker_steps=512,
            n_mini_batch=4,
            lr=2.5e-4,
            coeficient_entropy=0.01,
            coeficient_vf=0.5,
            actorGradientNormalizations=0.5,
            game='BreakoutNoFrameskip-v4',
            log_interval=10,
        )
        m.run_training_loop()
        m.destroy()
    asyncppo.play(game='BreakoutNoFrameskip-v4',
                  model_file='ppo_BreakoutNoFrameskip-v4_0K.pth',  plot=True)
