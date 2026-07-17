import tyro

from rgrl.train_loop import TrainConfig, train

if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
