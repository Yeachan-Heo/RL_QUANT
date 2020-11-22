from rl_quant_bithumb.imports import *
from rl_quant_bithumb.config import *
from rl_quant_bithumb.callbacks import *
from rl_quant_bithumb.env import *
from rl_quant_bithumb.data import *


def train(ticker, logdir, total_episodes, checkpoint_freq, restore=None):
    config = TRAIN_CONFIG.copy()
    print(config["env_config"])
    config["callbacks"] = QuantstatsCallback
    config["env"] = SimpleTradingEnv
    config["env_config"]["data"] = make_data(ticker)
    
    ray.init()

    ray.tune.run(
        ppo.PPOTrainer,
        config=config,
        local_dir=logdir,
        stop={"episodes_total": total_episodes},
        checkpoint_freq=checkpoint_freq,
        restore=restore,
    )

    ray.shutdown()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="BTC")
    parser.add_argument("--logdir", type=str, default="./ray_results/")
    parser.add_argument("--total_episodes", type=int, default=20000)
    parser.add_argument("--checkpoint_freq", type=int, default=100)
    parser.add_argument("--restore", type=str, default=None)
    args = parser.parse_args()

    train(args.ticker, args.logdir, args.total_episodes, args.checkpoint_freq, args.restore)
