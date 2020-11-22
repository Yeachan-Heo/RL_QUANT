import numpy as np
from rl_quant_bithumb.env import SimpleTradingEnv
from rl_quant_bithumb.config import *
from rl_quant_bithumb.data import *

def buy_n_hold(ticker):
    config = ENV_CONFIG_EVAL.copy()
    config["data"] = make_data(ticker)
    
    env = SimpleTradingEnv(config)
    
    env.reset()

    while True:
        s, r, d, _ = env.step(np.ones(1, ))
        if d:
            break
        
    return env.result(file=f"{ticker}_bnh.html")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str)
    args = parser.parse_args()

    buy_n_hold(args.ticker)    