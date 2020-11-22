from rl_quant_bithumb.imports import *
from rl_quant_bithumb.config import *
from rl_quant_bithumb.data import *
from rl_quant_bithumb.env import *

def evaluate(restore_path, ticker, render=False):
    ray.init()

    config_eval = EVAL_CONFIG.copy()
    config_eval["env"] = SimpleTradingEnv
    config_eval["env_config"]["data"] = make_data(ticker)

    agent = ppo.PPOTrainer(config_eval)
    agent.restore(restore_path)
    
    policy = agent.workers.local_worker().get_policy()
    
    env = SimpleTradingEnv(config_eval["env_config"])

    s = env.reset()

    state = policy.get_initial_state()

    while True:
        action, state, logits = agent.compute_action(s, state)
        s, r, d, _ = env.step(np.array(logits["action_dist_inputs"][0]))
        if d:
            break
    
    ray.shutdown()

    if render:
        env.render()

    return env.result(file=f"{ticker}_strategy.html")

import argparse
from rl_quant_bithumb.config import EVAL_CONFIG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path", type=str)
    parser.add_argument("--ticker", type=str)    
    parser.add_argument("--render", type=bool, default=False)
    args = parser.parse_args()

    evaluate(args.restore_path, args.ticker, args.render)