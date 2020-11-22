from collections import deque
from rl_quant_bithumb.imports import *
from rl_quant_bithumb.config import *
from rl_quant_bithumb.technical import *
from rl_quant_bithumb.env import *
import datetime
import time

@ray.remote
def plot_step(log):
    plt.clf()
    plt.plot(log["timestamp"], log["portfolio_value"])
    
class RealBroker():
    def __init__(self, conkey, seckey, target_currency="BTC", log_interval_sec=1000):
        self.bithumb = Bithumb(conkey, seckey)
        self.target_currency = target_currency
        self.env_config = ENV_CONFIG_EVAL.copy()
        self.prev_day = None
        self.prev_log = time.time()
        self.log_interval = log_interval_sec
        
        self.log = {
            "timestamp" : deque(maxlen=10000),
            "portfolio_value" : deque(maxlen=10000),
        }
        
        self.max_period = max(
            self.env_config["rsi_period"], 
            self.env_config["ma_period"], 
            self.env_config["roc_period"], 
            self.env_config["std_period"]
        )
        
        self.rsi = rsi(self.env_config["rsi_period"])
        self.ma = moving_average(self.env_config["ma_period"])
        self.stddev_ret = stddev_ret(self.env_config["std_period"])
        self.rate_of_change = rate_of_change(self.env_config["roc_period"])
        
        plt.ion()

    def order_weight(self, weight):
        weight = np.clip(weight, 0, 0.999) # for slippage
        price = self.bithumb.get_current_price(self.target_currency)
        balance = self.bithumb.get_balance(self.target_currency)
        
        portfolio_value = balance[2] + balance[0] * price
        
        target_amount = portfolio_value * weight / price
        order_amount = target_amount - balance[1]
        order_side = np.sign(order_amount)
        order_amount = np.abs(order_amount)
        
        if order_amount <= 0:
            return order_side, order_amount

        if order_side == 1:
            self.bithumb.buy_market_order(self.target_currency, order_amount)
            
        elif order_side == -1:
            self.bithumb.sell_market_order(self.target_currency, order_amount)
        
        return order_side, order_amount
    
    def has_day_updaten(self):
        df = self.bithumb.get_candlestick(self.target_currency)
        
        if self.prev_day != df.index[-1].day:
            self.prev_day = df.index[-1].day
            return True
        return False
    
    def get_balance(self):
        price = self.bithumb.get_current_price(self.target_currency)
        balance = self.bithumb.get_balance(self.target_currency)
        
        portfolio_value = balance[2] + balance[0] * price
        return balance, portfolio_value
    
    def get_observation(self):
        df = self.bithumb.get_candlestick(self.target_currency)
        df = df.iloc[-(self.max_period+5):-1]
        
        df["rsi"] = self.rsi(df)
        df["std"] = self.stddev_ret(df)
        df["ma"] = self.ma(df)
        df["roc"] = self.rate_of_change(df)
        
        ma_div = (df["Close"].values[-1] - df["ma"].values[-1]) / df["Close"].values[-1]
        rsi = df["rsi"].values[-1] / 100 # normalize
        ret = np.array([ma_div, rsi, df["std"].values[-1], df["roc"].values[-1]])
        
        if np.nan in ret:
            raise ValueError
        return ret
    
    def write_log(self):
        if time.time() - self.prev_log >= self.log_interval:
            self.prev_log = time.time()
            self.log["timestamp"].append(datetime.datetime.now())
            self.log["portfolio_value"].append(self.get_balance()[-1])
        plot_step.remote(self.log)
        
    def trade(self, restore_path):
        side_dict = {1 : "buy", -1 : "sell"}

        config = EVAL_CONFIG.copy()
        config["env_config"] = self.env_config
        config["env"] = SimpleTradingEnv
        config["env_config"]["data"] = make_data(self.target_currency)

        agent = ppo.PPOTrainer(config)
        agent.restore(restore_path)
        
        policy = agent.workers.local_worker().get_policy()

        state=policy.get_initial_state()
        action, state, logits = agent.compute_action(self.get_observation(), state)

        while True:
            self.write_log()
            
            if self.has_day_updaten():
                action, state, logits = agent.compute_action(self.get_observation(), state)
                
                action = np.array(logits["action_dist_inputs"][0])
                
                side, amount = self.order_weight(action)
                
                print(f"{side_dict[side]} {amount} {self.target_currency} at {str(datetime.datetime.now())}")
            