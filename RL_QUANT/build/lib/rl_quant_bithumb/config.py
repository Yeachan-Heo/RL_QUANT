from rl_quant_bithumb.imports import *
from copy import deepcopy
ENV_CONFIG = {
    "data" : None,
    "rsi_period" : 14, # rsi 지표의 기간
    "ma_period" : 20, # 이동평균 지표의 기간
    "std_period" : 20, # 이동 표준편차 지표의 기간
    "roc_period" : 20, # ROC 지표의 기간
    "initial_margin" : 100000, # 초기 증거금
    "fee" : 0.0003, # 수수료
    "use_rolling_index" : True, # 롤링 인덱스(학습 시 True로 설정)
    "episode_length" : 400, # 한 에피소드의 길이 (윈도우의 크기)
}


ENV_CONFIG_EVAL = deepcopy(ENV_CONFIG)
ENV_CONFIG_EVAL["use_rolling_index"] = False

TRAIN_CONFIG = ppo.DEFAULT_CONFIG.copy()
TRAIN_CONFIG["env_config"] = ENV_CONFIG
TRAIN_CONFIG["model"]["use_lstm"] = True
TRAIN_CONFIG["entropy_coeff_schedule"] = [[0, 0.2], [100000, 0]]
TRAIN_CONFIG["num_gpus"] = 1
TRAIN_CONFIG["num_workers"] = 11

EVAL_CONFIG = deepcopy(TRAIN_CONFIG)
EVAL_CONFIG["num_gpus"] = 0
EVAL_CONFIG["num_workers"] = 1
EVAL_CONFIG["env_config"]["use_rolling_index"] = False