from rl_quant_bithumb.imports import *

# ## 학습 상황 기록 Callback
# 물론 리워드의 변동에 따라 학습 진행 상황을 알 수 있지만, 금융 도메인의 메트릭을 함께 보면 더욱 더 풍성한 결과를 보며 학습할 수 있다.  
# 샤프 지수, 소르티노 지수, 총 거래일(%), 승률, 손익비로 총 5개의 지표를 사용한다


class QuantstatsCallback(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    
    def on_episode_end(self, worker, base_env,
                       policies, episode,
                       env_index, *args, **kwargs):
        env = base_env.get_unwrapped()[0]
        series = pd.Series(env.log["portfolio_value"], index=pd.DatetimeIndex(env.log["timestamp"]))
        try:
            episode.custom_metrics["sharpe"] = qs.stats.sharpe(series)
            episode.custom_metrics["sortino"] = qs.stats.sortino(series)
            episode.custom_metrics["exposure"] = qs.stats.exposure(series)
            episode.custom_metrics["win_rate"] = qs.stats.win_rate(series)
            episode.custom_metrics["pnl_ratio"] = qs.stats.win_loss_ratio(series)
        except:
            episode.custom_metrics["sharpe"] = np.nan
            episode.custom_metrics["sortino"] = np.nan
            episode.custom_metrics["exposure"] = np.nan
            episode.custom_metrics["win_rate"] = np.nan
            episode.custom_metrics["pnl_ratio"] = np.nan