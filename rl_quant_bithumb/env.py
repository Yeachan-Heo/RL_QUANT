from rl_quant_bithumb.imports import *
from rl_quant_bithumb.data import *
import numpy as np 
import gym


class Broker:
    def __init__(self, margin:float, fee:float, data:OHLCVData):
        self.data = data # 위에서 구현한 OHLCVData 객체
        self.initial_margin = margin # 증거금
        self.fee = fee # 수수료

    def next(self):
        self.data.next() # 다음 인덱스로 넘김

    def reset(self):
        # info_dict를 초기화하는 함수
        self.info_dict = {
            "n_shares" : 0, # 포지션 개수(암호화폐 보유량)
            "position_value" : 0, # 암호화폐 평가액
            "avg_position_price" : 0, # 암호화폐 평균매수단가
            "unrealized_pnl" : 0, # 평가손익
            "unrealized_pnl_per_share" : 0, # 암호화폐 1개당 평가손익
            "unrealized_pnl_perc" : 0, # 평가손익률
            "margin" : self.initial_margin, # 증거금
            "portfolio_value" : self.initial_margin, # 평가총액
            "position_weight" : 0, # 암호화폐 비중
            "current_price" : 0 # 암호화폐 현재가
        }

    def order_target_weight(self, weight):
        # 암호화폐 비중을 조절하는 주문을 낸다
        
        # 업데이트
        self._update_info()
        
        # 가격 가져오기
        price = self.data.current_value("Close")
        # 목표량: 평가총액 * 목표비중 / 현재가
        target_amount = self.info_dict["portfolio_value"] * weight / price
        # 주문량: 목표량 - 현재량
        order_amount = target_amount - self.info_dict["n_shares"]
        # 주문 방향, 목표량 > 현재량이면 매수, 반대일 시 매도
        order_side = np.sign(order_amount)
        # 주문량 (부호 제거)
        order_amount = np.abs(order_amount)
        # 주문 집행
        self._order(order_amount, order_side)
        
        # 업데이트
        self._update_info()
       
    def _get_max_amount(self):
        # 최대매수량을 구하는 함수 (매수주문 집행 시 사용)
        
        # 현재가 구하기
        price = self.data.current_value("Close")
        
        # 최대매수량 = 평가총액 / 현재가 - 현재보유량
        return self.info_dict["portfolio_value"] / price - self.info_dict["n_shares"]
    
    def _order(self, amount, side):
        
        # 기저 사례: 주문량이 0보다 작다면 주문을 집행하지 않는다. 
        if amount <= 0:
            return
        
        # 매도주문
        if side == -1:
            # 매도량 클리핑: 대상 거래소(빗썸) 에서 숏포지션이 허용되지 않으므로 보유량 이상으로 매도할 수 없다.
            amount = np.clip(amount, 0, self.info_dict["n_shares"])
            # info dict 업데이트
            self.info_dict["n_shares"] -= amount # 보유량 차감
            self.info_dict["margin"] += amount * self.info_dict["unrealized_pnl_per_share"] # 손익실현 
            self.info_dict["margin"] -= amount * self.info_dict["avg_position_price"] * self.fee # 수수료 차감
        # 매수주문
        elif side == 1:
            # 매수량 클리핑: 모든 가상화폐는 증거금률 100%로 매매된다고 가정한다.
            amount = np.clip(amount, 0, self._get_max_amount())
            # 평균매수단가 업데이트
            self.info_dict["avg_position_price"] =                 (self.info_dict["avg_position_price"] * self.info_dict["n_shares"] +
                 amount * self.info_dict["current_price"] * (1 + self.fee)) / (self.info_dict["n_shares"] + amount)
            # 보유량 업데이트
            self.info_dict["n_shares"] += amount
        else: 
            raise ValueError

    def _update_info(self):
        # 가격 변동에 따라 기타 정보들을 업데이트하는 함수(self.reset()에 각 key 값들의 정보가 소개되어 있다.)
        
        price = self.data.current_value("Close")

        self.info_dict["current_price"] = price

        self.info_dict["position_value"] = price * self.info_dict["n_shares"]

        self.info_dict["unrealized_pnl_per_share"] = (price - self.info_dict["avg_position_price"])

        self.info_dict["unrealized_pnl"] = self.info_dict["unrealized_pnl_per_share"] * self.info_dict["n_shares"]

        self.info_dict["unrealized_pnl_perc"] = (price / self.info_dict["avg_position_price"] - 1)

        self.info_dict["portfolio_value"] = self.info_dict["margin"] + self.info_dict["unrealized_pnl"]

        self.info_dict["position_weight"] = self.info_dict["position_value"] / self.info_dict["portfolio_value"]

    def __getitem__(self, other):
        return self.info_dict[other]


# ## SimpleTradingEnv 클래스
# 강화학습 환경을 정의하는 클래스입니다. rllib를 사용하기 위해 openai gym의 인터페이스를 따르도록 만들어줍니다.


class SimpleTradingEnv:
    def __init__(self, config):
        self.config = config

        self.data = config["data"]

        self.max_period = max(
            self.config["rsi_period"], 
            self.config["ma_period"], 
            self.config["roc_period"], 
            self.config["std_period"]
        )

        self.broker = Broker(config["initial_margin"], config["fee"], self.data)
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, ))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1, ))

        plt.ion()

    def get_observation(self):
        # 현재주가와 이동평균의 괴리율
        ma_div = (self.data["Close"] - self.data["ma"]) / self.data["Close"]
        rsi = self.data["rsi"] / 100 # normalize
        ret = np.array([ma_div, rsi, self.data["std"], self.data["roc"]]) # observation
        if np.nan in ret: # nan값은 허용되지 않음
            raise ValueError
        return ret

    def get_reward(self):
        # 리워드 산출
        # 리워드 : 이전 스텝 평가액과 현재 평가액의 변동(%)
        # 리워드가 음수라면 0.001을 한번 더 빼줌
        
        ret = (self.broker["portfolio_value"] - self.prev_portfolio_value) / self.broker["portfolio_value"]
        self.prev_portfolio_value = self.broker["portfolio_value"]
        
        if ret <= 0:
            return np.array(ret-0.001).item()
        return np.array(ret).item()
    
    def get_done(self):
        # 에피소드의 종결 판단하기
        
        # 데이터 rolling window sampling 사용할 시:
        # 데이터의 마지막 인덱스이거나 사전에 설정한 길이가 만료되면 True 반환
        if self.config["use_rolling_index"]:  
            return (self.data.is_last_index() or 
            (self.data.index - self.start_index + 1 >= self.config["episode_length"]))
        # 데이터의 마지막 인덱스이면 True 반환
        if not self.config["use_rolling_index"]:
            return self.data.is_last_index()

    def reset(self):
        # 데이터 인덱스 초기화
        self.data.reset()
        
        # 데이터 시작 인덱스 초기화
        if self.config["use_rolling_index"]:
            self.data.index = np.random.randint(self.max_period + 1, len(self.data) - self.config["episode_length"])
        else:
            self.data.index = self.max_period + 1
            
        self.start_index = self.data.index
        
        # 브로커 초기화
        self.broker.reset()
        
        # 이전 평가액 초기화
        self.prev_portfolio_value = self.broker["portfolio_value"]
        
        # 로그 초기화
        self.log = {
            "portfolio_value" : [],
            "timestamp" : []
        }
        
        # 상태 반환
        return self.get_observation()

    def step(self, action):
        # action: target weight (float)
        
        # 브로커에 비중주문 접수
        self.broker.order_target_weight(action.item())
        
        # 상태, 보상, 에피소드 종결 여부 구하기
        obs, rew, dne = self.get_observation(), self.get_reward(), self.get_done()
        
        # 로깅
        self.log["portfolio_value"].append(self.broker["portfolio_value"])
        self.log["timestamp"].append(self.data.df.index[self.data.index])
        
        # 인덱스 움직이기
        self.broker.next()
        
        return obs, rew, dne, {}

    def render(self):
        # 렌더링
        x = self.log["portfolio_value"] / self.log["portfolio_value"][0] # 포트폴리오 가치를 cumulative return(%)로 변경
        plt.clf() # 기존 플랏 지우기
        # 플라팅
        plt.plot(self.data.df.index, self.data.df["Close"], label="portfolio", color="r")
        plt.plot(self.data.df.index[:len(x)], x, label="benchmark", color="b")
        plt.legend()

    def result(self, file="result.html"):
        # html로 포트폴리오 분석결과 export
        return qs.reports.html(pd.Series(
            self.log["portfolio_value"], index=self.data.df.index[self.start_index:self.data.index]), output=file)


