from rl_quant_bithumb.config import *
from rl_quant_bithumb.technical import *
from rl_quant_bithumb.imports import *


class OHLCVData:
    def __init__(self, df:pd.DataFrame, open="open", high="high", low="low", close="close", volume="volume", change="change"):
        """
        df: pd.DataFrame (open, high, low, close, volume, change가 담긴 pd.DataFrame)
        open, high, low, close, volume, change: str,
        (주어진 데이터프레임의 ohlcvc의 레이블)
        """
        self.df = pd.DataFrame({
            "Open" : df[open],
            "High" : df[high],
            "Low" : df[low],
            "Close" : df[close],
            "Volume" : df[volume],
            "Change" : df[change]
            },
            index = df.index
        )

        self.index = 0 # 인덱스

    def __getitem__(self, item:str):
        # data["close"] 와 같은 표현을 사용해 쉽게 인덱싱 할 수 있도록 __getitem__ 매직메서드 구현
        return self.current_value(item)

    def __len__(self):
        # 데이터의 길이 반환
        return len(self.df.index)
    
    def current_value(self, label:str):
        # 인덱싱 후 해당 레이블의 현재 값 반환
        return self.df[label].iloc[self.index]
    
    def add_technical_indicator(self, func, label):
        # 위에서 구현한 기술적 지표 함수를 받아 데이터프레임에 추가
        self.df[label] = func(self.df)

    def next(self):
        # 다음 인덱스로 넘김
        self.index += 1
    
    def is_last_index(self):
        # 데이터의 끝에 도달했는지 여부 반환
        return self.index == (len(self.df.index) - 1)

    def reset(self):
        # 인덱스 초기화
        self.index = 0


def make_data(ticker="BTC"):
    # 데이터를 받아오고 정제하는 함수
    df = Bithumb.get_candlestick(ticker)
    
    # 전일대비 컬럼 추가
    df["change"] = df["close"].diff() / df["close"]

    data = OHLCVData(df)
    
    # 기술적 지표 추가
    data.add_technical_indicator(rsi(ENV_CONFIG["rsi_period"]), "rsi")
    data.add_technical_indicator(moving_average(ENV_CONFIG["ma_period"]), "ma")
    data.add_technical_indicator(stddev_ret(ENV_CONFIG["std_period"]), "std")
    data.add_technical_indicator(rate_of_change(ENV_CONFIG["roc_period"]), "roc")
    return data