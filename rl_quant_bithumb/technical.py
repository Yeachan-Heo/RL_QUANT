import pandas as pd


def moving_average(period:int):
    """
    period: int (주기)
    return: Callable (이동평균선 함수)
    """
    def wrapped(df:pd.DataFrame):
        """
        df: pd.DataFrame (캔들스틱이 들어있는 데이터프레임)
        return: pd.Series (이동평균선 값)
        """
        return df["Close"].rolling(period).mean()
    return wrapped


def stddev_ret(period):
    """
    period: int (주기)
    return: Callable (표준편차 함수)
    """
    def wrapped(df):
        """
        df: pd.DataFrame (캔들스틱이 들어있는 데이터프레임)
        return: pd.Series (이동 표준편차 값)
        """
        return df["Change"].rolling(period).std()
    return wrapped


def rate_of_change(period):
    """
    period: int (주기)
    return: Callable (roc 함수)
    """
    def wrapped(df):
        """
        df: pd.DataFrame (캔들스틱이 들어있는 데이터프레임)
        return: pd.Series (ROC 값)
        """
        return df["Close"].rolling(period).apply(lambda x: (x[-1] - x[0]) / x[-1])
    return wrapped


def rsi(period):
    """
    period: int (주기)
    return: Callable (rsi 함수)
    """
    def wrapped(df):
        """
        df: pd.DataFrame (캔들스틱이 들어있는 데이터프레임)
        return: pd.Series (rsi 값)
        """
        RS = -df["Close"].diff().rolling(period).apply(
            lambda x: x.loc[x > 0].mean() / x.loc[x < 0].mean()
        )
        RSI = RS / (1 + RS) * 100
        return RSI
    return wrapped