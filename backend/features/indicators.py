import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_bbands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def compute_volume_spike(volume: pd.Series, window: int = 20) -> pd.Series:
    avg_vol = volume.rolling(window=window).mean()
    return volume / avg_vol


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    plus_di = 100 * plus_dm.rolling(window).mean() / atr
    minus_di = 100 * minus_dm.rolling(window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df with columns: Open, High, Low, Close, Volume (yfinance style).
    Returns df with added feature columns, NaN rows dropped.
    """
    df = df.copy()
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Price returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling stats
    for window in [10, 20, 50]:
        df[f"sma_{window}"] = compute_sma(df["close"], window)
        df[f"volatility_{window}"] = df["returns"].rolling(window).std()

    # Momentum
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])

    # Stochastic
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df["high"], df["low"], df["close"])

    # Trend
    df["adx_14"] = compute_adx(df["high"], df["low"], df["close"])

    # Volatility
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = compute_bbands(df["close"])
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"])

    # Volume
    df["volume_spike"] = compute_volume_spike(df["volume"])
    df["vwap"] = compute_vwap(df["high"], df["low"], df["close"], df["volume"])

    # Price vs moving averages
    df["close_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["close_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    df.dropna(inplace=True)
    return df
