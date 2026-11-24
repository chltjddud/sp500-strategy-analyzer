"""
Technical Indicators, Base Trading Strategies, Strategy-Specific Take-Profit / Stop-Loss Rules,
and Multi-Strategy Combination Generator.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Metadata for the 6 core trading strategies including explicit TP / SL rules
STRATEGY_METADATA: Dict[str, Dict[str, str]] = {
    "MA_CROSS": {
        "name": "이평선 골든크로스·지지",
        "short": "이평선(MA)",
        "category": "추세 추종",
        "desc": "20일선이 50일선을 골든크로스하거나, 정배열(20>50) 상태에서 20일선 눌림목 지지 후 양봉 반등",
        "tp_rule": "20일 전고점 돌파 및 추세 확장 목표치 (진입가 + 2.4×ATR 또는 전고점+1.5%)",
        "sl_rule": "50일 이동평균선(SMA50) 또는 최근 5일 눌림목 저점 이탈 시 손절",
        "tp_short": "전고점 돌파·추세확장",
        "sl_short": "50일선·눌림저점 이탈",
    },
    "RSI_REBOUND": {
        "name": "RSI 과매도·눌림 반등",
        "short": "RSI반등",
        "category": "역추세/눌림목",
        "desc": "RSI(14)가 최근 과매도·눌림 구간(42 이하)에 진입한 뒤 상승 반전하며 모멘텀 회복",
        "tp_rule": "단기 평균회귀 저항선인 볼린저 상단(BB상단) 또는 20일선 회복 목표치 (진입가 + 2.0×ATR)",
        "sl_rule": "과매도 반등 직전 5일 스윙 최저가(Low5) 하향 이탈 시 손절 (-0.4×ATR 여유)",
        "tp_short": "볼밴상단·평균회귀",
        "sl_short": "5일 최저가 이탈",
    },
    "BB_LOWER": {
        "name": "볼린저밴드 하단 반등",
        "short": "볼밴하단",
        "category": "역추세/변동성",
        "desc": "저가가 볼린저밴드(20, 2) 하단 부근 터치·이탈 후 당일 양봉으로 반등 마감",
        "tp_rule": "볼린저밴드 중심선(20일선)~상단밴드 구간 목표가 (중심선 돌파 시 상단밴드 70% 구간)",
        "sl_rule": "볼린저밴드 하단선(BB하단) 및 최근 3일 최저가 재이탈 시 손절",
        "tp_short": "볼밴 중심선~상단",
        "sl_short": "볼밴 하단선 이탈",
    },
    "BB_BREAKOUT": {
        "name": "볼린저 상단·신고가 돌파",
        "short": "상단돌파",
        "category": "돌파/모멘텀",
        "desc": "종가가 볼린저밴드 상단 또는 20일 고점을 양봉으로 강하게 상향 돌파",
        "tp_rule": "밴드폭(상단-중심선) 1.2배 비례 확장 목표가 (또는 진입가 + 2.6×ATR 모멘텀 목표)",
        "sl_rule": "돌파 기준봉 저가 또는 20일 중심선(SMA20) 하향 이탈 시 손절",
        "tp_short": "밴드폭 1.2배 확장",
        "sl_short": "돌파봉 저가·20일선 이탈",
    },
    "MACD_TURN": {
        "name": "MACD 골든크로스·상승전환",
        "short": "MACD전환",
        "category": "모멘텀",
        "desc": "MACD가 시그널선을 상향 돌파(오실레이터 양전환)하거나 음수 구간에서 바닥 다지기 후 급반등",
        "tp_rule": "최근 20일 스윙 고점(High20) 저항선 또는 진입가 + 2.2×ATR 모멘텀 목표가",
        "sl_rule": "MACD 전환 직전 5일 스윙 저점(Low5) 이탈 시 손절",
        "tp_short": "20일 스윙고점 저항선",
        "sl_short": "전환직전 5일저점 이탈",
    },
    "VOL_SURGE": {
        "name": "거래량 급증 양봉",
        "short": "거래량급증",
        "category": "수급/거래량",
        "desc": "거래량이 20일 평균 대비 1.4배 이상 급증하면서 전일 대비 +1% 이상 상승한 양봉 발생",
        "tp_rule": "수급 기준봉 캔들 길이의 2.2배 확장 목표가 (또는 진입가 + 2.3×ATR)",
        "sl_rule": "거래량 터진 기준 장대양봉의 저가(또는 시가 하단) 이탈 시 즉시 손절",
        "tp_short": "기준봉 길이 2.2배 확장",
        "sl_short": "수급 양봉 저가 이탈",
    },
    "STOCH_REBOUND": {
        "name": "스토캐스틱 과매도 골든크로스",
        "short": "스토캐스틱",
        "category": "역추세/모멘텀",
        "desc": "슬로우 스토캐스틱(14,3,3) %K가 과매도 구간(32 이하)에서 %D를 상향 돌파하며 양봉 반등할 때",
        "tp_rule": "최근 20일 스윙 고점 저항선 또는 진입가 + 2.0×ATR 단기 반등 목표가",
        "sl_rule": "과매도 반등 직전 5일 스윙 최저가(Low5) 이탈 시 손절 (-0.35×ATR 여유)",
        "tp_short": "스윙고점·단기반등",
        "sl_short": "5일 최저가 이탈",
    },
    "EMA_PULLBACK": {
        "name": "단기 추세 EMA 8/21 눌림 반등",
        "short": "EMA눌림",
        "category": "추세 추종/눌림목",
        "desc": "단기 정배열(EMA8 > EMA21) 상승 추세 중 8일~21일 지수이평선 부근까지 눌렸다가 양봉으로 EMA8을 재돌파할 때",
        "tp_rule": "20일 전고점 돌파(+1%) 또는 진입가 + 2.2×ATR 추세 재개 목표가",
        "sl_rule": "21일 지수이평선(EMA21) 또는 최근 5일 저점 하향 이탈 시 손절",
        "tp_short": "전고점 돌파·추세재개",
        "sl_short": "EMA21·5일저점 이탈",
    },
}

BASE_STRATEGY_KEYS: List[str] = list(STRATEGY_METADATA.keys())


def format_combo_name(combo: Tuple[str, ...]) -> str:
    """Return human-readable Korean name for a strategy combination."""
    return " + ".join(STRATEGY_METADATA[k]["short"] for k in combo)


def format_combo_id(combo: Tuple[str, ...]) -> str:
    """Return canonical ID string for a strategy combination."""
    return "+".join(combo)


def parse_combo_id(combo_id: str) -> Tuple[str, ...]:
    """Convert canonical ID string back to tuple of strategy keys."""
    return tuple(k for k in combo_id.split("+") if k in STRATEGY_METADATA)


def describe_combo_tp_sl_rules(combo: Tuple[str, ...]) -> Tuple[str, str]:
    """
    Return clean, concise Korean explanation of how Take-Profit (익절선) and Stop-Loss (손절선)
    are determined for the given strategy or combination.
    """
    if len(combo) == 1:
        meta = STRATEGY_METADATA[combo[0]]
        return meta["tp_rule"], meta["sl_rule"]

    tp_parts = [f"{STRATEGY_METADATA[k]['short']}({STRATEGY_METADATA[k]['tp_short']})" for k in combo]
    sl_parts = [f"{STRATEGY_METADATA[k]['short']}({STRATEGY_METADATA[k]['sl_short']})" for k in combo]
    tp_desc = " + ".join(tp_parts) + " 목표가 결합"
    sl_desc = " + ".join(sl_parts) + " 지지선 결합"
    return tp_desc, sl_desc


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators required for all base strategies and dynamic TP/SL levels.
    Strictly causal (uses only current and past bars, no look-ahead bias).
    """
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"]

    # Moving Averages
    out["SMA20"] = close.rolling(window=20, min_periods=20).mean()
    out["SMA50"] = close.rolling(window=50, min_periods=50).mean()
    out["SMA200"] = close.rolling(window=200, min_periods=100).mean()

    # True Range & Wilder's ATR(14) for volatility-adjusted TP/SL
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
    out["ATR14"] = out["ATR14"].fillna(close * 0.02)

    # Recent Swing Lows & Highs (strictly causal, min_periods=1)
    out["LOW3"] = low.rolling(window=3, min_periods=1).min()
    out["LOW5"] = low.rolling(window=5, min_periods=1).min()
    out["HIGH20"] = high.rolling(window=20, min_periods=1).max()

    # Wilder's RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    out["RSI14"] = out["RSI14"].fillna(50.0)

    # Bollinger Bands (20, 2)
    bb_std = close.rolling(window=20, min_periods=20).std()
    out["BB_MID"] = out["SMA20"]
    out["BB_UPPER"] = out["BB_MID"] + 2.0 * bb_std
    out["BB_LOWER"] = out["BB_MID"] - 2.0 * bb_std
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / out["BB_MID"].replace(0, np.nan)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD_LINE"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD_LINE"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD_LINE"] - out["MACD_SIGNAL"]

    # Volume Moving Average & Ratio
    out["VOL_SMA20"] = vol.rolling(window=20, min_periods=10).mean()
    out["VOL_RATIO"] = vol / out["VOL_SMA20"].replace(0, np.nan)

    # 20-day Prior High (shifted by 1 to avoid look-ahead)
    out["HIGH20_PREV"] = high.shift(1).rolling(window=20, min_periods=15).max()

    # Slow Stochastic (14, 3, 3) - strictly causal
    low14 = low.rolling(window=14, min_periods=1).min()
    high14 = high.rolling(window=14, min_periods=1).max()
    fast_k = ((close - low14) / (high14 - low14).replace(0, np.nan)) * 100.0
    out["STOCH_K"] = fast_k.rolling(window=3, min_periods=1).mean().fillna(50.0)
    out["STOCH_D"] = out["STOCH_K"].rolling(window=3, min_periods=1).mean().fillna(50.0)

    # Short-term Exponential Moving Averages (EMA 8 & EMA 21)
    out["EMA8"] = close.ewm(span=8, adjust=False).mean()
    out["EMA21"] = close.ewm(span=21, adjust=False).mean()

    return out


def compute_strategy_tp_sl_pct(
    df_ind: pd.DataFrame,
    rr_multiplier: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute bar-by-bar dynamic Take-Profit (+%) and Stop-Loss (-%) arrays tailored to each
    of the 6 base trading strategies using technical levels (BB, SMA, Swing High/Low, Candle range, ATR).

    `rr_multiplier` (default 1.0) lets the user scale target aggressiveness in the UI if desired.
    Returns:
      - `base_tp_pct`: Dict[strategy_key -> np.ndarray of positive target % (e.g. +4.8)]
      - `base_sl_pct`: Dict[strategy_key -> np.ndarray of negative stop % (e.g. -2.9)]
    """
    close = df_ind["Close"].to_numpy(dtype=float)
    open_p = df_ind["Open"].to_numpy(dtype=float)
    high = df_ind["High"].to_numpy(dtype=float)
    low = df_ind["Low"].to_numpy(dtype=float)
    atr = df_ind["ATR14"].to_numpy(dtype=float)

    sma20 = df_ind["SMA20"].fillna(df_ind["Close"]).to_numpy(dtype=float)
    sma50 = df_ind["SMA50"].fillna(df_ind["Close"]).to_numpy(dtype=float)
    bb_upper = df_ind["BB_UPPER"].fillna(df_ind["Close"] * 1.04).to_numpy(dtype=float)
    bb_lower = df_ind["BB_LOWER"].fillna(df_ind["Close"] * 0.96).to_numpy(dtype=float)
    low3 = df_ind["LOW3"].fillna(df_ind["Low"]).to_numpy(dtype=float)
    low5 = df_ind["LOW5"].fillna(df_ind["Low"]).to_numpy(dtype=float)
    high20 = df_ind["HIGH20"].fillna(df_ind["High"]).to_numpy(dtype=float)

    safe_close = np.where(close > 0, close, 1.0)
    atr_pct = (atr / safe_close) * 100.0

    base_tp_pct: Dict[str, np.ndarray] = {}
    base_sl_pct: Dict[str, np.ndarray] = {}

    # 1. MA_CROSS:
    # SL: Below SMA50 or 5-day swing low (whichever is closer support below entry, buffered by 0.35*ATR)
    ma_support = np.where(sma50 < close, np.maximum(sma50, low5), low5) - 0.35 * atr
    sl_ma = ((ma_support / safe_close) - 1.0) * 100.0
    sl_ma = np.clip(sl_ma, -9.0, -1.5)
    # TP: Prior 20-day high breakout (+1.5%) or 2.4 * ATR trend extension
    tp_ma_price = np.maximum(high20 * 1.015, close + 2.4 * atr)
    tp_ma = ((tp_ma_price / safe_close) - 1.0) * 100.0
    tp_ma = np.clip(tp_ma * rr_multiplier, np.maximum(2.2, np.abs(sl_ma) * 1.3), 16.0)
    base_tp_pct["MA_CROSS"] = tp_ma
    base_sl_pct["MA_CROSS"] = sl_ma

    # 2. RSI_REBOUND:
    # SL: 5-day oversold swing low minus 0.4 * ATR
    rsi_support = low5 - 0.40 * atr
    sl_rsi = ((rsi_support / safe_close) - 1.0) * 100.0
    sl_rsi = np.clip(sl_rsi, -8.5, -1.5)
    # TP: Mean-reversion toward BB_UPPER (or SMA20 if far below SMA20, plus at least 2.0*ATR)
    rsi_target_price = np.where(
        sma20 > close * 1.025,
        0.5 * (sma20 + bb_upper),
        np.maximum(bb_upper, close + 2.0 * atr),
    )
    tp_rsi = ((rsi_target_price / safe_close) - 1.0) * 100.0
    tp_rsi = np.clip(tp_rsi * rr_multiplier, np.maximum(2.0, np.abs(sl_rsi) * 1.25), 15.0)
    base_tp_pct["RSI_REBOUND"] = tp_rsi
    base_sl_pct["RSI_REBOUND"] = sl_rsi

    # 3. BB_LOWER:
    # SL: Below lower Bollinger Band & 3-day low
    bb_low_support = np.minimum(bb_lower, low3) - 0.35 * atr
    sl_bb_low = ((bb_low_support / safe_close) - 1.0) * 100.0
    sl_bb_low = np.clip(sl_bb_low, -8.0, -1.5)
    # TP: Bollinger Middle Band (SMA20) if >= +2.2% away, else 65% of way from Middle to Upper Band
    bb_mid_plus_upper = sma20 + 0.65 * np.maximum(bb_upper - sma20, atr)
    bb_low_target = np.where(sma20 > close * 1.03, sma20, bb_mid_plus_upper)
    tp_bb_low = ((bb_low_target / safe_close) - 1.0) * 100.0
    tp_bb_low = np.clip(tp_bb_low * rr_multiplier, np.maximum(2.0, np.abs(sl_bb_low) * 1.25), 14.0)
    base_tp_pct["BB_LOWER"] = tp_bb_low
    base_sl_pct["BB_LOWER"] = sl_bb_low

    # 4. BB_BREAKOUT:
    # SL: Breakout candle low or SMA20 (buffered by 0.25*ATR)
    bb_brk_support = np.maximum(low - 0.25 * atr, sma20)
    bb_brk_support = np.where(bb_brk_support < close, bb_brk_support, close - 1.5 * atr)
    sl_bb_brk = ((bb_brk_support / safe_close) - 1.0) * 100.0
    sl_bb_brk = np.clip(sl_bb_brk, -8.5, -1.5)
    # TP: Upper band expansion (1.2x half-bandwidth or 2.6 * ATR)
    half_bw = np.maximum(bb_upper - sma20, 2.0 * atr)
    bb_brk_target = close + np.maximum(1.2 * half_bw, 2.5 * atr)
    tp_bb_brk = ((bb_brk_target / safe_close) - 1.0) * 100.0
    tp_bb_brk = np.clip(tp_bb_brk * rr_multiplier, np.maximum(2.5, np.abs(sl_bb_brk) * 1.4), 18.0)
    base_tp_pct["BB_BREAKOUT"] = tp_bb_brk
    base_sl_pct["BB_BREAKOUT"] = sl_bb_brk

    # 5. MACD_TURN:
    # SL: 5-day swing low before MACD turn
    macd_support = low5 - 0.35 * atr
    sl_macd = ((macd_support / safe_close) - 1.0) * 100.0
    sl_macd = np.clip(sl_macd, -8.5, -1.5)
    # TP: 20-day swing high or +2.2 * ATR
    macd_target = np.maximum(high20, close + 2.2 * atr)
    tp_macd = ((macd_target / safe_close) - 1.0) * 100.0
    tp_macd = np.clip(tp_macd * rr_multiplier, np.maximum(2.0, np.abs(sl_macd) * 1.3), 15.0)
    base_tp_pct["MACD_TURN"] = tp_macd
    base_sl_pct["MACD_TURN"] = sl_macd

    # 6. VOL_SURGE:
    # SL: Below the surge candle's low (or open) - key institutional support line
    vol_support = np.minimum(low, open_p) - 0.25 * atr
    sl_vol = ((vol_support / safe_close) - 1.0) * 100.0
    sl_vol = np.clip(sl_vol, -8.0, -1.5)
    # TP: 2.2x surge candle range or 2.3 * ATR
    candle_range = np.maximum(high - low, atr)
    vol_target = close + np.maximum(2.2 * candle_range, 2.3 * atr)
    tp_vol = ((vol_target / safe_close) - 1.0) * 100.0
    tp_vol = np.clip(tp_vol * rr_multiplier, np.maximum(2.2, np.abs(sl_vol) * 1.35), 16.0)
    base_tp_pct["VOL_SURGE"] = tp_vol
    base_sl_pct["VOL_SURGE"] = sl_vol

    # 7. STOCH_REBOUND:
    # SL: 5-day oversold swing low minus 0.35 * ATR
    stoch_support = low5 - 0.35 * atr
    sl_stoch = ((stoch_support / safe_close) - 1.0) * 100.0
    sl_stoch = np.clip(sl_stoch, -8.0, -1.5)
    # TP: 20-day swing high or +2.0 * ATR
    stoch_target = np.maximum(high20 * 0.995, close + 2.0 * atr)
    tp_stoch = ((stoch_target / safe_close) - 1.0) * 100.0
    tp_stoch = np.clip(tp_stoch * rr_multiplier, np.maximum(2.0, np.abs(sl_stoch) * 1.25), 14.0)
    base_tp_pct["STOCH_REBOUND"] = tp_stoch
    base_sl_pct["STOCH_REBOUND"] = sl_stoch

    # 8. EMA_PULLBACK:
    # SL: Below EMA21 or 5-day swing low minus 0.30 * ATR
    ema21 = df_ind["EMA21"].fillna(df_ind["Close"]).to_numpy(dtype=float)
    ema_support = np.minimum(ema21, low5) - 0.30 * atr
    sl_ema = ((ema_support / safe_close) - 1.0) * 100.0
    sl_ema = np.clip(sl_ema, -8.0, -1.5)
    # TP: 20-day high (+1.0%) or +2.2 * ATR
    ema_target = np.maximum(high20 * 1.01, close + 2.2 * atr)
    tp_ema = ((ema_target / safe_close) - 1.0) * 100.0
    tp_ema = np.clip(tp_ema * rr_multiplier, np.maximum(2.2, np.abs(sl_ema) * 1.3), 15.0)
    base_tp_pct["EMA_PULLBACK"] = tp_ema
    base_sl_pct["EMA_PULLBACK"] = sl_ema

    return base_tp_pct, base_sl_pct


def get_combo_tp_sl_arrays(
    combo: Tuple[str, ...],
    base_tp_pct: Dict[str, np.ndarray],
    base_sl_pct: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine the strategy-specific Take-Profit (+%) and Stop-Loss (-%) arrays
    for a 1..3 strategy combination.
    """
    if len(combo) == 1:
        k = combo[0]
        return base_tp_pct[k], base_sl_pct[k]

    tp_stack = np.vstack([base_tp_pct[k] for k in combo])
    sl_stack = np.vstack([base_sl_pct[k] for k in combo])
    # Average technical target & support levels across the combined strategies
    combo_tp = np.mean(tp_stack, axis=0)
    combo_sl = np.mean(sl_stack, axis=0)
    return combo_tp, combo_sl


def generate_raw_signals(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily boolean signals for each of the 6 base trading strategies.
    Returns a DataFrame of bool columns indexed identically to df_ind.
    """
    open_p = df_ind["Open"]
    close = df_ind["Close"]
    high = df_ind["High"]
    low = df_ind["Low"]

    sma20 = df_ind["SMA20"]
    sma50 = df_ind["SMA50"]
    rsi = df_ind["RSI14"]
    bb_upper = df_ind["BB_UPPER"]
    bb_lower = df_ind["BB_LOWER"]
    macd_hist = df_ind["MACD_HIST"]
    vol_ratio = df_ind["VOL_RATIO"]
    high20_prev = df_ind["HIGH20_PREV"]

    is_bullish_candle = close > open_p
    daily_ret = close.pct_change()

    # 1. MA_CROSS: Golden Cross (SMA20 crosses above SMA50) OR Pullback bounce on SMA20 in uptrend
    golden_cross = (sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))
    uptrend_pullback = (
        (sma20 > sma50)
        & (low <= sma20 * 1.01)
        & (close > sma20)
        & is_bullish_candle
        & (close.shift(1) <= sma20.shift(1) * 1.015)
    )
    sig_ma = (golden_cross | uptrend_pullback).fillna(False)

    # 2. RSI_REBOUND: RSI dipped <= 42 within last 3 bars and turns up strongly today
    rsi_dipped = rsi.rolling(window=3, min_periods=1).min() <= 42.0
    sig_rsi = (
        rsi_dipped
        & (rsi > rsi.shift(1) + 1.5)
        & (rsi >= 33.0)
        & (rsi <= 58.0)
        & (close > close.shift(1))
    ).fillna(False)

    # 3. BB_LOWER: Low touched near/below lower Bollinger Band within last 2 bars + bullish bounce
    touched_lower = (low <= bb_lower * 1.012) | (low.shift(1) <= bb_lower.shift(1) * 1.012)
    sig_bb_low = (
        touched_lower
        & is_bullish_candle
        & (close > close.shift(1))
        & (close > bb_lower)
    ).fillna(False)

    # 4. BB_BREAKOUT: Close breaks above upper Bollinger Band or 20-day high with bullish candle
    bb_break = (close > bb_upper) & (close.shift(1) <= bb_upper.shift(1))
    high_break = (close > high20_prev) & (close.shift(1) <= high20_prev)
    sig_bb_break = ((bb_break | high_break) & is_bullish_candle).fillna(False)

    # 5. MACD_TURN: MACD Histogram crosses from <=0 to >0, or turns up sharply after declining
    hist_cross_up = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
    hist_rebound = (
        (macd_hist > macd_hist.shift(1))
        & (macd_hist.shift(1) > macd_hist.shift(2))
        & (macd_hist.shift(2) < macd_hist.shift(3))
        & (macd_hist < 0)
        & is_bullish_candle
    )
    sig_macd = (hist_cross_up | hist_rebound).fillna(False)

    # 6. VOL_SURGE: Volume >= 1.4x 20-day avg + bullish candle + daily gain >= 1.0%
    sig_vol = (
        (vol_ratio >= 1.4)
        & is_bullish_candle
        & (daily_ret >= 0.01)
    ).fillna(False)

    # 7. STOCH_REBOUND: Slow Stochastic %K crosses above %D from oversold (<= 32) with bullish bounce
    stoch_k = df_ind["STOCH_K"]
    stoch_d = df_ind["STOCH_D"]
    stoch_oversold = stoch_k.rolling(window=3, min_periods=1).min() <= 32.0
    sig_stoch = (
        stoch_oversold
        & (stoch_k > stoch_d)
        & (stoch_k.shift(1) <= stoch_d.shift(1))
        & is_bullish_candle
        & (close > close.shift(1))
    ).fillna(False)

    # 8. EMA_PULLBACK: Short-term uptrend (EMA8 > EMA21), pullback near EMA8/EMA21, bullish reclaim of EMA8
    ema8 = df_ind["EMA8"]
    ema21 = df_ind["EMA21"]
    sig_ema_pullback = (
        (ema8 > ema21)
        & (low <= ema8 * 1.003)
        & (low >= ema21 * 0.985)
        & (close > ema8)
        & is_bullish_candle
        & (close > close.shift(1))
    ).fillna(False)

    signals = pd.DataFrame(
        {
            "MA_CROSS": sig_ma.astype(bool),
            "RSI_REBOUND": sig_rsi.astype(bool),
            "BB_LOWER": sig_bb_low.astype(bool),
            "BB_BREAKOUT": sig_bb_break.astype(bool),
            "MACD_TURN": sig_macd.astype(bool),
            "VOL_SURGE": sig_vol.astype(bool),
            "STOCH_REBOUND": sig_stoch.astype(bool),
            "EMA_PULLBACK": sig_ema_pullback.astype(bool),
        },
        index=df_ind.index,
    )
    return signals


def generate_all_combinations(max_comb: int = 3) -> List[Tuple[str, ...]]:
    """Return list of all strategy combinations from size 1 up to max_comb."""
    combos: List[Tuple[str, ...]] = []
    for k in range(1, min(max_comb, len(BASE_STRATEGY_KEYS)) + 1):
        combos.extend(combinations(BASE_STRATEGY_KEYS, k))
    return combos


def compute_combination_signals(
    raw_signals: pd.DataFrame,
    signal_window: int = 2,
    max_comb: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Compute boolean signal arrays for all 1..max_comb strategy combinations.
    """
    n_rows = len(raw_signals)
    raw_dict: Dict[str, np.ndarray] = {
        col: raw_signals[col].to_numpy(dtype=bool) for col in BASE_STRATEGY_KEYS
    }

    if signal_window > 1:
        rolled_dict: Dict[str, np.ndarray] = {
            col: (
                raw_signals[col]
                .rolling(window=signal_window, min_periods=1)
                .max()
                .fillna(0)
                .to_numpy(dtype=bool)
            )
            for col in BASE_STRATEGY_KEYS
        }
    else:
        rolled_dict = raw_dict

    combos = generate_all_combinations(max_comb=max_comb)
    combo_signals: Dict[str, np.ndarray] = {}

    for combo in combos:
        cid = format_combo_id(combo)
        if len(combo) == 1:
            combo_signals[cid] = raw_dict[combo[0]]
        else:
            all_within_window = np.ones(n_rows, dtype=bool)
            any_today = np.zeros(n_rows, dtype=bool)
            for k in combo:
                all_within_window &= rolled_dict[k]
                any_today |= raw_dict[k]
            combo_signals[cid] = all_within_window & any_today

    return combo_signals


def generate_trading_plan(
    combo: Tuple[str, ...],
    entry_price: float,
    tp_price: float,
    sl_price: float,
    atr_val: float,
    hold_days: int = 10,
) -> Dict[str, str]:
    """
    Generate concrete entry price range, max chase price, order type, and
    position management instructions tailored to the strategy combination.
    """
    is_breakout = any(k in ("BB_BREAKOUT", "VOL_SURGE") for k in combo)
    is_rebound = any(k in ("RSI_REBOUND", "BB_LOWER", "STOCH_REBOUND", "EMA_PULLBACK") for k in combo)
    atr_safe = max(entry_price * 0.01, atr_val)

    if is_breakout and not is_rebound:
        buy_low = round(entry_price * 0.996, 2)
        buy_high = round(entry_price * 1.006, 2)
        max_chase = round(entry_price + 0.65 * atr_safe, 2)
        method_type = "돌파·모멘텀 추종 매수"
        order_step = (
            f"1) 진입: 장 마감 부근 또는 익일 장초반 ${buy_low:.2f}~${buy_high:.2f} 구간에서 1차 진입 "
            f"(시초가가 ${max_chase:.2f} 이상 갭상승 시 추격 매수 금지)\n"
            f"2) 익절: 목표가 ${tp_price:.2f}에 지정가 매도 예약 (또는 절반 익절 후 트레일링)\n"
            f"3) 손절: 기준봉 저가/지지선인 ${sl_price:.2f} 하향 이탈 시 즉시 시장가 손절\n"
            f"4) 시간 청산: {hold_days}거래일 내 목표가 미도달 시 본전/약수익 구간에서 정리"
        )
    elif is_rebound and not is_breakout:
        buy_low = round(max(sl_price * 1.005, entry_price - 0.35 * atr_safe), 2)
        buy_high = round(entry_price * 1.002, 2)
        max_chase = round(entry_price + 0.45 * atr_safe, 2)
        method_type = "과매도·눌림목 분할 매수"
        order_step = (
            f"1) 진입: ${buy_low:.2f}~${buy_high:.2f} 눌림 구간에서 2회 분할 지정가 매수 "
            f"(${max_chase:.2f} 이상 급등 시 대기)\n"
            f"2) 익절: 20일선/밴드 저항 목표가 ${tp_price:.2f} 도달 시 지정가 전량 익절\n"
            f"3) 손절: 직전 스윙 저점인 ${sl_price:.2f} 이탈 시 예외 없이 손절 (물타기 금지)\n"
            f"4) 시간 청산: {hold_days}거래일 경과 시 추세 미회복이면 종가 청산"
        )
    else:
        buy_low = round(max(sl_price * 1.005, entry_price - 0.25 * atr_safe), 2)
        buy_high = round(entry_price * 1.004, 2)
        max_chase = round(entry_price + 0.55 * atr_safe, 2)
        method_type = "복합 시그널(눌림+수급) 확인 매수"
        order_step = (
            f"1) 진입: ${buy_low:.2f}~${buy_high:.2f} 구간 지정가 진입 "
            f"(${max_chase:.2f} 초과 시 눌림 대기)\n"
            f"2) 익절: 복합 목표가 ${tp_price:.2f} 지정가 매도 걸어두기\n"
            f"3) 손절: 핵심 지지선 ${sl_price:.2f} 이탈 시 스탑로스 자동 손절\n"
            f"4) 보유 관리: 최대 {hold_days}거래일 스윙 보유 원칙 준수"
        )

    risk_pct = abs((sl_price / max(0.01, entry_price) - 1.0) * 100.0)
    # 1% account risk sizing rule: position_weight = 1% / stop_loss_pct
    rec_weight = min(25.0, round((1.0 / max(0.5, risk_pct)) * 100.0, 1))

    return {
        "method_type": method_type,
        "entry_zone": f"${buy_low:.2f} ~ ${buy_high:.2f}",
        "max_chase": f"${max_chase:.2f}",
        "rec_weight": f"총 자본의 최대 {rec_weight:.0f}% 이내 (1회 손실 1% 기준)",
        "order_step": order_step,
    }
