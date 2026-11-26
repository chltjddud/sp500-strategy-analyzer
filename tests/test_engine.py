"""
Automated Unit & Integration Tests for the S&P 500 Strategy Combination Analyzer
including Strategy-Specific Take-Profit (익절선) and Stop-Loss (손절선) verification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    analyze_single_stock,
    get_trade_history_for_combo,
    precompute_forward_outcomes,
    scan_market,
)
from src.data_loader import get_sp500_constituents
from src.strategies import (
    BASE_STRATEGY_KEYS,
    compute_combination_signals,
    compute_indicators,
    compute_strategy_tp_sl_pct,
    generate_all_combinations,
    generate_raw_signals,
)


def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV stock data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    rets = rng.normal(loc=0.0008, scale=0.018, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_p = close * (1.0 + rng.normal(0.0, 0.006, size=n))
    high = np.maximum(open_p, close) * (1.0 + np.abs(rng.normal(0.0, 0.008, size=n)))
    low = np.minimum(open_p, close) * (1.0 - np.abs(rng.normal(0.0, 0.008, size=n)))
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def test_combination_counts() -> None:
    """Verify 8 base strategies produce 8 + 28 + 56 = 92 combinations."""
    assert len(BASE_STRATEGY_KEYS) == 8
    assert len(generate_all_combinations(max_comb=1)) == 8
    assert len(generate_all_combinations(max_comb=2)) == 36
    assert len(generate_all_combinations(max_comb=3)) == 92


def test_no_lookahead_bias_in_indicators_signals_and_tpsl() -> None:
    """
    Verify indicators, combination signals, and strategy-specific TP/SL percentages
    at bar t are strictly causal and do not change when future bars (t+1..N) are appended.
    """
    df_full = make_synthetic_ohlcv(n=250, seed=7)
    cutoff = 180
    df_partial = df_full.iloc[:cutoff].copy()

    ind_full = compute_indicators(df_full)
    ind_part = compute_indicators(df_partial)

    for col in ["SMA20", "SMA50", "ATR14", "RSI14", "BB_UPPER", "BB_LOWER", "MACD_HIST", "VOL_RATIO", "STOCH_K", "STOCH_D", "EMA8", "EMA21"]:
        np.testing.assert_allclose(
            ind_full[col].iloc[:cutoff].to_numpy(),
            ind_part[col].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Look-ahead bias detected in indicator {col}",
        )

    tp_full, sl_full = compute_strategy_tp_sl_pct(ind_full)
    tp_part, sl_part = compute_strategy_tp_sl_pct(ind_part)
    for k in BASE_STRATEGY_KEYS:
        np.testing.assert_allclose(tp_full[k][:cutoff], tp_part[k], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sl_full[k][:cutoff], sl_part[k], rtol=1e-12, atol=1e-12)
        assert np.all(tp_full[k] > 0)
        assert np.all(sl_full[k] < 0)

    raw_full = generate_raw_signals(ind_full)
    raw_part = generate_raw_signals(ind_part)
    comb_full = compute_combination_signals(raw_full, signal_window=2, max_comb=3)
    comb_part = compute_combination_signals(raw_part, signal_window=2, max_comb=3)

    for cid in comb_full:
        np.testing.assert_array_equal(
            comb_full[cid][:cutoff],
            comb_part[cid],
            err_msg=f"Look-ahead bias detected in strategy combination {cid}",
        )


def test_forward_outcomes_accuracy() -> None:
    """Verify target hit and stop loss precedence in precompute_forward_outcomes."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {
            "Open":  [100, 101, 100, 99, 100, 100, 100, 100, 100, 100],
            "High":  [101, 106, 101, 100, 107, 101, 101, 101, 101, 101],
            "Low":   [99,   99,  99,  94,  99,  99,  99,  99,  99,  99],
            "Close": [100, 104, 100, 96, 105, 100, 100, 100, 100, 100],
            "Volume": [1000] * 10,
        },
        index=dates,
        dtype=float,
    )
    outcomes = precompute_forward_outcomes(df, hold_days=4, target_pct=5.0, stop_pct=-5.0)

    assert bool(outcomes["target_hit"][0]) is True
    assert bool(outcomes["pure_win"][0]) is True
    assert outcomes["realized_ret"][0] == pytest.approx(5.0)

    assert bool(outcomes["target_hit"][2]) is True
    assert bool(outcomes["pure_win"][2]) is False
    assert outcomes["realized_ret"][2] == pytest.approx(-5.0)


def test_analyze_single_stock_with_strategy_tpsl() -> None:
    """Verify end-to-end single stock analysis and market scan with strategy-specific TP/SL."""
    df1 = make_synthetic_ohlcv(n=400, seed=10)
    df2 = make_synthetic_ohlcv(n=400, seed=20)

    stats_df, df_ind, combo_signals, outcomes_ctx = analyze_single_stock(
        df=df1,
        ticker="TEST1",
        hold_days=10,
        signal_window=2,
        min_trades=2,
        max_comb=3,
        use_dynamic_tpsl=True,
    )
    assert not stats_df.empty
    for col in ["기준 진입가($)", "익절가($)", "익절폭(+%)", "손절가($)", "손절폭(-%)", "익절선 기준", "손절선 기준"]:
        assert col in stats_df.columns

    # Verify different strategies produce distinct TP/SL levels tailored to their logic
    assert stats_df["익절가($)"].nunique() > 1
    assert stats_df["손절가($)"].nunique() > 1
    assert (stats_df["익절가($)"] > stats_df["기준 진입가($)"]).all()
    assert (stats_df["손절가($)"] < stats_df["기준 진입가($)"]).all()

    top_cid = stats_df["ComboID"].iloc[0]
    trades_df = get_trade_history_for_combo(df_ind, combo_signals, outcomes_ctx, top_cid)
    assert not trades_df.empty
    assert "익절 목표가($)" in trades_df.columns
    assert "손절 기준가($)" in trades_df.columns

    constituents = get_sp500_constituents()
    active_df, best_df, leader_df = scan_market(
        stock_data={"AAPL": df1, "NVDA": df2},
        constituents_df=constituents,
        hold_days=10,
        min_trades=2,
        use_dynamic_tpsl=True,
    )
    assert len(best_df) == 2
    assert not leader_df.empty


def test_4way_feedback_comparison() -> None:
    """Verify that compute_4way_feedback_comparison produces 4 distinct evaluation tracks."""
    from src.feedback_db import compute_4way_feedback_comparison

    mock_logs = pd.DataFrame(
        [
            {"ticker": "NVDA", "status": "WIN", "actual_return": 4.5, "days_held": 3, "universe_scope": "TOP50", "regime_filter_on": True},
            {"ticker": "AAPL", "status": "LOSS", "actual_return": -3.2, "days_held": 5, "universe_scope": "TOP50", "regime_filter_on": False},
            {"ticker": "PLTR", "status": "WIN", "actual_return": 6.1, "days_held": 4, "universe_scope": "ALL", "regime_filter_on": True},
            {"ticker": "COIN", "status": "EXPIRED_LOSS", "actual_return": -1.2, "days_held": 10, "universe_scope": "ALL", "regime_filter_on": False},
        ]
    )

    comp_df = compute_4way_feedback_comparison(mock_logs)
    assert len(comp_df) == 4
    assert set(comp_df["트랙"].tolist()) == {"Track 1", "Track 2", "Track 3", "Track 4"}

    t1 = comp_df[comp_df["트랙"] == "Track 1"].iloc[0]
    assert t1["종목 범위"] == "상위 50대 종목"
    assert "ON" in t1["시장국면 필터"]
    assert t1["채점 건수"] == 1
    assert t1["익절 적중률(%)"] == 100.0
