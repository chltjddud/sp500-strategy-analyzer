"""
Vectorized Short-Term Swing Backtester & Strategy-Specific Take-Profit / Stop-Loss Evaluator.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.strategies import (
    STRATEGY_METADATA,
    compute_combination_signals,
    compute_indicators,
    compute_strategy_tp_sl_pct,
    describe_combo_tp_sl_rules,
    format_combo_name,
    generate_raw_signals,
    generate_trading_plan,
    get_combo_tp_sl_arrays,
    parse_combo_id,
)


def _deduplicate_indices(indices: np.ndarray, cooldown: int = 3) -> np.ndarray:
    """
    Prevent overlapping entries within `cooldown` trading days so a single cluster
    of signals is counted as one distinct trade setup.
    """
    if len(indices) <= 1 or cooldown <= 1:
        return indices
    selected = [int(indices[0])]
    last_idx = int(indices[0])
    for idx in indices[1:]:
        if int(idx) - last_idx >= cooldown:
            selected.append(int(idx))
            last_idx = int(idx)
    return np.array(selected, dtype=int)


def precompute_forward_outcomes(
    df: pd.DataFrame,
    hold_days: int = 10,
    target_pct: Union[float, np.ndarray] = 4.0,
    stop_pct: Union[float, np.ndarray] = -4.0,
) -> Dict[str, np.ndarray]:
    """
    Compute forward trade outcomes for every bar t in `df` over the next `hold_days` bars.
    Supports either scalar `target_pct`/`stop_pct` or bar-by-bar arrays (`np.ndarray`)
    tailored to a specific trading strategy.
    """
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(close)

    if np.ndim(target_pct) == 0:
        tp_arr = np.full(n, float(target_pct), dtype=float)
    else:
        tp_arr = np.asarray(target_pct, dtype=float)

    if np.ndim(stop_pct) == 0:
        sl_arr = np.full(n, float(stop_pct), dtype=float)
    else:
        sl_arr = np.asarray(stop_pct, dtype=float)

    target_hit = np.zeros(n, dtype=bool)
    stop_hit_first = np.zeros(n, dtype=bool)
    pure_win = np.zeros(n, dtype=bool)
    realized_ret = np.zeros(n, dtype=float)
    max_ret = np.zeros(n, dtype=float)
    min_ret = np.zeros(n, dtype=float)
    end_ret = np.zeros(n, dtype=float)
    valid_eval = np.zeros(n, dtype=bool)

    min_forward_bars = max(2, min(hold_days, 3))

    for t in range(n - min_forward_bars):
        entry = close[t]
        if entry <= 0 or np.isnan(entry):
            continue

        end_idx = min(n, t + 1 + hold_days)
        w_high = high[t + 1 : end_idx]
        w_low = low[t + 1 : end_idx]
        w_close = close[t + 1 : end_idx]

        if len(w_close) == 0:
            continue

        valid_eval[t] = True
        t_pct = float(tp_arr[t])
        s_pct = float(sl_arr[t])  # negative, e.g. -3.5

        max_h = float(np.max(w_high))
        min_l = float(np.min(w_low))
        final_c = float(w_close[-1])

        max_ret[t] = (max_h / entry - 1.0) * 100.0
        min_ret[t] = (min_l / entry - 1.0) * 100.0
        end_ret[t] = (final_c / entry - 1.0) * 100.0

        t_price = entry * (1.0 + t_pct / 100.0)
        s_price = entry * (1.0 + s_pct / 100.0)

        hit_t_bars = np.where(w_high >= t_price)[0]
        hit_s_bars = np.where(w_low <= s_price)[0]

        first_t = int(hit_t_bars[0]) if len(hit_t_bars) > 0 else 999999
        first_s = int(hit_s_bars[0]) if len(hit_s_bars) > 0 else 999999

        if first_t < 999999:
            target_hit[t] = True

        if first_t < first_s:
            pure_win[t] = True
            realized_ret[t] = t_pct
        elif first_s < first_t:
            pure_win[t] = False
            stop_hit_first[t] = True
            realized_ret[t] = s_pct
        elif first_t < 999999 and first_t == first_s:
            if w_close[first_t] >= entry:
                pure_win[t] = True
                realized_ret[t] = t_pct
            else:
                pure_win[t] = False
                stop_hit_first[t] = True
                realized_ret[t] = s_pct
        else:
            pure_win[t] = end_ret[t] > 0.0
            realized_ret[t] = end_ret[t]

    return {
        "target_hit": target_hit,
        "stop_hit_first": stop_hit_first,
        "pure_win": pure_win,
        "realized_ret": realized_ret,
        "max_ret": max_ret,
        "min_ret": min_ret,
        "end_ret": end_ret,
        "valid_eval": valid_eval,
        "tp_pct_arr": tp_arr,
        "sl_pct_arr": sl_arr,
    }


def _evaluate_combo_on_bars(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sig_indices: np.ndarray,
    tp_arr: np.ndarray,
    sl_arr: np.ndarray,
    hold_days: int,
) -> Dict[str, np.ndarray]:
    """
    Fast evaluation of forward outcomes specifically on the bars where a strategy signaled.
    """
    n = len(close)
    m = len(sig_indices)
    min_forward_bars = max(2, min(hold_days, 3))
    max_valid_bar = n - min_forward_bars

    valid_eval = sig_indices < max_valid_bar
    target_hit = np.zeros(m, dtype=bool)
    stop_hit_first = np.zeros(m, dtype=bool)
    pure_win = np.zeros(m, dtype=bool)
    realized_ret = np.zeros(m, dtype=float)
    max_ret = np.zeros(m, dtype=float)
    min_ret = np.zeros(m, dtype=float)
    end_ret = np.zeros(m, dtype=float)

    for i, t in enumerate(sig_indices):
        if not valid_eval[i]:
            continue
        entry = close[t]
        if entry <= 0 or np.isnan(entry):
            valid_eval[i] = False
            continue

        end_idx = min(n, t + 1 + hold_days)
        w_high = high[t + 1 : end_idx]
        w_low = low[t + 1 : end_idx]
        w_close = close[t + 1 : end_idx]
        if len(w_close) == 0:
            valid_eval[i] = False
            continue

        t_pct = float(tp_arr[t])
        s_pct = float(sl_arr[t])

        max_h = float(np.max(w_high))
        min_l = float(np.min(w_low))
        final_c = float(w_close[-1])

        max_ret[i] = (max_h / entry - 1.0) * 100.0
        min_ret[i] = (min_l / entry - 1.0) * 100.0
        end_ret[i] = (final_c / entry - 1.0) * 100.0

        t_price = entry * (1.0 + t_pct / 100.0)
        s_price = entry * (1.0 + s_pct / 100.0)

        hit_t_bars = np.where(w_high >= t_price)[0]
        hit_s_bars = np.where(w_low <= s_price)[0]

        first_t = int(hit_t_bars[0]) if len(hit_t_bars) > 0 else 999999
        first_s = int(hit_s_bars[0]) if len(hit_s_bars) > 0 else 999999

        if first_t < 999999:
            target_hit[i] = True

        if first_t < first_s:
            pure_win[i] = True
            realized_ret[i] = t_pct
        elif first_s < first_t:
            pure_win[i] = False
            stop_hit_first[i] = True
            realized_ret[i] = s_pct
        elif first_t < 999999 and first_t == first_s:
            if w_close[first_t] >= entry:
                pure_win[i] = True
                realized_ret[i] = t_pct
            else:
                pure_win[i] = False
                stop_hit_first[i] = True
                realized_ret[i] = s_pct
        else:
            pure_win[i] = end_ret[i] > 0.0
            realized_ret[i] = end_ret[i]

    return {
        "valid_eval": valid_eval,
        "target_hit": target_hit,
        "stop_hit_first": stop_hit_first,
        "pure_win": pure_win,
        "realized_ret": realized_ret,
        "max_ret": max_ret,
        "min_ret": min_ret,
        "end_ret": end_ret,
    }


def _build_env_mask(
    df_ind: pd.DataFrame,
    market_ctx_df: Optional[pd.DataFrame],
    use_regime_filter: bool,
    vix_max: float,
    use_rs_filter: bool,
    min_rs_pct: float,
) -> Tuple[np.ndarray, float]:
    """
    Build a causal boolean mask combining Market Regime (SPY > SMA200 & VIX <= vix_max)
    and Relative Strength (60d stock return - 60d SPY return >= min_rs_pct).
    Returns (mask_array, latest_rs_vs_spy).
    """
    n = len(df_ind)
    env_mask = np.ones(n, dtype=bool)
    latest_rs = 0.0

    stock_ret60 = (df_ind["Close"].pct_change(60) * 100.0).fillna(0.0)

    if market_ctx_df is not None and not market_ctx_df.empty:
        aligned = market_ctx_df.reindex(df_ind.index).ffill().bfill()
        if use_regime_filter:
            spy_ok = aligned["SPY_ABOVE_200"].fillna(True).to_numpy(dtype=bool)
            vix_ok = (aligned["VIX_CLOSE"].fillna(18.0) <= vix_max).to_numpy(dtype=bool)
            env_mask &= spy_ok & vix_ok

        spy_ret60 = aligned["SPY_RET60"].fillna(0.0)
        rs_series = stock_ret60 - spy_ret60
        latest_rs = round(float(rs_series.iloc[-1]), 2)
        if use_rs_filter:
            env_mask &= (rs_series >= min_rs_pct).to_numpy(dtype=bool)
    else:
        latest_rs = round(float(stock_ret60.iloc[-1]), 2)
        if use_rs_filter:
            env_mask &= (stock_ret60 >= min_rs_pct).to_numpy(dtype=bool)

    return env_mask, latest_rs


def analyze_single_stock(
    df: pd.DataFrame,
    ticker: str = "",
    hold_days: int = 10,
    target_pct: float = 4.0,
    stop_pct: float = -4.0,
    signal_window: int = 2,
    min_trades: int = 5,
    recent_signal_days: int = 2,
    max_comb: int = 3,
    use_dynamic_tpsl: bool = True,
    rr_multiplier: float = 1.0,
    market_ctx_df: Optional[pd.DataFrame] = None,
    use_regime_filter: bool = False,
    vix_max: float = 26.0,
    use_rs_filter: bool = False,
    min_rs_pct: float = 0.0,
    feedback_bonus_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run full indicator, combination signal, and strategy-tailored TP/SL backtest on a single stock.
    Optionally applies causal Market Regime (SPY/VIX), Relative Strength (RS), and Supabase feedback weights.
    """
    if df is None or len(df) < 60:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    df_ind = compute_indicators(df)
    raw_signals = generate_raw_signals(df_ind)
    raw_combo_signals = compute_combination_signals(
        raw_signals, signal_window=signal_window, max_comb=max_comb
    )
    env_mask, latest_rs = _build_env_mask(
        df_ind=df_ind,
        market_ctx_df=market_ctx_df,
        use_regime_filter=use_regime_filter,
        vix_max=vix_max,
        use_rs_filter=use_rs_filter,
        min_rs_pct=min_rs_pct,
    )
    combo_signals = {cid: (arr & env_mask) for cid, arr in raw_combo_signals.items()}

    base_tp_pct, base_sl_pct = compute_strategy_tp_sl_pct(
        df_ind, rr_multiplier=rr_multiplier
    )

    close = df_ind["Close"].to_numpy(dtype=float)
    high = df_ind["High"].to_numpy(dtype=float)
    low = df_ind["Low"].to_numpy(dtype=float)
    atr_arr = df_ind["ATR14"].to_numpy(dtype=float)
    n = len(df_ind)
    dates = df_ind.index

    outcomes_ctx: Dict[str, Any] = {
        "hold_days": hold_days,
        "use_dynamic_tpsl": use_dynamic_tpsl,
        "fixed_target_pct": target_pct,
        "fixed_stop_pct": stop_pct,
        "base_tp_pct": base_tp_pct,
        "base_sl_pct": base_sl_pct,
    }

    records: List[Dict[str, Any]] = []
    fb_map = feedback_bonus_map or {}

    for cid, sig_arr in combo_signals.items():
        combo_tuple = parse_combo_id(cid)
        combo_name = format_combo_name(combo_tuple)
        num_strategies = len(combo_tuple)

        if use_dynamic_tpsl:
            tp_arr, sl_arr = get_combo_tp_sl_arrays(combo_tuple, base_tp_pct, base_sl_pct)
            tp_rule_desc, sl_rule_desc = describe_combo_tp_sl_rules(combo_tuple)
        else:
            tp_arr = np.full(n, float(target_pct), dtype=float)
            sl_arr = np.full(n, float(stop_pct), dtype=float)
            tp_rule_desc = f"고정 익절 비율 (+{target_pct:.1f}%)"
            sl_rule_desc = f"고정 손절 비율 ({stop_pct:.1f}%)"

        all_sig_raw = np.where(sig_arr)[0]
        dedup_sig_idx = _deduplicate_indices(all_sig_raw, cooldown=3)

        eval_res = _evaluate_combo_on_bars(
            close=close,
            high=high,
            low=low,
            sig_indices=dedup_sig_idx,
            tp_arr=tp_arr,
            sl_arr=sl_arr,
            hold_days=hold_days,
        )
        valid_mask = eval_res["valid_eval"]
        trade_count = int(np.sum(valid_mask))
        if trade_count < min_trades:
            continue

        t_hits = eval_res["target_hit"][valid_mask]
        p_wins = eval_res["pure_win"][valid_mask]
        r_rets = eval_res["realized_ret"][valid_mask]
        m_rets = eval_res["max_ret"][valid_mask]
        e_rets = eval_res["end_ret"][valid_mask]

        eval_bars = dedup_sig_idx[valid_mask]
        avg_tp_pct = float(np.mean(tp_arr[eval_bars]))
        avg_sl_pct = float(np.mean(sl_arr[eval_bars]))

        hit_rate = float(np.mean(t_hits) * 100.0)
        win_rate = float(np.mean(p_wins) * 100.0)
        avg_realized = float(np.mean(r_rets))
        avg_max_ret = float(np.mean(m_rets))
        avg_end_ret = float(np.mean(e_rets))

        gains = r_rets[r_rets > 0]
        losses = r_rets[r_rets < 0]
        gross_profit = float(np.sum(gains)) if len(gains) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = (
            round(gross_profit / gross_loss, 2)
            if gross_loss > 0
            else (9.99 if gross_profit > 0 else 0.0)
        )

        # Recent signal check
        recent_slice = sig_arr[max(0, n - recent_signal_days) : n]
        active_now = bool(np.any(recent_slice))
        active_today = bool(sig_arr[-1]) if n > 0 else False
        last_sig_date = dates[all_sig_raw[-1]].strftime("%Y-%m-%d") if len(all_sig_raw) > 0 else "-"

        # Reference bar for explicit current TP / SL price lines:
        # If there is an active signal within `recent_signal_days`, anchor TP/SL to that signal bar;
        # otherwise anchor to the latest bar (-1) so the user knows exact levels if entering today.
        if active_now and len(all_sig_raw) > 0:
            ref_bar = int(all_sig_raw[-1])
        else:
            ref_bar = n - 1

        ref_entry_price = float(close[ref_bar])
        cur_tp_pct = float(tp_arr[ref_bar])
        cur_sl_pct = float(sl_arr[ref_bar])
        cur_tp_price = round(ref_entry_price * (1.0 + cur_tp_pct / 100.0), 2)
        cur_sl_price = round(ref_entry_price * (1.0 + cur_sl_pct / 100.0), 2)
        rr_ratio = round(cur_tp_pct / max(0.5, abs(cur_sl_pct)), 2)

        confidence_weight = trade_count / (trade_count + 5.0)
        adj_hit_rate = hit_rate * confidence_weight + 50.0 * (1.0 - confidence_weight)
        fb_bonus = float(fb_map.get(cid, 0.0))
        composite_score = (0.55 * adj_hit_rate) + (0.45 * win_rate) + (avg_realized * 1.5) + fb_bonus

        plan = generate_trading_plan(
            combo=combo_tuple,
            entry_price=ref_entry_price,
            tp_price=cur_tp_price,
            sl_price=cur_sl_price,
            atr_val=float(atr_arr[ref_bar]),
            hold_days=hold_days,
        )

        records.append(
            {
                "Ticker": ticker,
                "ComboID": cid,
                "매매기법 조합": combo_name,
                "기법 수": num_strategies,
                "발생 횟수": trade_count,
                "목표 도달 확률(%)": round(hit_rate, 1),
                "순수 승률(%)": round(win_rate, 1),
                "기대 수익률(%)": round(avg_realized, 2),
                "기준 진입가($)": round(ref_entry_price, 2),
                "익절가($)": cur_tp_price,
                "익절폭(+%)": round(cur_tp_pct, 2),
                "손절가($)": cur_sl_price,
                "손절폭(-%)": round(cur_sl_pct, 2),
                "손익비(R:R)": rr_ratio,
                "평균 익절폭(+%)": round(avg_tp_pct, 2),
                "평균 손절폭(-%)": round(avg_sl_pct, 2),
                "평균 최고수익(%)": round(avg_max_ret, 2),
                "만기 평균수익(%)": round(avg_end_ret, 2),
                "손익비(PF)": profit_factor,
                "상대강도(RS%)": latest_rs,
                "피드백 가중치": round(fb_bonus, 2),
                "종합 점수": round(composite_score, 1),
                "최근 시그널": active_now,
                "당일 시그널": active_today,
                "마지막 발생일": last_sig_date,
                "익절선 기준": tp_rule_desc,
                "손절선 기준": sl_rule_desc,
                "매매 방법": plan["method_type"],
                "권장 진입가": plan["entry_zone"],
                "추격금지 상한": plan["max_chase"],
                "권장 비중": plan["rec_weight"],
                "매매 실행 가이드": plan["order_step"],
            }
        )

    stats_df = pd.DataFrame(records)
    if not stats_df.empty:
        stats_df.sort_values(
            by=["종합 점수", "목표 도달 확률(%)", "순수 승률(%)", "기대 수익률(%)"],
            ascending=[False, False, False, False],
            inplace=True,
        )
        stats_df.reset_index(drop=True, inplace=True)

    return stats_df, df_ind, combo_signals, outcomes_ctx


def get_trade_history_for_combo(
    df_ind: pd.DataFrame,
    combo_signals: Dict[str, np.ndarray],
    outcomes_ctx: Dict[str, Any],
    combo_id: str,
) -> pd.DataFrame:
    """
    Extract detailed trade-by-trade historical records for a specific strategy combination,
    including each trade's exact entry price, strategy-specific Take-Profit price ($ & +%),
    Stop-Loss price ($ & -%), and outcome.
    """
    if combo_id not in combo_signals or df_ind.empty:
        return pd.DataFrame()

    sig_arr = combo_signals[combo_id]
    all_idx = _deduplicate_indices(np.where(sig_arr)[0], cooldown=3)
    if len(all_idx) == 0:
        return pd.DataFrame()

    dates = df_ind.index
    close = df_ind["Close"].to_numpy(dtype=float)
    high = df_ind["High"].to_numpy(dtype=float)
    low = df_ind["Low"].to_numpy(dtype=float)
    n = len(close)

    combo_tuple = parse_combo_id(combo_id)
    hold_days = int(outcomes_ctx.get("hold_days", 10))
    use_dynamic = bool(outcomes_ctx.get("use_dynamic_tpsl", True))

    if use_dynamic and "base_tp_pct" in outcomes_ctx:
        tp_arr, sl_arr = get_combo_tp_sl_arrays(
            combo_tuple, outcomes_ctx["base_tp_pct"], outcomes_ctx["base_sl_pct"]
        )
    elif "tp_pct_arr" in outcomes_ctx:
        tp_arr = outcomes_ctx["tp_pct_arr"]
        sl_arr = outcomes_ctx["sl_pct_arr"]
    else:
        tp_arr = np.full(n, float(outcomes_ctx.get("fixed_target_pct", 4.0)), dtype=float)
        sl_arr = np.full(n, float(outcomes_ctx.get("fixed_stop_pct", -4.0)), dtype=float)

    eval_res = _evaluate_combo_on_bars(
        close=close,
        high=high,
        low=low,
        sig_indices=all_idx,
        tp_arr=tp_arr,
        sl_arr=sl_arr,
        hold_days=hold_days,
    )

    rows: List[Dict[str, Any]] = []
    for i, idx in enumerate(all_idx):
        entry_p = float(close[idx])
        t_pct = float(tp_arr[idx])
        s_pct = float(sl_arr[idx])
        tp_price = round(entry_p * (1.0 + t_pct / 100.0), 2)
        sl_price = round(entry_p * (1.0 + s_pct / 100.0), 2)

        is_eval = bool(eval_res["valid_eval"][i])
        t_hit = bool(eval_res["target_hit"][i]) if is_eval else None
        s_first = bool(eval_res["stop_hit_first"][i]) if is_eval else None
        p_win = bool(eval_res["pure_win"][i]) if is_eval else None

        if not is_eval:
            status = "진행중(최근 시그널)"
        elif p_win and t_hit:
            status = "익절선 도달(성공)"
        elif s_first:
            status = "손절선 이탈(손절)"
        elif p_win:
            status = "만기 수익마감"
        else:
            status = "만기 손실마감"

        rows.append(
            {
                "BarIndex": int(idx),
                "시그널 일자": dates[idx],
                "진입 종가($)": round(entry_p, 2),
                "익절 목표가($)": tp_price,
                "익절폭(+%)": round(t_pct, 2),
                "손절 기준가($)": sl_price,
                "손절폭(-%)": round(s_pct, 2),
                "결과 상태": status,
                "목표 도달": t_hit,
                "순수 승리": p_win,
                "실현 수익률(%)": round(float(eval_res["realized_ret"][i]), 2) if is_eval else np.nan,
                "기간내 최고(%)": round(float(eval_res["max_ret"][i]), 2) if is_eval else np.nan,
                "기간내 최저(%)": round(float(eval_res["min_ret"][i]), 2) if is_eval else np.nan,
                "만기 수익률(%)": round(float(eval_res["end_ret"][i]), 2) if is_eval else np.nan,
            }
        )

    return pd.DataFrame(rows)


def scan_market(
    stock_data: Dict[str, pd.DataFrame],
    constituents_df: pd.DataFrame,
    hold_days: int = 10,
    target_pct: float = 4.0,
    stop_pct: float = -4.0,
    signal_window: int = 2,
    min_trades: int = 5,
    recent_signal_days: int = 2,
    max_comb: int = 3,
    use_dynamic_tpsl: bool = True,
    rr_multiplier: float = 1.0,
    market_ctx_df: Optional[pd.DataFrame] = None,
    use_regime_filter: bool = False,
    vix_max: float = 26.0,
    use_rs_filter: bool = False,
    min_rs_pct: float = 0.0,
    feedback_bonus_map: Optional[Dict[str, float]] = None,
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scan multiple S&P 500 stocks and aggregate:
      1. `active_signals_df`: Stocks currently flashing an active signal with exact TP/SL prices
      2. `best_per_stock_df`: The #1 highest-probability strategy combination for each stock
      3. `leaderboard_df`: S&P 500 market-wide ranking of all 41 strategy combinations
    """
    meta_map = {}
    if constituents_df is not None and not constituents_df.empty:
        for _, row in constituents_df.iterrows():
            meta_map[str(row["Symbol"])] = {
                "Security": str(row.get("Security", row["Symbol"])),
                "Sector": str(row.get("Sector", "Unknown")),
            }

    all_stats_list: List[pd.DataFrame] = []
    best_per_stock: List[Dict[str, Any]] = []
    active_signals: List[Dict[str, Any]] = []

    tickers = list(stock_data.keys())
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        if progress_callback and (idx % 5 == 0 or idx == total - 1):
            progress_callback(
                (idx + 1) / max(1, total),
                f"매매기법별 맞춤 익절·손절 백테스팅 중... ({idx + 1}/{total}: {ticker})",
            )

        df = stock_data[ticker]
        stats_df, _, _, _ = analyze_single_stock(
            df=df,
            ticker=ticker,
            hold_days=hold_days,
            target_pct=target_pct,
            stop_pct=stop_pct,
            signal_window=signal_window,
            min_trades=min_trades,
            recent_signal_days=recent_signal_days,
            max_comb=max_comb,
            use_dynamic_tpsl=use_dynamic_tpsl,
            rr_multiplier=rr_multiplier,
            market_ctx_df=market_ctx_df,
            use_regime_filter=use_regime_filter,
            vix_max=vix_max,
            use_rs_filter=use_rs_filter,
            min_rs_pct=min_rs_pct,
            feedback_bonus_map=feedback_bonus_map,
        )
        if stats_df.empty:
            continue

        sec_info = meta_map.get(ticker, {"Security": ticker, "Sector": "-"})
        last_close = round(float(df["Close"].iloc[-1]), 2)
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        daily_chg = round((last_close / prev_close - 1.0) * 100.0, 2) if prev_close > 0 else 0.0

        stats_df["기업명"] = sec_info["Security"]
        stats_df["섹터"] = sec_info["Sector"]
        stats_df["현재가($)"] = last_close
        stats_df["당일등락(%)"] = daily_chg

        all_stats_list.append(stats_df)

        top_row = stats_df.iloc[0].to_dict()
        best_per_stock.append(top_row)

        active_sub = stats_df[stats_df["최근 시그널"] == True]
        if not active_sub.empty:
            best_active = active_sub.iloc[0].to_dict()
            best_active["활성 조합 수"] = int(len(active_sub))
            active_signals.append(best_active)

    active_signals_df = pd.DataFrame(active_signals)
    if not active_signals_df.empty:
        active_signals_df.sort_values(
            by=["종합 점수", "목표 도달 확률(%)", "순수 승률(%)", "기대 수익률(%)"],
            ascending=[False, False, False, False],
            inplace=True,
        )
        active_signals_df.reset_index(drop=True, inplace=True)

    best_per_stock_df = pd.DataFrame(best_per_stock)
    if not best_per_stock_df.empty:
        best_per_stock_df.sort_values(
            by=["종합 점수", "목표 도달 확률(%)", "순수 승률(%)", "기대 수익률(%)"],
            ascending=[False, False, False, False],
            inplace=True,
        )
        best_per_stock_df.reset_index(drop=True, inplace=True)

    if all_stats_list:
        combined_all = pd.concat(all_stats_list, ignore_index=True)
        leaderboard_df = (
            combined_all.groupby(["ComboID", "매매기법 조합", "기법 수"], as_index=False)
            .agg(
                검증종목수=("Ticker", "count"),
                총발생횟수=("발생 횟수", "sum"),
                평균_목표도달확률=("목표 도달 확률(%)", "mean"),
                평균_순수승률=("순수 승률(%)", "mean"),
                평균_기대수익률=("기대 수익률(%)", "mean"),
                평균_익절폭=("평균 익절폭(+%)", "mean"),
                평균_손절폭=("평균 손절폭(-%)", "mean"),
                승률70이상_종목수=("목표 도달 확률(%)", lambda s: int((s >= 70.0).sum())),
                현재시그널_종목수=("최근 시그널", lambda s: int(s.sum())),
            )
        )
        leaderboard_df["평균_목표도달확률"] = leaderboard_df["평균_목표도달확률"].round(1)
        leaderboard_df["평균_순수승률"] = leaderboard_df["평균_순수승률"].round(1)
        leaderboard_df["평균_기대수익률"] = leaderboard_df["평균_기대수익률"].round(2)
        leaderboard_df["평균_익절폭"] = leaderboard_df["평균_익절폭"].round(2)
        leaderboard_df["평균_손절폭"] = leaderboard_df["평균_손절폭"].round(2)

        breadth_weight = leaderboard_df["검증종목수"] / (leaderboard_df["검증종목수"] + 3.0)
        leaderboard_df["_rank_score"] = breadth_weight * (
            0.5 * leaderboard_df["평균_목표도달확률"]
            + 0.5 * leaderboard_df["평균_순수승률"]
            + leaderboard_df["평균_기대수익률"] * 2.0
        )
        leaderboard_df.sort_values(
            by=["_rank_score", "평균_목표도달확률", "평균_순수승률"],
            ascending=[False, False, False],
            inplace=True,
        )
        leaderboard_df.drop(columns=["_rank_score"], inplace=True)
        leaderboard_df.reset_index(drop=True, inplace=True)
    else:
        leaderboard_df = pd.DataFrame()

    return active_signals_df, best_per_stock_df, leaderboard_df
