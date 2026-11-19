"""
Automated Daily Feedback & Paper-Trading Job.
Designed to run automatically after US market close via GitHub Actions:
  1. Grades all PENDING predictions in Supabase `signal_logs` against actual OHLC bars.
  2. Recalculates `strategy_feedback` self-learning weights.
  3. Scans S&P 500 stocks under BOTH `시장국면 필터 ON` and `시장국면 필터 OFF`.
  4. Saves newly triggered >= 70% hit-rate signals (excluding imminent earnings) into `signal_logs`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtester import scan_market
from src.data_loader import (
    TOP_50_TICKERS,
    fetch_market_context,
    fetch_multiple_stocks,
    fetch_upcoming_earnings_days,
    get_sp500_constituents,
)
from src.feedback_db import (
    fetch_strategy_feedback,
    get_supabase_client,
    grade_pending_signals,
    load_saved_credentials,
    save_signals_to_db,
)


def main() -> None:
    url, key = load_saved_credentials()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL 또는 SUPABASE_KEY 환경 변수가 설정되어 있지 않습니다.")

    sb = get_supabase_client(url, key)

    print("=== [1/4] 기존 진행중(PENDING) 예측 자동 채점 시작 ===")
    grade_summary = grade_pending_signals(sb)
    print(
        f"채점 결과: 총 {grade_summary['checked']}건 확인 | "
        f"익절 성공 {grade_summary['wins']}건 | 손절 {grade_summary['losses']}건 | "
        f"만기 청산 {grade_summary['expired']}건 | 대기중 {grade_summary['still_pending']}건"
    )

    print("=== [2/4] S&P 500 주가 및 시장 국면(SPY/VIX) 데이터 수집 ===")
    constituents_df = get_sp500_constituents()
    scope = os.environ.get("SCAN_SCOPE", "all").strip().lower()
    tickers = (
        constituents_df["Symbol"].unique().tolist()
        if scope == "all" and not constituents_df.empty
        else TOP_50_TICKERS
    )
    print(f"스캔 대상 종목 수: {len(tickers)}개 (scope={scope})")

    market_ctx_df, market_summary = fetch_market_context(period="3y", use_cache=False)
    print(f"시장 국면: {market_summary}")

    stock_data = fetch_multiple_stocks(tickers=tickers, period="3y", use_cache=False)
    print(f"데이터 로드 완료: {len(stock_data)}개 종목")

    fb_bonus_map = {}
    try:
        fb_df = fetch_strategy_feedback(sb)
        if not fb_df.empty:
            fb_bonus_map = {
                str(r["combo_id"]): float(r["feedback_bonus"])
                for _, r in fb_df.iterrows()
            }
    except Exception:
        pass

    hold_days = 10
    min_prob = 70.0

    print("=== [3/4] 4대 트랙 (상위 50 / 전체 500 × 시장국면 ON / OFF) 스캔 ===")
    # Track 1 & 2: Top 50 universe
    stock_data_top50 = {t: stock_data[t] for t in TOP_50_TICKERS if t in stock_data}
    constituents_top50 = constituents_df[constituents_df["Symbol"].isin(TOP_50_TICKERS)]

    print(f"-> 상위 50개 데이터: {len(stock_data_top50)}개 종목")
    act_top50_on, _, _ = scan_market(
        stock_data=stock_data_top50,
        constituents_df=constituents_top50,
        hold_days=hold_days,
        use_dynamic_tpsl=True,
        market_ctx_df=market_ctx_df,
        use_regime_filter=True,
        vix_max=26.0,
        use_rs_filter=True,
        feedback_bonus_map=fb_bonus_map,
    )
    act_top50_off, _, _ = scan_market(
        stock_data=stock_data_top50,
        constituents_df=constituents_top50,
        hold_days=hold_days,
        use_dynamic_tpsl=True,
        market_ctx_df=market_ctx_df,
        use_regime_filter=False,
        vix_max=26.0,
        use_rs_filter=True,
        feedback_bonus_map=fb_bonus_map,
    )

    # Track 3 & 4: All S&P 500 universe
    print(f"-> S&P 500 전체 데이터: {len(stock_data)}개 종목")
    act_all_on, _, _ = scan_market(
        stock_data=stock_data,
        constituents_df=constituents_df,
        hold_days=hold_days,
        use_dynamic_tpsl=True,
        market_ctx_df=market_ctx_df,
        use_regime_filter=True,
        vix_max=26.0,
        use_rs_filter=True,
        feedback_bonus_map=fb_bonus_map,
    )
    act_all_off, _, _ = scan_market(
        stock_data=stock_data,
        constituents_df=constituents_df,
        hold_days=hold_days,
        use_dynamic_tpsl=True,
        market_ctx_df=market_ctx_df,
        use_regime_filter=False,
        vix_max=26.0,
        use_rs_filter=True,
        feedback_bonus_map=fb_bonus_map,
    )

    # Filter out stocks with upcoming earnings within hold_days + 3
    all_cands = list(
        set(act_top50_on["Ticker"].tolist() if not act_top50_on.empty else [])
        | set(act_top50_off["Ticker"].tolist() if not act_top50_off.empty else [])
        | set(act_all_on["Ticker"].tolist() if not act_all_on.empty else [])
        | set(act_all_off["Ticker"].tolist() if not act_all_off.empty else [])
    )
    earnings_map = fetch_upcoming_earnings_days(all_cands) if all_cands else {}

    def _filter_earnings(df):
        if df is None or df.empty:
            return df
        safe = df["Ticker"].map(
            lambda t: earnings_map.get(t) is None or int(earnings_map[t]) > (hold_days + 3)
        )
        return df[safe].reset_index(drop=True)

    act_top50_on = _filter_earnings(act_top50_on)
    act_top50_off = _filter_earnings(act_top50_off)
    act_all_on = _filter_earnings(act_all_on)
    act_all_off = _filter_earnings(act_all_off)

    print("=== [4/4] 4대 트랙 70% 이상 고확률 시그널 Supabase 저장 ===")
    saved_top50_on = save_signals_to_db(
        sb, act_top50_on, hold_days=hold_days, min_prob=min_prob,
        regime_filter_on=True, rs_filter_on=True, universe_scope="TOP50",
    )
    saved_top50_off = save_signals_to_db(
        sb, act_top50_off, hold_days=hold_days, min_prob=min_prob,
        regime_filter_on=False, rs_filter_on=True, universe_scope="TOP50",
    )
    saved_all_on = save_signals_to_db(
        sb, act_all_on, hold_days=hold_days, min_prob=min_prob,
        regime_filter_on=True, rs_filter_on=True, universe_scope="ALL",
    )
    saved_all_off = save_signals_to_db(
        sb, act_all_off, hold_days=hold_days, min_prob=min_prob,
        regime_filter_on=False, rs_filter_on=True, universe_scope="ALL",
    )
    print(
        f"4대 트랙 저장 완료:\n"
        f"  1) 상위 50 종목 · 국면 ON : {saved_top50_on}건\n"
        f"  2) 상위 50 종목 · 국면 OFF: {saved_top50_off}건\n"
        f"  3) 전체 500 종목 · 국면 ON : {saved_all_on}건\n"
        f"  4) 전체 500 종목 · 국면 OFF: {saved_all_off}건"
    )


if __name__ == "__main__":
    main()

