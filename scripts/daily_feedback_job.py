"""
Automated Daily Feedback & Paper-Trading Job.
Designed to run automatically after US market close via GitHub Actions:
  1. Grades all PENDING predictions in Supabase `signal_logs` against actual OHLC bars.
  2. Fetches S&P 500 & SPY/VIX data and skips new signal recording on US market holidays.
  3. Scans S&P 500 stocks under BOTH `시장국면 필터 ON` and `시장국면 필터 OFF`.
  4. Saves newly triggered >= 70% hit-rate signals (excluding imminent earnings) into `signal_logs`.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def is_us_market_trading_day(
    market_ctx_df: pd.DataFrame,
    now_et: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Check whether the latest bar in `market_ctx_df` matches today's date in US/Eastern.
    Returns (is_trading_day, reason_detail).
    """
    if market_ctx_df is None or market_ctx_df.empty:
        return False, "SPY 시장 데이터가 비어 있습니다."

    et_now = now_et or datetime.now(ZoneInfo("America/New_York"))
    last_bar_date = pd.to_datetime(market_ctx_df.index[-1]).date()
    today_et = et_now.date()

    if last_bar_date == today_et:
        return True, f"미국 거래일 확인 ({today_et})"
    return False, f"미국 휴장일 또는 장 데이터 미갱신 (최근 바: {last_bar_date}, 미국 오늘: {today_et})"


def main() -> None:
    job_t0 = time.perf_counter()
    url, key = load_saved_credentials()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL 또는 SUPABASE_KEY 환경 변수가 설정되어 있지 않습니다.")

    sb = get_supabase_client(url, key)

    logger.info("=== [1/4] 기존 진행중(PENDING) 예측 자동 채점 시작 ===")
    t0 = time.perf_counter()
    grade_summary = grade_pending_signals(sb)
    logger.info(
        "채점 완료 (%.1fs): 총 %d건 확인 | 익절 성공 %d건 | 손절 %d건 | 만기 청산 %d건 | 대기중 %d건",
        time.perf_counter() - t0,
        grade_summary["checked"],
        grade_summary["wins"],
        grade_summary["losses"],
        grade_summary["expired"],
        grade_summary["still_pending"],
    )

    logger.info("=== [2/4] S&P 500 주가 및 시장 국면(SPY/VIX) 데이터 수집 ===")
    t0 = time.perf_counter()
    market_ctx_df, market_summary = fetch_market_context(period="3y", use_cache=False)
    logger.info("시장 국면: %s", market_summary)

    is_open, market_msg = is_us_market_trading_day(market_ctx_df)
    force_run = os.environ.get("FORCE_RUN", "").strip().lower() in ("1", "true", "yes")
    if not is_open and not force_run:
        logger.info("%s -> 신규 시그널 저장을 건너뜁니다. (강제 실행은 FORCE_RUN=1)", market_msg)
        return
    if not is_open and force_run:
        logger.warning("%s 이지만 FORCE_RUN=1 설정으로 계속 진행합니다.", market_msg)

    constituents_df = get_sp500_constituents()
    scope = os.environ.get("SCAN_SCOPE", "all").strip().lower()
    tickers = (
        constituents_df["Symbol"].unique().tolist()
        if scope == "all" and not constituents_df.empty
        else TOP_50_TICKERS
    )
    logger.info("스캔 대상 종목 수: %d개 (scope=%s)", len(tickers), scope)

    stock_data = fetch_multiple_stocks(tickers=tickers, period="3y", use_cache=False)
    logger.info("데이터 로드 완료 (%.1fs): %d/%d개 종목", time.perf_counter() - t0, len(stock_data), len(tickers))

    fb_bonus_map = {}
    try:
        fb_df = fetch_strategy_feedback(sb)
        if not fb_df.empty:
            fb_bonus_map = {
                str(r["combo_id"]): float(r["feedback_bonus"])
                for _, r in fb_df.iterrows()
            }
    except Exception as exc:
        logger.warning("기존 피드백 가중치 로드 실패 (기본값 사용): %s", exc)

    hold_days = 10
    min_prob = 70.0

    logger.info("=== [3/4] 4대 트랙 (상위 50 / 전체 500 × 시장국면 ON / OFF) 스캔 ===")
    t0 = time.perf_counter()
    stock_data_top50 = {t: stock_data[t] for t in TOP_50_TICKERS if t in stock_data}
    constituents_top50 = constituents_df[constituents_df["Symbol"].isin(TOP_50_TICKERS)]

    logger.info("-> 상위 50개 데이터: %d개 종목", len(stock_data_top50))
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

    logger.info("-> S&P 500 전체 데이터: %d개 종목", len(stock_data))
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
    logger.info("4대 트랙 스캔 완료 (%.1fs)", time.perf_counter() - t0)

    # Filter out stocks with upcoming earnings within hold_days + 3
    all_cands = list(
        set(act_top50_on["Ticker"].tolist() if not act_top50_on.empty else [])
        | set(act_top50_off["Ticker"].tolist() if not act_top50_off.empty else [])
        | set(act_all_on["Ticker"].tolist() if not act_all_on.empty else [])
        | set(act_all_off["Ticker"].tolist() if not act_all_off.empty else [])
    )
    earnings_map = fetch_upcoming_earnings_days(all_cands) if all_cands else {}

    def _filter_earnings(df: pd.DataFrame) -> pd.DataFrame:
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

    logger.info("=== [4/4] 4대 트랙 70% 이상 고확률 시그널 Supabase 저장 ===")
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
    logger.info(
        "4대 트랙 저장 완료 (총 소요 %.1fs): TOP50 ON=%d, TOP50 OFF=%d, ALL ON=%d, ALL OFF=%d",
        time.perf_counter() - job_t0,
        saved_top50_on,
        saved_top50_off,
        saved_all_on,
        saved_all_off,
    )


if __name__ == "__main__":
    main()
