"""Unit tests for src/feedback_db.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.feedback_db import (
    compute_4way_feedback_comparison,
    grade_pending_signals,
    normalize_supabase_url,
    save_signals_to_db,
)


def test_normalize_supabase_url() -> None:
    assert normalize_supabase_url("https://abc.supabase.co/rest/v1/") == "https://abc.supabase.co"
    assert normalize_supabase_url("https://abc.supabase.co/rest/v1") == "https://abc.supabase.co"
    assert normalize_supabase_url("  https://abc.supabase.co/  ") == "https://abc.supabase.co"


def test_compute_4way_feedback_comparison_legacy_without_scope() -> None:
    logs = pd.DataFrame(
        [
            {"ticker": "AAPL", "status": "WIN", "actual_return": 5.0, "regime_filter_on": True},
            {"ticker": "COIN", "status": "LOSS", "actual_return": -3.0, "regime_filter_on": False},
        ]
    )
    comp = compute_4way_feedback_comparison(logs)
    assert len(comp) == 4
    t1 = comp[comp["트랙"] == "Track 1"].iloc[0]
    t4 = comp[comp["트랙"] == "Track 4"].iloc[0]
    assert t1["채점 건수"] == 1
    assert t1["익절 적중률(%)"] == 100.0
    assert t4["채점 건수"] == 1
    assert t4["손절 이탈(LOSS)"] == 1


def _make_mock_supabase(pending_rows=None, all_graded_rows=None):
    sb = MagicMock()

    def table_side_effect(table_name: str):
        tbl = MagicMock()
        if table_name == "signal_logs":
            select_chain = MagicMock()
            tbl.select.return_value = select_chain
            select_chain.eq.return_value = select_chain
            select_chain.neq.return_value = select_chain
            select_chain.execute.side_effect = lambda: MagicMock(
                data=pending_rows if pending_rows is not None else []
            )
            tbl.update.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
            tbl.upsert.return_value.execute.return_value = MagicMock(data=[])
        elif table_name == "strategy_feedback":
            tbl.upsert.return_value.execute.return_value = MagicMock(data=[])
        return tbl

    sb.table.side_effect = table_side_effect
    return sb


def test_save_signals_to_db_filters_top50_and_pending() -> None:
    sb = MagicMock()
    pending_res = MagicMock(data=[{"ticker": "MSFT"}])
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value = (
        pending_res
    )
    sb.table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[])

    df = pd.DataFrame(
        [
            {
                "Ticker": "AAPL",
                "ComboID": "RSI_OVERSOLD",
                "매매기법 조합": "RSI 과매도",
                "마지막 발생일": "2026-11-20",
                "기준 진입가($)": 200.0,
                "익절가($)": 210.0,
                "손절가($)": 194.0,
                "목표 도달 확률(%)": 75.0,
            },
            {
                "Ticker": "MSFT",  # already pending -> skipped
                "ComboID": "RSI_OVERSOLD",
                "매매기법 조합": "RSI 과매도",
                "마지막 발생일": "2026-11-20",
                "기준 진입가($)": 400.0,
                "익절가($)": 420.0,
                "손절가($)": 388.0,
                "목표 도달 확률(%)": 80.0,
            },
            {
                "Ticker": "PLTR",  # not in TOP_50_TICKERS -> skipped for TOP50 track
                "ComboID": "RSI_OVERSOLD",
                "매매기법 조합": "RSI 과매도",
                "마지막 발생일": "2026-11-20",
                "기준 진입가($)": 50.0,
                "익절가($)": 55.0,
                "손절가($)": 47.0,
                "목표 도달 확률(%)": 85.0,
            },
        ]
    )

    saved = save_signals_to_db(sb, df, universe_scope="TOP50", min_prob=70.0)
    assert saved == 1


def test_grade_pending_signals_win_and_loss() -> None:
    pending_rows = [
        {
            "id": 1,
            "ticker": "AAPL",
            "combo_id": "C1",
            "combo_name": "Combo 1",
            "signal_date": "2026-11-10",
            "entry_price": 100.0,
            "tp_price": 105.0,
            "sl_price": 95.0,
            "hold_days": 5,
        }
    ]
    sb = _make_mock_supabase(pending_rows=pending_rows)

    dates = pd.bdate_range("2026-11-10", periods=4)
    ohlcv = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [101.0, 106.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 105.5, 102.0, 103.0],
            "Volume": [1_000_000] * 4,
        },
        index=dates,
    )

    with patch("src.feedback_db.fetch_single_stock", return_value=ohlcv), \
         patch("src.feedback_db.recalculate_strategy_feedback", return_value=pd.DataFrame()):
        summary = grade_pending_signals(sb)

    assert summary["checked"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
