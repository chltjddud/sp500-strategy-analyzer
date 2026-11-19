"""
Supabase Feedback & Self-Learning Database Integration.
Stores signal predictions, grades outcomes against real subsequent market prices,
and computes self-feedback bonus/penalty weights per strategy combination.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data_loader import BASE_DIR, fetch_single_stock

SECRETS_FILE = BASE_DIR / ".streamlit" / "secrets.toml"
ENV_FILE = BASE_DIR / ".env"


def normalize_supabase_url(url: str) -> str:
    """Ensure Supabase URL is the base project URL without /rest/v1/ suffix."""
    cleaned = url.strip()
    if cleaned.endswith("/rest/v1/"):
        cleaned = cleaned[:-9]
    elif cleaned.endswith("/rest/v1"):
        cleaned = cleaned[:-8]
    return cleaned.rstrip("/")


def load_saved_credentials() -> Tuple[str, str]:
    """Load SUPABASE_URL and SUPABASE_KEY from env, Streamlit st.secrets, .env, or secrets.toml."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        try:
            import streamlit as st

            if not url and "SUPABASE_URL" in st.secrets:
                url = str(st.secrets["SUPABASE_URL"])
            if not key and "SUPABASE_KEY" in st.secrets:
                key = str(st.secrets["SUPABASE_KEY"])
        except Exception:
            pass

    # Check .env first
    if ENV_FILE.exists() and (not url or not key):
        try:
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "SUPABASE_URL" and not url:
                        url = v
                    elif k == "SUPABASE_KEY" and not key:
                        key = v
        except Exception:
            pass

    # Check .streamlit/secrets.toml next
    if SECRETS_FILE.exists() and (not url or not key):
        try:
            for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == "SUPABASE_URL" and not url:
                        url = v
                    elif k == "SUPABASE_KEY" and not key:
                        key = v
        except Exception:
            pass

    return normalize_supabase_url(url), key


def save_credentials(url: str, key: str) -> None:
    """Persist SUPABASE_URL and SUPABASE_KEY into both .env and .streamlit/secrets.toml."""
    norm_url = normalize_supabase_url(url)
    clean_key = key.strip()

    # Save to .env
    env_content = (
        f'SUPABASE_URL={norm_url}\n'
        f'SUPABASE_KEY={clean_key}\n'
    )
    ENV_FILE.write_text(env_content, encoding="utf-8")

    # Save to .streamlit/secrets.toml
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f'SUPABASE_URL = "{norm_url}"\n'
        f'SUPABASE_KEY = "{clean_key}"\n'
    )
    SECRETS_FILE.write_text(content, encoding="utf-8")


def get_supabase_client(url: str, key: str):
    """Create and return a Supabase client instance."""
    from supabase import create_client

    if not url or not key:
        raise ValueError("Supabase URL과 API Key를 모두 입력해 주세요.")
    return create_client(normalize_supabase_url(url), key.strip())


def test_connection(url: str, key: str) -> Tuple[bool, str]:
    """Verify Supabase credentials and confirm tables exist."""
    try:
        sb = get_supabase_client(url, key)
        sb.table("signal_logs").select("id").limit(1).execute()
        return True, "Supabase 연결 및 테이블 확인 완료"
    except Exception as exc:
        msg = str(exc)
        if "relation" in msg and "does not exist" in msg:
            return False, "연결은 되었으나 테이블이 없습니다. supabase_schema.sql 을 SQL Editor에서 먼저 실행해 주세요."
        return False, f"연결 실패: {msg}"


def save_signals_to_db(
    sb,
    signals_df: pd.DataFrame,
    hold_days: int = 10,
    min_prob: float = 70.0,
    regime_filter_on: bool = True,
    rs_filter_on: bool = True,
    max_rows: int = 50,
) -> int:
    """
    Save active signals meeting `min_prob` (e.g. >= 70%) into Supabase `signal_logs`
    as simulated paper-trades (assumed bought at `entry_price`, exited at `tp_price` or `sl_price`).
    Tracks `regime_filter_on` separately so ON vs OFF versions can be compared.
    """
    if signals_df is None or signals_df.empty:
        return 0

    qualified = signals_df[signals_df["목표 도달 확률(%)"] >= float(min_prob)].head(max_rows)
    if qualified.empty:
        return 0

    rows_to_save: List[Dict[str, Any]] = []
    for _, row in qualified.iterrows():
        sig_date = str(row.get("마지막 발생일", "")).strip()
        if not sig_date or sig_date == "-":
            continue
        rows_to_save.append(
            {
                "ticker": str(row["Ticker"]),
                "combo_id": str(row["ComboID"]),
                "combo_name": str(row["매매기법 조합"]),
                "signal_date": sig_date,
                "entry_price": float(row["기준 진입가($)"]),
                "tp_price": float(row["익절가($)"]),
                "sl_price": float(row["손절가($)"]),
                "hold_days": int(hold_days),
                "predicted_hit_rate": float(row["목표 도달 확률(%)"]),
                "regime_filter_on": bool(regime_filter_on),
                "rs_filter_on": bool(rs_filter_on),
                "status": "PENDING",
            }
        )

    if not rows_to_save:
        return 0

    try:
        sb.table("signal_logs").upsert(
            rows_to_save,
            on_conflict="ticker,combo_id,signal_date,regime_filter_on",
            ignore_duplicates=True,
        ).execute()
    except Exception:
        # Fallback if user hasn't run the ALTER TABLE migration yet
        base_rows = [
            {k: v for k, v in r.items() if k not in ("regime_filter_on", "rs_filter_on")}
            for r in rows_to_save
        ]
        sb.table("signal_logs").upsert(
            base_rows,
            on_conflict="ticker,combo_id,signal_date",
            ignore_duplicates=True,
        ).execute()

    return len(rows_to_save)


def grade_pending_signals(sb) -> Dict[str, int]:
    """
    Fetch all PENDING predictions from `signal_logs`, evaluate them against actual
    subsequent daily bars from Yahoo Finance, update `signal_logs`, and recalculate
    `strategy_feedback` weights.
    """
    res = sb.table("signal_logs").select("*").eq("status", "PENDING").execute()
    pending_rows = res.data or []
    summary = {"checked": len(pending_rows), "wins": 0, "losses": 0, "expired": 0, "still_pending": 0}

    if not pending_rows:
        return summary

    # Group by ticker to minimize downloads
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for r in pending_rows:
        by_ticker.setdefault(r["ticker"], []).append(r)

    now_iso = datetime.now(timezone.utc).isoformat()

    for ticker, items in by_ticker.items():
        df = fetch_single_stock(ticker, period="3mo", use_cache=False)
        if df.empty:
            summary["still_pending"] += len(items)
            continue

        for item in items:
            sig_dt = pd.to_datetime(item["signal_date"])
            fwd = df[df.index > sig_dt]
            if fwd.empty:
                summary["still_pending"] += 1
                continue

            hold_days = int(item.get("hold_days", 10))
            window = fwd.iloc[:hold_days]

            entry_p = float(item["entry_price"])
            tp_p = float(item["tp_price"])
            sl_p = float(item["sl_price"])

            new_status: Optional[str] = None
            exit_p: Optional[float] = None
            exit_dt_str: Optional[str] = None
            days_held: Optional[int] = None
            actual_ret: Optional[float] = None

            for day_idx, (bar_dt, bar) in enumerate(window.iterrows(), start=1):
                bar_high = float(bar["High"])
                bar_low = float(bar["Low"])
                bar_close = float(bar["Close"])

                hit_tp = bar_high >= tp_p
                hit_sl = bar_low <= sl_p

                if hit_tp and not hit_sl:
                    new_status = "WIN"
                    exit_p = tp_p
                    exit_dt_str = pd.to_datetime(bar_dt).strftime("%Y-%m-%d")
                    days_held = day_idx
                    break
                elif hit_sl and not hit_tp:
                    new_status = "LOSS"
                    exit_p = sl_p
                    exit_dt_str = pd.to_datetime(bar_dt).strftime("%Y-%m-%d")
                    days_held = day_idx
                    break
                elif hit_tp and hit_sl:
                    if bar_close >= entry_p:
                        new_status = "WIN"
                        exit_p = tp_p
                    else:
                        new_status = "LOSS"
                        exit_p = sl_p
                    exit_dt_str = pd.to_datetime(bar_dt).strftime("%Y-%m-%d")
                    days_held = day_idx
                    break

            if new_status is None:
                if len(fwd) >= hold_days:
                    last_close = float(window["Close"].iloc[-1])
                    exit_p = last_close
                    exit_dt_str = pd.to_datetime(window.index[-1]).strftime("%Y-%m-%d")
                    days_held = len(window)
                    actual_ret = round((last_close / entry_p - 1.0) * 100.0, 2)
                    new_status = "EXPIRED_WIN" if actual_ret > 0 else "EXPIRED_LOSS"
                    summary["expired"] += 1
                else:
                    summary["still_pending"] += 1
                    continue
            else:
                actual_ret = round((exit_p / entry_p - 1.0) * 100.0, 2)
                if new_status == "WIN":
                    summary["wins"] += 1
                else:
                    summary["losses"] += 1

            update_payload: Dict[str, Any] = {
                "status": new_status,
                "exit_price": round(float(exit_p), 2),
                "actual_return": actual_ret,
                "exit_date": exit_dt_str,
                "days_held": days_held,
                "graded_at": now_iso,
            }
            try:
                sb.table("signal_logs").update(update_payload).eq("id", item["id"]).execute()
            except Exception:
                fallback_payload = {
                    k: v for k, v in update_payload.items() if k not in ("exit_date", "days_held")
                }
                sb.table("signal_logs").update(fallback_payload).eq("id", item["id"]).execute()

    recalculate_strategy_feedback(sb)
    return summary


def recalculate_strategy_feedback(sb) -> pd.DataFrame:
    """
    Aggregate all graded `signal_logs` per `combo_id` and update `strategy_feedback`
    with live win rates and feedback bonus/penalty scores.
    """
    res = sb.table("signal_logs").select("*").neq("status", "PENDING").execute()
    graded = res.data or []
    if not graded:
        return pd.DataFrame()

    df = pd.DataFrame(graded)
    now_iso = datetime.now(timezone.utc).isoformat()
    feedback_rows: List[Dict[str, Any]] = []

    for combo_id, grp in df.groupby("combo_id"):
        total = int(len(grp))
        wins = int(grp["status"].isin(["WIN", "EXPIRED_WIN"]).sum())
        losses = total - wins
        live_wr = round((wins / total) * 100.0, 1) if total > 0 else 0.0
        avg_ret = round(float(grp["actual_return"].dropna().mean()), 2) if total > 0 else 0.0
        # Self-feedback bonus: compares live win rate vs 50% baseline, scaled by confidence
        confidence = min(1.0, total / 10.0)
        bonus = round(((live_wr - 50.0) * 0.25 + avg_ret * 0.8) * confidence, 2)

        feedback_rows.append(
            {
                "combo_id": str(combo_id),
                "combo_name": str(grp["combo_name"].iloc[0]),
                "total_graded": total,
                "wins": wins,
                "losses": losses,
                "live_win_rate": live_wr,
                "avg_live_return": avg_ret,
                "feedback_bonus": bonus,
                "updated_at": now_iso,
            }
        )

    if feedback_rows:
        sb.table("strategy_feedback").upsert(feedback_rows, on_conflict="combo_id").execute()
    return pd.DataFrame(feedback_rows)


def fetch_signal_logs(sb, limit: int = 200) -> pd.DataFrame:
    """Fetch recent prediction logs from Supabase."""
    res = (
        sb.table("signal_logs")
        .select("*")
        .order("signal_date", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(res.data or [])


def fetch_strategy_feedback(sb) -> pd.DataFrame:
    """Fetch aggregated self-feedback weights per strategy combination."""
    res = (
        sb.table("strategy_feedback")
        .select("*")
        .order("feedback_bonus", desc=True)
        .execute()
    )
    return pd.DataFrame(res.data or [])


def fetch_baseline_count(sb) -> int:
    """Return number of Colab baseline records stored in `strategy_baseline`."""
    res = sb.table("strategy_baseline").select("ticker", count="exact").limit(1).execute()
    return int(res.count or 0)


def compute_regime_comparison(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare live paper-trading accuracy between `시장국면 필터 ON` and `시장국면 필터 OFF`.
    """
    if logs_df is None or logs_df.empty or "regime_filter_on" not in logs_df.columns:
        return pd.DataFrame()

    graded = logs_df[logs_df["status"] != "PENDING"].copy()
    if graded.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for flag_val, grp in graded.groupby("regime_filter_on"):
        total = int(len(grp))
        tp_wins = int((grp["status"] == "WIN").sum())
        all_wins = int(grp["status"].isin(["WIN", "EXPIRED_WIN"]).sum())
        sl_losses = int((grp["status"] == "LOSS").sum())
        tp_hit_rate = round((tp_wins / total) * 100.0, 1) if total > 0 else 0.0
        total_win_rate = round((all_wins / total) * 100.0, 1) if total > 0 else 0.0
        avg_ret = round(float(grp["actual_return"].dropna().mean()), 2) if total > 0 else 0.0
        avg_days = (
            round(float(grp["days_held"].dropna().mean()), 1)
            if "days_held" in grp.columns and grp["days_held"].notna().any()
            else 0.0
        )
        rows.append(
            {
                "필터 버전": "시장국면 필터 ON (안정장만)" if bool(flag_val) else "시장국면 필터 OFF (전체 장세)",
                "채점 건수": total,
                "익절 도달(WIN)": tp_wins,
                "손절 이탈(LOSS)": sl_losses,
                "익절 적중률(%)": tp_hit_rate,
                "전체 승률(%)": total_win_rate,
                "평균 수익률(%)": avg_ret,
                "평균 보유일(일)": avg_days,
            }
        )
    return pd.DataFrame(rows)

