"""
S&P 500 Ticker & Historical OHLCV Data Loader with Local Caching.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CONSTITUENTS_FILE = DATA_DIR / "sp500_constituents.csv"

# Top 50 representative S&P 500 stocks by market cap across major sectors
TOP_50_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "AVGO", "LLY", "TSLA",
    "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "WMT", "JNJ",
    "NFLX", "ABBV", "CRM", "BAC", "ORCL", "MRK", "CVX", "AMD", "KO", "PEP",
    "CSCO", "TMO", "ACN", "LIN", "MCD", "ADBE", "WFC", "IBM", "PM", "ABT",
    "GE", "CAT", "ISRG", "TXN", "INTU", "QCOM", "VZ", "GS", "AMGN", "BKNG",
]

# Built-in fallback metadata in case Wikipedia is unreachable
FALLBACK_SP500: List[Dict[str, str]] = [
    {"Symbol": "AAPL", "Security": "Apple Inc.", "Sector": "Information Technology"},
    {"Symbol": "MSFT", "Security": "Microsoft Corp.", "Sector": "Information Technology"},
    {"Symbol": "NVDA", "Security": "NVIDIA Corp.", "Sector": "Information Technology"},
    {"Symbol": "AMZN", "Security": "Amazon.com Inc.", "Sector": "Consumer Discretionary"},
    {"Symbol": "META", "Security": "Meta Platforms", "Sector": "Communication Services"},
    {"Symbol": "GOOGL", "Security": "Alphabet Inc. (Class A)", "Sector": "Communication Services"},
    {"Symbol": "BRK-B", "Security": "Berkshire Hathaway", "Sector": "Financials"},
    {"Symbol": "AVGO", "Security": "Broadcom Inc.", "Sector": "Information Technology"},
    {"Symbol": "LLY", "Security": "Eli Lilly & Co.", "Sector": "Health Care"},
    {"Symbol": "TSLA", "Security": "Tesla Inc.", "Sector": "Consumer Discretionary"},
    {"Symbol": "JPM", "Security": "JPMorgan Chase & Co.", "Sector": "Financials"},
    {"Symbol": "V", "Security": "Visa Inc.", "Sector": "Financials"},
    {"Symbol": "UNH", "Security": "UnitedHealth Group", "Sector": "Health Care"},
    {"Symbol": "XOM", "Security": "Exxon Mobil Corp.", "Sector": "Energy"},
    {"Symbol": "MA", "Security": "Mastercard Inc.", "Sector": "Financials"},
    {"Symbol": "COST", "Security": "Costco Wholesale", "Sector": "Consumer Staples"},
    {"Symbol": "HD", "Security": "Home Depot", "Sector": "Consumer Discretionary"},
    {"Symbol": "PG", "Security": "Procter & Gamble", "Sector": "Consumer Staples"},
    {"Symbol": "WMT", "Security": "Walmart Inc.", "Sector": "Consumer Staples"},
    {"Symbol": "JNJ", "Security": "Johnson & Johnson", "Sector": "Health Care"},
    {"Symbol": "NFLX", "Security": "Netflix Inc.", "Sector": "Communication Services"},
    {"Symbol": "ABBV", "Security": "AbbVie Inc.", "Sector": "Health Care"},
    {"Symbol": "CRM", "Security": "Salesforce Inc.", "Sector": "Information Technology"},
    {"Symbol": "BAC", "Security": "Bank of America", "Sector": "Financials"},
    {"Symbol": "ORCL", "Security": "Oracle Corp.", "Sector": "Information Technology"},
    {"Symbol": "MRK", "Security": "Merck & Co.", "Sector": "Health Care"},
    {"Symbol": "CVX", "Security": "Chevron Corp.", "Sector": "Energy"},
    {"Symbol": "AMD", "Security": "Advanced Micro Devices", "Sector": "Information Technology"},
    {"Symbol": "KO", "Security": "Coca-Cola Co.", "Sector": "Consumer Staples"},
    {"Symbol": "PEP", "Security": "PepsiCo Inc.", "Sector": "Consumer Staples"},
    {"Symbol": "CSCO", "Security": "Cisco Systems", "Sector": "Information Technology"},
    {"Symbol": "TMO", "Security": "Thermo Fisher Scientific", "Sector": "Health Care"},
    {"Symbol": "ACN", "Security": "Accenture plc", "Sector": "Information Technology"},
    {"Symbol": "LIN", "Security": "Linde plc", "Sector": "Materials"},
    {"Symbol": "MCD", "Security": "McDonald's Corp.", "Sector": "Consumer Discretionary"},
    {"Symbol": "ADBE", "Security": "Adobe Inc.", "Sector": "Information Technology"},
    {"Symbol": "WFC", "Security": "Wells Fargo & Co.", "Sector": "Financials"},
    {"Symbol": "IBM", "Security": "IBM Corp.", "Sector": "Information Technology"},
    {"Symbol": "PM", "Security": "Philip Morris International", "Sector": "Consumer Staples"},
    {"Symbol": "ABT", "Security": "Abbott Laboratories", "Sector": "Health Care"},
    {"Symbol": "GE", "Security": "GE Aerospace", "Sector": "Industrials"},
    {"Symbol": "CAT", "Security": "Caterpillar Inc.", "Sector": "Industrials"},
    {"Symbol": "ISRG", "Security": "Intuitive Surgical", "Sector": "Health Care"},
    {"Symbol": "TXN", "Security": "Texas Instruments", "Sector": "Information Technology"},
    {"Symbol": "INTU", "Security": "Intuit Inc.", "Sector": "Information Technology"},
    {"Symbol": "QCOM", "Security": "Qualcomm Inc.", "Sector": "Information Technology"},
    {"Symbol": "VZ", "Security": "Verizon Communications", "Sector": "Communication Services"},
    {"Symbol": "GS", "Security": "Goldman Sachs", "Sector": "Financials"},
    {"Symbol": "AMGN", "Security": "Amgen Inc.", "Sector": "Health Care"},
    {"Symbol": "BKNG", "Security": "Booking Holdings", "Sector": "Consumer Discretionary"},
    {"Symbol": "NEE", "Security": "NextEra Energy", "Sector": "Utilities"},
    {"Symbol": "PLD", "Security": "Prologis Inc.", "Sector": "Real Estate"},
    {"Symbol": "FCX", "Security": "Freeport-McMoRan", "Sector": "Materials"},
]


def ensure_dirs() -> None:
    """Ensure data and cache directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_ticker(symbol: str) -> str:
    """Convert S&P 500 ticker format (e.g. BRK.B) to Yahoo Finance format (BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch S&P 500 constituents (Symbol, Security, Sector) from Wikipedia
    or local cache / built-in fallback.
    """
    ensure_dirs()
    if CONSTITUENTS_FILE.exists() and not force_refresh:
        try:
            df = pd.read_csv(CONSTITUENTS_FILE)
            if {"Symbol", "Security", "Sector"}.issubset(df.columns) and len(df) > 40:
                return df
        except Exception:
            pass

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.rename(columns={"GICS Sector": "Sector"}, inplace=True)
        df["Symbol"] = df["Symbol"].astype(str).map(normalize_ticker)
        df.to_csv(CONSTITUENTS_FILE, index=False)
        return df
    except Exception:
        df = pd.DataFrame(FALLBACK_SP500)
        df.to_csv(CONSTITUENTS_FILE, index=False)
        return df


def _cache_path(ticker: str, period: str) -> Path:
    safe_ticker = normalize_ticker(ticker).replace("^", "")
    return CACHE_DIR / f"{safe_ticker}_{period}.csv"


def _is_cache_fresh(path: Path, max_age_hours: float = 12.0) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize OHLCV DataFrame columns and index."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    out = df[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.sort_index(inplace=True)
    out = out[~out.index.duplicated(keep="last")]
    return out


def _read_existing_cache(cpath: Path) -> pd.DataFrame:
    if not cpath.exists():
        return pd.DataFrame()
    try:
        return _clean_ohlcv(pd.read_csv(cpath, index_col=0, parse_dates=True))
    except Exception:
        return pd.DataFrame()


def _merge_and_save_cache(cpath: Path, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        merged = new_df
    elif new_df.empty:
        merged = existing_df
    else:
        merged = pd.concat([existing_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if not merged.empty:
        merged.to_csv(cpath)
    return merged


def fetch_single_stock(
    ticker: str,
    period: str = "3y",
    use_cache: bool = True,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single ticker with incremental local CSV cache."""
    ensure_dirs()
    norm_ticker = normalize_ticker(ticker)
    cpath = _cache_path(norm_ticker, period)

    existing_df = _read_existing_cache(cpath)
    if use_cache and _is_cache_fresh(cpath, max_age_hours=max_age_hours) and not existing_df.empty:
        return existing_df

    # Incremental update if existing cache is valid and recent (within 21 days)
    inc_start: Optional[str] = None
    if len(existing_df) >= 60:
        last_dt = pd.to_datetime(existing_df.index[-1])
        if (datetime.now() - last_dt.to_pydatetime()).days <= 21:
            inc_start = (last_dt - timedelta(days=4)).strftime("%Y-%m-%d")

    try:
        if inc_start:
            raw = yf.download(
                norm_ticker,
                start=inc_start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            cleaned_new = _clean_ohlcv(raw)
            return _merge_and_save_cache(cpath, existing_df, cleaned_new)

        raw = yf.download(
            norm_ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        cleaned = _clean_ohlcv(raw)
        if not cleaned.empty:
            cleaned.to_csv(cpath)
        return cleaned
    except Exception:
        if not existing_df.empty:
            return existing_df
        return pd.DataFrame()


def fetch_multiple_stocks(
    tickers: List[str],
    period: str = "3y",
    use_cache: bool = True,
    max_age_hours: float = 12.0,
    batch_size: int = 50,
    progress_callback=None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for multiple tickers efficiently using incremental local cache
    and batched yfinance downloads.
    """
    ensure_dirs()
    results: Dict[str, pd.DataFrame] = {}
    inc_items: Dict[str, pd.DataFrame] = {}
    full_tickers: List[str] = []

    norm_tickers = [normalize_ticker(t) for t in tickers if t.strip()]
    norm_tickers = list(dict.fromkeys(norm_tickers))

    for t in norm_tickers:
        cpath = _cache_path(t, period)
        existing_df = _read_existing_cache(cpath)
        if use_cache and _is_cache_fresh(cpath, max_age_hours=max_age_hours) and len(existing_df) >= 60:
            results[t] = existing_df
            continue
        if len(existing_df) >= 60:
            last_dt = pd.to_datetime(existing_df.index[-1])
            if (datetime.now() - last_dt.to_pydatetime()).days <= 21:
                inc_items[t] = existing_df
                continue
        full_tickers.append(t)

    if not inc_items and not full_tickers:
        if progress_callback:
            progress_callback(1.0, f"캐시된 데이터 {len(results)}개 종목 로드 완료")
        return results

    # 1) Incremental batch download for stale caches
    if inc_items:
        min_last_dt = min(pd.to_datetime(df.index[-1]) for df in inc_items.values())
        inc_start = (min_last_dt - timedelta(days=4)).strftime("%Y-%m-%d")
        inc_list = list(inc_items.keys())
        inc_batches = (len(inc_list) + batch_size - 1) // batch_size
        for b_idx in range(inc_batches):
            batch = inc_list[b_idx * batch_size : (b_idx + 1) * batch_size]
            if progress_callback:
                progress_callback(
                    (b_idx + 0.5) / max(1, inc_batches + 1),
                    f"증분 데이터 갱신 중... ({min(len(inc_list), (b_idx + 1) * batch_size)}/{len(inc_list)} 종목)",
                )
            try:
                raw = yf.download(
                    batch,
                    start=inc_start,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if len(batch) == 1:
                    t = batch[0]
                    merged = _merge_and_save_cache(_cache_path(t, period), inc_items[t], _clean_ohlcv(raw))
                    if len(merged) >= 60:
                        results[t] = merged
                else:
                    for t in batch:
                        new_sub = _clean_ohlcv(raw[t].copy()) if t in raw.columns.get_level_values(0) else pd.DataFrame()
                        merged = _merge_and_save_cache(_cache_path(t, period), inc_items[t], new_sub)
                        if len(merged) >= 60:
                            results[t] = merged
            except Exception:
                for t in batch:
                    results[t] = inc_items[t]

    # 2) Full batch download for uncached tickers
    total_batches = (len(full_tickers) + batch_size - 1) // batch_size
    for b_idx in range(total_batches):
        batch = full_tickers[b_idx * batch_size : (b_idx + 1) * batch_size]
        if progress_callback:
            progress_callback(
                b_idx / max(1, total_batches),
                f"주가 데이터 다운로드 중... ({min(len(full_tickers), (b_idx + 1) * batch_size)}/{len(full_tickers)} 종목)",
            )
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if len(batch) == 1:
                t = batch[0]
                cleaned = _clean_ohlcv(raw)
                if len(cleaned) >= 60:
                    cleaned.to_csv(_cache_path(t, period))
                    results[t] = cleaned
            else:
                for t in batch:
                    if t in raw.columns.get_level_values(0):
                        sub = raw[t].copy()
                        cleaned = _clean_ohlcv(sub)
                        if len(cleaned) >= 60:
                            cleaned.to_csv(_cache_path(t, period))
                            results[t] = cleaned
        except Exception:
            for t in batch:
                df = fetch_single_stock(t, period=period, use_cache=use_cache)
                if len(df) >= 60:
                    results[t] = df
        time.sleep(0.1)

    if progress_callback:
        progress_callback(1.0, f"총 {len(results)}개 종목 데이터 준비 완료")

    return results


def fetch_market_context(
    period: str = "3y",
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Fetch SPY (S&P 500 ETF) and ^VIX (Volatility Index) to determine daily
    Market Regime (SPY > SMA50 / SMA200, VIX level) and benchmark returns for Relative Strength.
    """
    spy_df = fetch_single_stock("SPY", period=period, use_cache=use_cache)
    vix_df = fetch_single_stock("^VIX", period=period, use_cache=use_cache)

    if spy_df.empty:
        return pd.DataFrame(), {
            "available": False,
            "spy_close": 0.0,
            "spy_above_sma50": True,
            "spy_above_sma200": True,
            "vix_close": 0.0,
            "regime_label": "시장 데이터 없음",
        }

    ctx = pd.DataFrame(index=spy_df.index)
    ctx["SPY_CLOSE"] = spy_df["Close"]
    ctx["SPY_SMA50"] = spy_df["Close"].rolling(50, min_periods=20).mean()
    ctx["SPY_SMA200"] = spy_df["Close"].rolling(200, min_periods=50).mean()
    ctx["SPY_RET20"] = spy_df["Close"].pct_change(20) * 100.0
    ctx["SPY_RET60"] = spy_df["Close"].pct_change(60) * 100.0
    ctx["SPY_ABOVE_50"] = (ctx["SPY_CLOSE"] > ctx["SPY_SMA50"]).fillna(True)
    ctx["SPY_ABOVE_200"] = (ctx["SPY_CLOSE"] > ctx["SPY_SMA200"]).fillna(True)

    if not vix_df.empty:
        ctx["VIX_CLOSE"] = vix_df["Close"].reindex(ctx.index).ffill().fillna(18.0)
    else:
        ctx["VIX_CLOSE"] = 18.0

    last = ctx.iloc[-1]
    spy_c = float(last["SPY_CLOSE"])
    above50 = bool(last["SPY_ABOVE_50"])
    above200 = bool(last["SPY_ABOVE_200"])
    vix_c = float(last["VIX_CLOSE"])

    if above50 and above200 and vix_c <= 22.0:
        regime_label = "상승 안정장 (매수 유리)"
    elif above200 and vix_c <= 28.0:
        regime_label = "중립/혼조장 (선별 매수)"
    else:
        regime_label = "하락/변동성 경계장 (보수적 접근)"

    summary = {
        "available": True,
        "spy_close": round(spy_c, 2),
        "spy_above_sma50": above50,
        "spy_above_sma200": above200,
        "vix_close": round(vix_c, 2),
        "regime_label": regime_label,
    }
    return ctx, summary


def fetch_upcoming_earnings_days(
    tickers: List[str],
    max_age_hours: float = 24.0,
) -> Dict[str, Optional[int]]:
    """
    Fetch days until the next upcoming earnings announcement for each ticker.
    Uses a 24h local JSON cache so repeated UI interactions remain instantaneous.
    """
    import json

    ensure_dirs()
    cache_file = CACHE_DIR / "earnings_cache.json"
    cache_data: Dict[str, Dict[str, object]] = {}

    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache_data = {}

    now_dt = datetime.now()
    today_date = now_dt.date()
    results: Dict[str, Optional[int]] = {}
    updated = False

    for raw_t in tickers:
        t = normalize_ticker(raw_t)
        entry = cache_data.get(t)
        if entry:
            try:
                cached_at = datetime.fromisoformat(str(entry["cached_at"]))
                if (now_dt - cached_at) < timedelta(hours=max_age_hours):
                    ed_str = entry.get("earnings_date")
                    if ed_str:
                        ed_date = datetime.strptime(str(ed_str), "%Y-%m-%d").date()
                        diff = (ed_date - today_date).days
                        results[t] = diff if diff >= 0 else None
                    else:
                        results[t] = None
                    continue
            except Exception:
                pass

        # Fetch from yfinance
        next_ed_str: Optional[str] = None
        days_until: Optional[int] = None
        try:
            cal = yf.Ticker(t).calendar
            ed_val = None
            if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
                ed_val = cal["Earnings Date"][0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
                ed_val = cal.loc["Earnings Date"].iloc[0]

            if ed_val is not None:
                ed_dt = pd.to_datetime(ed_val).date()
                diff = (ed_dt - today_date).days
                if diff >= 0:
                    days_until = int(diff)
                    next_ed_str = ed_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

        results[t] = days_until
        cache_data[t] = {
            "cached_at": now_dt.isoformat(),
            "earnings_date": next_ed_str,
        }
        updated = True

    if updated:
        try:
            cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return results


