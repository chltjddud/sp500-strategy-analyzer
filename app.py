"""
S&P 500 Multi-Strategy Combination Backtester & Probability Analyzer Dashboard
with Strategy-Specific Take-Profit / Stop-Loss Lines and Light/Dark Themes.
Run with: streamlit run app.py
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import src.data_loader as _dl_mod
import src.strategies as _st_mod
import src.backtester as _bt_mod
import src.feedback_db as _fb_mod
import src.ai_analyst as _ai_mod

importlib.reload(_dl_mod)
importlib.reload(_st_mod)
importlib.reload(_bt_mod)
importlib.reload(_fb_mod)
importlib.reload(_ai_mod)

from src.ai_analyst import generate_ai_briefing, load_gemini_api_key

from src.backtester import (
    analyze_single_stock,
    get_trade_history_for_combo,
    scan_market,
)
from src.data_loader import (
    TOP_50_TICKERS,
    fetch_market_context,
    fetch_multiple_stocks,
    fetch_single_stock,
    fetch_upcoming_earnings_days,
    get_sp500_constituents,
    normalize_ticker,
)
from src.feedback_db import (
    compute_regime_comparison,
    fetch_baseline_count,
    fetch_signal_logs,
    fetch_strategy_feedback,
    get_supabase_client,
    grade_pending_signals,
    load_saved_credentials,
    save_signals_to_db,
)
from src.strategies import (
    STRATEGY_METADATA,
    format_combo_name,
    parse_combo_id,
)

st.set_page_config(
    page_title="S&P 500 매매기법 조합 확률 분석기",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME DEFINITIONS (라이트 테마 / 다크 테마)
# ---------------------------------------------------------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "라이트 테마 (밝은 화면)": {
        "mode": "light",
        "app_bg": "#f8fafc",
        "sidebar_bg": "#ffffff",
        "card_bg": "#ffffff",
        "subcard_bg": "#f1f5f9",
        "border": "#cbd5e1",
        "text_main": "#0f172a",
        "text_sub": "#475569",
        "text_muted": "#64748b",
        "accent": "#0284c7",
        "accent_soft": "#e0f2fe",
        "tp_color": "#15803d",
        "tp_bg": "#dcfce7",
        "sl_color": "#b91c1c",
        "sl_bg": "#fee2e2",
        "plotly_template": "plotly_white",
        "chart_bg": "#ffffff",
        "grid_color": "#e2e8f0",
    },
    "다크 테마 (어두운 화면)": {
        "mode": "dark",
        "app_bg": "#0b1120",
        "sidebar_bg": "#111827",
        "card_bg": "#1e293b",
        "subcard_bg": "#0f172a",
        "border": "#334155",
        "text_main": "#f8fafc",
        "text_sub": "#cbd5e1",
        "text_muted": "#94a3b8",
        "accent": "#38bdf8",
        "accent_soft": "#0c4a6e",
        "tp_color": "#4ade80",
        "tp_bg": "rgba(34, 197, 94, 0.16)",
        "sl_color": "#f87171",
        "sl_bg": "rgba(239, 68, 68, 0.16)",
        "plotly_template": "plotly_dark",
        "chart_bg": "#0f172a",
        "grid_color": "#1e293b",
    },
}


def apply_theme_css(theme: Dict[str, str]) -> None:
    """Inject responsive, collision-free CSS tailored to Light or Dark theme."""
    is_dark = theme["mode"] == "dark"
    widget_bg = "#1e293b" if is_dark else "#ffffff"
    widget_text = "#f8fafc" if is_dark else "#0f172a"
    tab_bg = "#1e293b" if is_dark else "#e2e8f0"
    tab_active_bg = "#0284c7" if is_dark else "#ffffff"
    tab_active_text = "#ffffff" if is_dark else "#0284c7"
    btn_bg = "#1e293b" if is_dark else "#ffffff"
    btn_hover = "#334155" if is_dark else "#f1f5f9"

    st.markdown(
        f"""
        <style>
        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {theme['app_bg']} !important;
            color: {theme['text_main']} !important;
            font-family: "Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif !important;
        }}
        [data-testid="stHeader"] {{
            background-color: {theme['app_bg']} !important;
        }}
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button,
        [data-testid="stExpandSidebarButton"] button {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_main']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background-color: {theme['sidebar_bg']} !important;
            color: {theme['text_main']} !important;
            border-right: 1px solid {theme['border']};
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] p {{
            color: {theme['text_main']} !important;
        }}
        [data-testid="stCaptionContainer"] p {{
            color: {theme['text_sub']} !important;
            font-size: 0.92rem !important;
        }}

        /* Inputs, Selectboxes, Expanders & Buttons */
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {{
            background-color: {widget_bg} !important;
            color: {widget_text} !important;
            border-color: {theme['border']} !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            color: {widget_text} !important;
        }}
        .stButton > button, .stDownloadButton > button {{
            background-color: {btn_bg} !important;
            color: {theme['text_main']} !important;
            border: 1px solid {theme['border']} !important;
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {btn_hover} !important;
            border-color: {theme['accent']} !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {theme['accent']} !important;
            color: #ffffff !important;
            border: 1px solid {theme['accent']} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            opacity: 0.92 !important;
            border-color: {theme['accent']} !important;
        }}
        [data-testid="stExpander"] {{
            background-color: {theme['card_bg']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 10px !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {tab_bg};
            padding: 6px;
            border-radius: 12px;
            border: 1px solid {theme['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            color: {theme['text_sub']} !important;
            white-space: nowrap;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {tab_active_bg} !important;
            color: {tab_active_text} !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }}

        /* Top setting badges */
        .pill-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 6px 0 16px 0;
        }}
        .info-pill {{
            display: inline-block;
            background-color: {theme['card_bg']};
            border: 1px solid {theme['border']};
            color: {theme['text_sub']};
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 600;
        }}

        /* Clean KPI Cards */
        .kpi-card {{
            background-color: {theme['card_bg']};
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.83rem;
            font-weight: 600;
            color: {theme['text_sub']};
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 800;
            color: {theme['text_main']};
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.82rem;
            color: {theme['text_muted']};
            margin-top: 5px;
            line-height: 1.35;
        }}

        /* Top Signal Recommendation Cards */
        .signal-card {{
            background-color: {theme['card_bg']};
            border: 1px solid {theme['border']};
            border-top: 4px solid {theme['accent']};
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        }}
        .signal-badge {{
            display: inline-block;
            background-color: {theme['accent_soft']};
            color: {theme['accent']};
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .signal-title {{
            font-size: 1.15rem;
            font-weight: 800;
            color: {theme['text_main']};
            margin: 2px 0 6px 0;
            word-break: keep-all;
        }}
        .prob-highlight {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {theme['tp_color']};
            margin-bottom: 4px;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }}
        .grid-item {{
            background-color: {theme['subcard_bg']};
            border-radius: 8px;
            padding: 8px 10px;
        }}
        .grid-item-tp {{
            background-color: {theme['tp_bg']};
            border-radius: 8px;
            padding: 8px 10px;
        }}
        .grid-item-sl {{
            background-color: {theme['sl_bg']};
            border-radius: 8px;
            padding: 8px 10px;
        }}
        .grid-label {{
            font-size: 0.76rem;
            font-weight: 600;
            color: {theme['text_sub']};
            margin-bottom: 2px;
        }}
        .grid-val {{
            font-size: 0.98rem;
            font-weight: 800;
            color: {theme['text_main']};
        }}
        .grid-val-tp {{
            font-size: 0.98rem;
            font-weight: 800;
            color: {theme['tp_color']};
        }}
        .grid-val-sl {{
            font-size: 0.98rem;
            font-weight: 800;
            color: {theme['sl_color']};
        }}
        .rule-box {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed {theme['border']};
            font-size: 0.79rem;
            color: {theme['text_sub']};
            line-height: 1.45;
            word-break: keep-all;
        }}

        /* Strategy Action Plan Banner */
        .plan-banner {{
            background-color: {theme['card_bg']};
            border: 1px solid {theme['border']};
            border-left: 5px solid {theme['accent']};
            border-radius: 12px;
            padding: 14px 18px;
            margin: 10px 0 16px 0;
            color: {theme['text_main']};
            line-height: 1.6;
        }}

        /* Landing Page Hero & Universe Selection Cards */
        .landing-hero {{
            background: linear-gradient(135deg, {theme['card_bg']} 0%, {theme['subcard_bg']} 100%);
            border: 1px solid {theme['border']};
            border-radius: 18px;
            padding: 36px 32px;
            margin: 12px 0 24px 0;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }}
        .landing-eyebrow {{
            display: inline-block;
            background-color: {theme['accent_soft']};
            color: {theme['accent']};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            padding: 5px 14px;
            border-radius: 999px;
            margin-bottom: 14px;
        }}
        .landing-title {{
            font-size: 2.05rem;
            font-weight: 800;
            color: {theme['text_main']};
            margin: 0 0 12px 0;
            line-height: 1.25;
            word-break: keep-all;
        }}
        .landing-sub {{
            font-size: 1.02rem;
            color: {theme['text_sub']};
            max-width: 720px;
            margin: 0 auto;
            line-height: 1.65;
            word-break: keep-all;
        }}
        .mode-card {{
            background-color: {theme['card_bg']};
            border: 1px solid {theme['border']};
            border-top: 4px solid {theme['accent']};
            border-radius: 16px;
            padding: 24px 22px 18px 22px;
            margin-bottom: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
            min-height: 185px;
        }}
        .mode-badge {{
            display: inline-block;
            background-color: {theme['accent_soft']};
            color: {theme['accent']};
            font-size: 0.78rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 999px;
            margin-bottom: 10px;
        }}
        .mode-title {{
            font-size: 1.32rem;
            font-weight: 800;
            color: {theme['text_main']};
            margin-bottom: 8px;
        }}
        .mode-desc {{
            font-size: 0.94rem;
            color: {theme['text_sub']};
            line-height: 1.55;
            word-break: keep-all;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def cached_constituents() -> pd.DataFrame:
    return get_sp500_constituents()


@st.cache_data(ttl=1800, show_spinner=False)
def cached_market_context(period: str, refresh_nonce: int = 0) -> Tuple[pd.DataFrame, Dict]:
    return fetch_market_context(period=period, use_cache=(refresh_nonce == 0))


@st.cache_data(ttl=1800, show_spinner=False)
def cached_stock_data(tickers: Tuple[str, ...], period: str, refresh_nonce: int = 0) -> Dict[str, pd.DataFrame]:
    return fetch_multiple_stocks(
        tickers=list(tickers),
        period=period,
        use_cache=(refresh_nonce == 0),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_feedback_bonus(sb_url: str, sb_key: str) -> Dict[str, float]:
    if not sb_url or not sb_key:
        return {}
    try:
        _sb_tmp = get_supabase_client(sb_url, sb_key)
        _fb_tmp_df = fetch_strategy_feedback(_sb_tmp)
        if not _fb_tmp_df.empty:
            return {
                str(r["combo_id"]): float(r["feedback_bonus"])
                for _, r in _fb_tmp_df.iterrows()
            }
    except Exception:
        pass
    return {}


@st.cache_data(ttl=1800, show_spinner=False)
def cached_scan_and_filter(
    tickers: Tuple[str, ...],
    period: str,
    hold_days: int,
    target_pct: float,
    stop_pct: float,
    signal_window: int,
    min_trades: int,
    recent_signal_days: int,
    max_comb: int,
    use_dynamic_tpsl: bool,
    rr_multiplier: float,
    use_regime_filter: bool,
    vix_max: float,
    use_rs_filter: bool,
    exclude_earnings: bool,
    feedback_items: Tuple[Tuple[str, float], ...],
    use_sma200_filter: bool = False,
    use_breakeven_stop: bool = True,
    require_multi_in_high_vix: bool = True,
    use_sector_bonus: bool = True,
    use_walk_forward: bool = True,
    next_open_entry: bool = False,
    slippage_pct: float = 0.0,
    use_partial_tp: bool = False,
    max_per_sector: int = 0,
    fast_active_only: bool = False,
    use_strategy_hold_days: bool = False,
    auto_tune_rr: bool = False,
    use_dynamic_sizing: bool = True,
    refresh_nonce: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    constituents_df = cached_constituents()
    market_ctx_df, _ = cached_market_context(period=period, refresh_nonce=refresh_nonce)
    stock_data = cached_stock_data(tickers=tickers, period=period, refresh_nonce=refresh_nonce)
    feedback_bonus_map = dict(feedback_items)

    active_signals_df, best_per_stock_df, leaderboard_df = scan_market(
        stock_data=stock_data,
        constituents_df=constituents_df,
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
        min_rs_pct=0.0,
        feedback_bonus_map=feedback_bonus_map,
        use_sma200_filter=use_sma200_filter,
        use_breakeven_stop=use_breakeven_stop,
        require_multi_in_high_vix=require_multi_in_high_vix,
        use_sector_bonus=use_sector_bonus,
        use_walk_forward=use_walk_forward,
        next_open_entry=next_open_entry,
        slippage_pct=slippage_pct,
        use_partial_tp=use_partial_tp,
        max_per_sector=max_per_sector,
        fast_active_only=fast_active_only,
        use_strategy_hold_days=use_strategy_hold_days,
        auto_tune_rr=auto_tune_rr,
        use_dynamic_sizing=use_dynamic_sizing,
    )

    excluded_earnings_tickers: List[str] = []
    if not active_signals_df.empty:
        cand_tickers = active_signals_df["Ticker"].unique().tolist()
        earnings_map = fetch_upcoming_earnings_days(cand_tickers)
        active_signals_df["실적발표(D-day)"] = active_signals_df["Ticker"].map(
            lambda t: f"D-{earnings_map[t]}" if earnings_map.get(t) is not None else "-"
        )
        if exclude_earnings:
            safe_mask = active_signals_df["Ticker"].map(
                lambda t: earnings_map.get(t) is None or int(earnings_map[t]) > (hold_days + 3)
            )
            excluded_earnings_tickers = active_signals_df.loc[~safe_mask, "Ticker"].tolist()
            active_signals_df = active_signals_df[safe_mask].reset_index(drop=True)

    return active_signals_df, best_per_stock_df, leaderboard_df, excluded_earnings_tickers


def _compute_non_overlapping_yshifts(
    prices: List[float], y_min: float, y_max: float, min_gap_ratio: float = 0.085
) -> List[int]:
    """
    Prevent horizontal-line price labels from overlapping vertically when
    entry, TP, and SL prices are close to each other.
    """
    span = max(1e-6, y_max - y_min)
    indexed = sorted(enumerate(prices), key=lambda x: x[1])
    norm_pos = [((p - y_min) / span) for _, p in indexed]

    shifts_sorted = [0, 0, 0]
    if len(norm_pos) == 3:
        gap_low_mid = norm_pos[1] - norm_pos[0]
        gap_mid_high = norm_pos[2] - norm_pos[1]
        if gap_low_mid < min_gap_ratio:
            shifts_sorted[0] = -14
        if gap_mid_high < min_gap_ratio:
            shifts_sorted[2] = 14
        if gap_low_mid < min_gap_ratio and gap_mid_high < min_gap_ratio:
            shifts_sorted[0] = -18
            shifts_sorted[1] = 0
            shifts_sorted[2] = 18

    out = [0] * len(prices)
    for (orig_idx, _), shift_val in zip(indexed, shifts_sorted):
        out[orig_idx] = shift_val
    return out


def build_candlestick_chart(
    df_ind: pd.DataFrame,
    trades_df: pd.DataFrame,
    ticker: str,
    combo_name: str,
    entry_price: float,
    tp_price: float,
    tp_pct: float,
    sl_price: float,
    sl_pct: float,
    theme: Dict[str, str],
    lookback_bars: int = 250,
) -> go.Figure:
    """
    Create interactive Plotly Candlestick + BB + Volume + RSI + MACD chart
    with collision-free right-margin Take-Profit / Entry / Stop-Loss price badges.
    """
    plot_df = df_ind.iloc[-lookback_bars:].copy() if len(df_ind) > lookback_bars else df_ind.copy()
    min_date = plot_df.index[0]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        row_heights=[0.52, 0.14, 0.17, 0.17],
        subplot_titles=(
            "1. 주가 캔들 · 이동평균선 · 볼린저밴드 · 익절/손절선",
            "2. 거래량 (Volume & 20일 평균)",
            "3. RSI (14)",
            "4. MACD (12, 26, 9)",
        ),
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="주가(OHLC)",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    # Moving Averages & Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["SMA20"],
            name="SMA 20",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["SMA50"],
            name="SMA 50",
            line=dict(color="#2563eb", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["BB_UPPER"],
            name="BB 상단",
            line=dict(color="rgba(100, 116, 139, 0.50)", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["BB_LOWER"],
            name="BB 하단",
            line=dict(color="rgba(100, 116, 139, 0.50)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.08)",
        ),
        row=1,
        col=1,
    )

    data_low = float(plot_df["Low"].min())
    data_high = float(plot_df["High"].max())
    y_min = min(data_low, sl_price) * 0.985
    y_max = max(data_high, tp_price) * 1.015

    # Horizontal Lines for TP, Entry, SL
    fig.add_hline(y=tp_price, line_dash="dash", line_width=2.2, line_color="#16a34a", row=1, col=1)
    fig.add_hline(y=entry_price, line_dash="dot", line_width=1.6, line_color="#0284c7", row=1, col=1)
    fig.add_hline(y=sl_price, line_dash="dash", line_width=2.2, line_color="#dc2626", row=1, col=1)

    shifts = _compute_non_overlapping_yshifts([tp_price, entry_price, sl_price], y_min, y_max)
    right_labels = [
        (tp_price, f"익절선 ${tp_price:.2f} (+{tp_pct:.1f}%)", "#ffffff", "#16a34a", shifts[0]),
        (entry_price, f"진입가 ${entry_price:.2f}", "#ffffff", "#0284c7", shifts[1]),
        (sl_price, f"손절선 ${sl_price:.2f} ({sl_pct:.1f}%)", "#ffffff", "#dc2626", shifts[2]),
    ]
    for y_val, txt, font_c, bg_c, y_sh in right_labels:
        fig.add_annotation(
            xref="paper",
            x=1.01,
            yref="y",
            y=y_val,
            text=txt,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            yshift=y_sh,
            font=dict(color=font_c, size=11.5, family="Pretendard, Malgun Gothic, sans-serif"),
            bgcolor=bg_c,
            bordercolor=bg_c,
            borderpad=4,
        )

    # Plot trade entry markers
    if not trades_df.empty:
        vis_trades = trades_df[trades_df["시그널 일자"] >= min_date].copy()
        if not vis_trades.empty:
            wins = vis_trades[vis_trades["목표 도달"] == True]
            losses = vis_trades[vis_trades["목표 도달"] == False]
            active = vis_trades[vis_trades["목표 도달"].isna()]

            if not wins.empty:
                win_dates = wins["시그널 일자"]
                win_lows = plot_df.loc[win_dates, "Low"] * 0.985
                fig.add_trace(
                    go.Scatter(
                        x=win_dates,
                        y=win_lows,
                        mode="markers",
                        name="성공 타점(익절 도달)",
                        marker=dict(
                            symbol="triangle-up",
                            size=13,
                            color="#16a34a",
                            line=dict(color="#ffffff", width=1.2),
                        ),
                        customdata=np.stack(
                            [
                                wins["진입 종가($)"],
                                wins["익절 목표가($)"],
                                wins["익절폭(+%)"],
                                wins["손절 기준가($)"],
                                wins["손절폭(-%)"],
                                wins["기간내 최고(%)"],
                                wins["실현 수익률(%)"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>[익절 성공 타점]</b> %{x|%Y-%m-%d}<br>"
                            "• 진입 종가: $%{customdata[0]:.2f}<br>"
                            "• 기법 익절선: $%{customdata[1]:.2f} (+%{customdata[2]:.2f}%)<br>"
                            "• 기법 손절선: $%{customdata[3]:.2f} (%{customdata[4]:.2f}%)<br>"
                            "• 기간내 최고: +%{customdata[5]:.2f}% | 실현 수익률: +%{customdata[6]:.2f}%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            if not losses.empty:
                loss_dates = losses["시그널 일자"]
                loss_lows = plot_df.loc[loss_dates, "Low"] * 0.985
                fig.add_trace(
                    go.Scatter(
                        x=loss_dates,
                        y=loss_lows,
                        mode="markers",
                        name="실패 타점(미달/손절)",
                        marker=dict(
                            symbol="x",
                            size=10,
                            color="#dc2626",
                            line=dict(color="#ffffff", width=1),
                        ),
                        customdata=np.stack(
                            [
                                losses["진입 종가($)"],
                                losses["익절 목표가($)"],
                                losses["익절폭(+%)"],
                                losses["손절 기준가($)"],
                                losses["손절폭(-%)"],
                                losses["기간내 최고(%)"],
                                losses["실현 수익률(%)"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>[익절 미달 / 손절 타점]</b> %{x|%Y-%m-%d}<br>"
                            "• 진입 종가: $%{customdata[0]:.2f}<br>"
                            "• 기법 익절선: $%{customdata[1]:.2f} (+%{customdata[2]:.2f}%)<br>"
                            "• 기법 손절선: $%{customdata[3]:.2f} (%{customdata[4]:.2f}%)<br>"
                            "• 기간내 최고: %{customdata[5]:.2f}% | 실현 수익률: %{customdata[6]:.2f}%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            if not active.empty:
                act_dates = active["시그널 일자"]
                act_lows = plot_df.loc[act_dates, "Low"] * 0.980
                fig.add_trace(
                    go.Scatter(
                        x=act_dates,
                        y=act_lows,
                        mode="markers",
                        name="최근 시그널(진행중)",
                        marker=dict(
                            symbol="star",
                            size=16,
                            color="#eab308",
                            line=dict(color="#1e293b", width=1),
                        ),
                        customdata=np.stack(
                            [
                                active["진입 종가($)"],
                                active["익절 목표가($)"],
                                active["익절폭(+%)"],
                                active["손절 기준가($)"],
                                active["손절폭(-%)"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "<b>[최근 매수 시그널]</b> %{x|%Y-%m-%d}<br>"
                            "• 진입 종가: $%{customdata[0]:.2f}<br>"
                            "• 목표 익절선: $%{customdata[1]:.2f} (+%{customdata[2]:.2f}%)<br>"
                            "• 기준 손절선: $%{customdata[3]:.2f} (%{customdata[4]:.2f}%)<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

    # 2. Volume
    vol_colors = np.where(plot_df["Close"] >= plot_df["Open"], "#16a34a", "#dc2626")
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["Volume"],
            name="거래량",
            marker_color=vol_colors,
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["VOL_SMA20"],
            name="거래량 20일선",
            line=dict(color="#f59e0b", width=1.3),
        ),
        row=2,
        col=1,
    )

    # 3. RSI(14)
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RSI14"],
            name="RSI(14)",
            line=dict(color="#9333ea", width=1.6),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220,38,38,0.45)", row=3, col=1)
    fig.add_hline(y=40, line_dash="dash", line_color="rgba(22,163,74,0.45)", row=3, col=1)

    # 4. MACD
    macd_colors = np.where(plot_df["MACD_HIST"] >= 0, "#16a34a", "#dc2626")
    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["MACD_HIST"],
            name="MACD Hist",
            marker_color=macd_colors,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MACD_LINE"],
            name="MACD",
            line=dict(color="#0284c7", width=1.4),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MACD_SIGNAL"],
            name="Signal",
            line=dict(color="#ea580c", width=1.4),
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    fig.update_yaxes(gridcolor=theme["grid_color"])
    fig.update_xaxes(gridcolor=theme["grid_color"])

    for ann in fig.layout.annotations:
        if ann.text and ann.text[0].isdigit():
            ann.font.size = 13
            ann.font.color = theme["text_main"]

    fig.update_layout(
        height=960,
        template=theme["plotly_template"],
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        font=dict(
            color=theme["text_main"],
            family="Pretendard, Malgun Gothic, Apple SD Gothic Neo, sans-serif",
            size=12,
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=55, r=215, t=95, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0.0,
            font=dict(size=11.5),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def render_kpi_row(items: List[Dict[str, str]]) -> None:
    """Render a row of clean, non-truncating KPI cards."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        val_color = item.get("color", "")
        style_attr = f"style='color:{val_color};'" if val_color else ""
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{item['label']}</div>
                    <div class="kpi-value" {style_attr}>{item['value']}</div>
                    <div class="kpi-sub">{item.get('sub', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_supabase_setup_guide() -> None:
    """Display safe notice when server-side database credentials are not configured."""
    st.info(
        "피드백 데이터베이스가 아직 서버 환경변수(.env 또는 Streamlit Secrets)에 설정되지 않았습니다. "
        "관리자 환경설정 후 자동으로 활성화됩니다."
    )


def render_feedback_results(sb_client, theme: Dict[str, str]) -> None:
    """Render KPI metrics and 3 feedback/grading sub-tabs from Supabase."""
    try:
        logs_df = fetch_signal_logs(sb_client, limit=200)
        fb_df = fetch_strategy_feedback(sb_client)
        baseline_cnt = fetch_baseline_count(sb_client)

        total_logs = len(logs_df)
        pending_cnt = int((logs_df["status"] == "PENDING").sum()) if not logs_df.empty else 0
        graded_df = logs_df[logs_df["status"] != "PENDING"] if not logs_df.empty else pd.DataFrame()
        graded_cnt = len(graded_df)
        live_win_rate = (
            round(float(graded_df["status"].isin(["WIN", "EXPIRED_WIN"]).mean() * 100.0), 1)
            if graded_cnt > 0
            else 0.0
        )

        render_kpi_row(
            [
                {"label": "Colab 초반 학습 데이터", "value": f"{baseline_cnt:,}건", "sub": "strategy_baseline 테이블"},
                {"label": "누적 시그널 기록", "value": f"{total_logs}건", "sub": f"채점 대기중(PENDING): {pending_cnt}건"},
                {"label": "자동 채점 완료", "value": f"{graded_cnt}건", "sub": "실제 주가 기반 승패 판정 완료", "color": theme["accent"]},
                {"label": "실전 누적 승률", "value": f"{live_win_rate:.1f}%" if graded_cnt > 0 else "-", "sub": "채점 완료 건 기준 실제 승률", "color": theme["tp_color"]},
            ]
        )

        st.write("")
        fb_tab1, fb_tab2, fb_tab3 = st.tabs(
            [
                "4대 트랙(50/500 × 국면ON/OFF) 실전 성과 비교",
                "기법 조합별 자가 피드백 가중치",
                "가상 매매 기록 및 채점 상세 로그",
            ]
        )
        with fb_tab1:
            comp_df = compute_regime_comparison(logs_df)
            if comp_df.empty:
                st.info(
                    "아직 채점이 완료된 4대 트랙 비교 데이터가 없습니다. "
                    "'4대 트랙 동시 기록' 버튼으로 시그널을 기록하거나 장 마감 후 매일 자동 채점이 진행되면 "
                    "이곳에서 4개 트랙의 실제 적중률과 손익비를 비교할 수 있습니다."
                )
            else:
                st.dataframe(
                    comp_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "트랙": st.column_config.TextColumn("트랙", width="small"),
                        "종목 범위": st.column_config.TextColumn("종목 범위", width="medium"),
                        "시장국면 필터": st.column_config.TextColumn("시장국면 필터", width="medium"),
                        "채점 건수": st.column_config.NumberColumn("채점 수"),
                        "익절 도달(WIN)": st.column_config.NumberColumn("익절(WIN)"),
                        "손절 이탈(LOSS)": st.column_config.NumberColumn("손절(LOSS)"),
                        "만기 청산": st.column_config.NumberColumn("만기 청산"),
                        "익절 적중률(%)": st.column_config.ProgressColumn(
                            "익절 적중률", format="%.1f%%", min_value=0, max_value=100
                        ),
                        "전체 승률(%)": st.column_config.ProgressColumn(
                            "전체 승률", format="%.1f%%", min_value=0, max_value=100
                        ),
                        "평균 수익률(%)": st.column_config.NumberColumn("평균 수익률", format="%+.2f%%"),
                        "평균 보유일(일)": st.column_config.NumberColumn("평균 보유일", format="%.1f일"),
                    },
                )

        with fb_tab2:
            if fb_df.empty:
                st.info("아직 채점이 완료된 기록이 없습니다. 시그널을 저장하고 기간이 지난 뒤 '자동 채점'을 실행하면 기법별 피드백 가중치가 자동 계산됩니다.")
            else:
                st.dataframe(
                    fb_df[
                        [
                            "combo_name",
                            "total_graded",
                            "wins",
                            "losses",
                            "live_win_rate",
                            "avg_live_return",
                            "feedback_bonus",
                            "updated_at",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "combo_name": st.column_config.TextColumn("매매기법 조합"),
                        "total_graded": st.column_config.NumberColumn("채점 수"),
                        "wins": st.column_config.NumberColumn("성공"),
                        "losses": st.column_config.NumberColumn("실패"),
                        "live_win_rate": st.column_config.NumberColumn("실전 승률", format="%.1f%%"),
                        "avg_live_return": st.column_config.NumberColumn("평균 실전수익", format="%+.2f%%"),
                        "feedback_bonus": st.column_config.NumberColumn("피드백 가중치", format="%+.2f"),
                    },
                )

        with fb_tab3:
            if logs_df.empty:
                st.info("저장된 시그널 기록이 없습니다. 상단의 가상 매수 기록 버튼을 눌러보세요.")
            else:
                status_label_map = {
                    "WIN": "적중 (O) - 익절 도달",
                    "EXPIRED_WIN": "적중 (O) - 만기 수익",
                    "LOSS": "실패 (X) - 손절 이탈",
                    "EXPIRED_LOSS": "실패 (X) - 만기 손실",
                    "PENDING": "진행중 (대기)",
                }
                logs_view = logs_df.copy()
                logs_view["예측 판정"] = logs_view["status"].map(
                    lambda s: status_label_map.get(str(s), str(s))
                )
                if "regime_filter_on" in logs_view.columns:
                    logs_view["필터 버전"] = logs_view["regime_filter_on"].map(
                        lambda v: "국면 ON" if bool(v) else "국면 OFF"
                    )
                else:
                    logs_view["필터 버전"] = "국면 ON"

                def _resolve_scope_label(r: pd.Series) -> str:
                    if str(r.get("ticker", "")) not in TOP_50_TICKERS:
                        return "전체 종목"
                    u_val = str(r.get("universe_scope", "")).upper()
                    if u_val == "ALL":
                        return "전체 종목"
                    return "상위 50개"

                logs_view["종목 범위"] = logs_view.apply(_resolve_scope_label, axis=1)

                if "exit_date" not in logs_view.columns:
                    logs_view["exit_date"] = "-"
                if "days_held" not in logs_view.columns:
                    logs_view["days_held"] = None

                f_r1, f_r2, f_r3 = st.columns(3)
                with f_r1:
                    filter_choice = st.radio(
                        "예측 결과 필터",
                        options=["전체", "적중 (O)만 보기", "실패 (X)만 보기", "진행중 (대기)만 보기"],
                        horizontal=True,
                        key="fb_res_filter",
                    )
                with f_r2:
                    scope_choice = st.radio(
                        "종목 범위 필터",
                        options=["전체 범위", "상위 50개만", "전체 종목만"],
                        horizontal=True,
                        key="fb_scope_filter",
                    )
                with f_r3:
                    ver_choice = st.radio(
                        "시장국면 필터 버전",
                        options=["전체 버전", "국면 ON만", "국면 OFF만"],
                        horizontal=True,
                        key="fb_ver_filter",
                    )

                if filter_choice.startswith("적중"):
                    logs_view = logs_view[logs_view["status"].isin(["WIN", "EXPIRED_WIN"])]
                elif filter_choice.startswith("실패"):
                    logs_view = logs_view[logs_view["status"].isin(["LOSS", "EXPIRED_LOSS"])]
                elif filter_choice.startswith("진행중"):
                    logs_view = logs_view[logs_view["status"] == "PENDING"]

                if scope_choice == "상위 50개만":
                    logs_view = logs_view[logs_view["종목 범위"] == "상위 50개"]
                elif scope_choice == "전체 종목만":
                    logs_view = logs_view[logs_view["종목 범위"] == "전체 종목"]

                if ver_choice == "국면 ON만":
                    logs_view = logs_view[logs_view["필터 버전"] == "국면 ON"]
                elif ver_choice == "국면 OFF만":
                    logs_view = logs_view[logs_view["필터 버전"] == "국면 OFF"]

                st.dataframe(
                    logs_view[
                        [
                            "signal_date",
                            "ticker",
                            "종목 범위",
                            "필터 버전",
                            "combo_name",
                            "예측 판정",
                            "entry_price",
                            "tp_price",
                            "sl_price",
                            "predicted_hit_rate",
                            "exit_price",
                            "exit_date",
                            "days_held",
                            "actual_return",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "signal_date": st.column_config.TextColumn("매수(시그널)일"),
                        "ticker": st.column_config.TextColumn("티커"),
                        "종목 범위": st.column_config.TextColumn("종목 범위"),
                        "필터 버전": st.column_config.TextColumn("국면 필터"),
                        "combo_name": st.column_config.TextColumn("매매기법 조합"),
                        "예측 판정": st.column_config.TextColumn("예측 판정"),
                        "entry_price": st.column_config.NumberColumn("매수 진입가", format="$%.2f"),
                        "tp_price": st.column_config.NumberColumn("익절선", format="$%.2f"),
                        "sl_price": st.column_config.NumberColumn("손절선", format="$%.2f"),
                        "predicted_hit_rate": st.column_config.NumberColumn("예측 확률", format="%.1f%%"),
                        "exit_price": st.column_config.NumberColumn("매도(청산)가", format="$%.2f"),
                        "exit_date": st.column_config.TextColumn("매도일"),
                        "days_held": st.column_config.NumberColumn("보유일"),
                        "actual_return": st.column_config.NumberColumn("실제 수익률", format="%+.2f%%"),
                    },
                )
    except Exception:
        st.warning("피드백 데이터를 불러오는 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")


def render_landing_page(theme: Dict[str, str], total_sp500: int) -> None:
    """Render a minimal, polished landing page with 3 buttons: Top 50, All 500, and Auto-Grading."""
    st.markdown(
        f"""
        <style>
        .st-key-landing_btn_50 button,
        .st-key-landing_btn_500 button,
        .st-key-landing_btn_feedback button {{
            min-height: 122px !important;
            border-radius: 18px !important;
            padding: 22px 20px !important;
            border: none !important;
            color: #ffffff !important;
            font-size: 1.08rem !important;
            font-weight: 700 !important;
            line-height: 1.55 !important;
            white-space: pre-line !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
        }}
        .st-key-landing_btn_50 button {{
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
            box-shadow: 0 10px 26px rgba(37, 99, 235, 0.24) !important;
        }}
        .st-key-landing_btn_500 button {{
            background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
            box-shadow: 0 10px 26px rgba(5, 150, 105, 0.24) !important;
        }}
        .st-key-landing_btn_feedback button {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            box-shadow: 0 10px 26px rgba(99, 102, 241, 0.24) !important;
        }}
        .st-key-landing_btn_50 button:hover,
        .st-key-landing_btn_500 button:hover,
        .st-key-landing_btn_feedback button:hover {{
            transform: translateY(-3px) !important;
            filter: brightness(1.06) !important;
        }}
        .st-key-landing_btn_50 button:hover {{
            box-shadow: 0 16px 32px rgba(37, 99, 235, 0.34) !important;
        }}
        .st-key-landing_btn_500 button:hover {{
            box-shadow: 0 16px 32px rgba(5, 150, 105, 0.34) !important;
        }}
        .st-key-landing_btn_feedback button:hover {{
            box-shadow: 0 16px 32px rgba(99, 102, 241, 0.34) !important;
        }}
        .st-key-landing_btn_50 button p,
        .st-key-landing_btn_500 button p,
        .st-key-landing_btn_feedback button p {{
            color: #ffffff !important;
            font-size: 1.06rem !important;
            font-weight: 700 !important;
        }}
        </style>
        <div style="text-align:center; padding: 95px 16px 40px 16px;">
            <h1 class="landing-title" style="letter-spacing:-0.02em;">S&amp;P 500 매매기법 조합 확률 분석기</h1>
            <p style="color:{theme['text_sub']}; font-size:1.02rem; margin-top:10px;">
                실행할 모드를 선택하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_50, col_500, col_fb, _ = st.columns([0.5, 2, 2, 2, 0.5], gap="large")
    with col_50:
        if st.button(
            "S&P 500 상위 50개\n빠른 스캔 시작",
            type="primary",
            width="stretch",
            key="landing_btn_50",
        ):
            st.session_state["scan_mode"] = "빠른 분석 (시총 상위 50대 종목)"
            st.session_state["app_view"] = "scanner"
            st.session_state["entered_app"] = True
            st.session_state.pop("selected_signal_ticker", None)
            st.session_state.pop("selected_signal_combo", None)
            st.rerun()

    with col_500:
        if st.button(
            f"S&P 500 전체 ({total_sp500}개)\n전 종목 정밀 스캔",
            type="primary",
            width="stretch",
            key="landing_btn_500",
        ):
            st.session_state["scan_mode"] = "S&P 500 전체 종목 스캔 (~500개)"
            st.session_state["app_view"] = "scanner"
            st.session_state["entered_app"] = True
            st.session_state.pop("selected_signal_ticker", None)
            st.session_state.pop("selected_signal_combo", None)
            st.rerun()

    with col_fb:
        if st.button(
            "자동 채점 · 피드백\n4대 트랙 실전 성적표",
            type="primary",
            width="stretch",
            key="landing_btn_feedback",
        ):
            st.session_state["app_view"] = "feedback"
            st.session_state["entered_app"] = True
            st.rerun()


def render_standalone_feedback_view(theme: Dict[str, str]) -> None:
    """Render the dedicated Auto-Grading & 4-Track Feedback view without running stock scans."""
    hdr1, hdr2 = st.columns([5, 1.3])
    with hdr1:
        st.title("자동 채점 및 4대 트랙 피드백 시스템")
        st.caption(
            "상위 50개 / 전체 500개 × 시장국면 ON / OFF 4개 트랙의 가상 매수 기록을 실제 주가로 자동 채점하고 성과를 비교합니다."
        )
    with hdr2:
        st.write("")
        if st.button("메인 화면으로", width="stretch", key="fb_standalone_home"):
            st.session_state["entered_app"] = False
            st.rerun()

    sb_url, sb_key = load_saved_credentials()

    sb_client = None
    if sb_url and sb_key:
        try:
            sb_client = get_supabase_client(sb_url, sb_key)
        except Exception:
            st.error("데이터베이스 연결 초기화 중 오류가 발생했습니다.")

    if sb_client is None:
        render_supabase_setup_guide()
        return

    act_c1, act_c2 = st.columns([2, 1])
    with act_c1:
        if st.button(
            "과거 기록 자동 채점 실행 (익절·손절 판정)",
            type="primary",
            width="stretch",
            key="standalone_grade_btn",
        ):
            with st.spinner("진입가 매수 후 익절선/손절선 도달 여부를 실제 주가로 채점하는 중..."):
                try:
                    res_sum = grade_pending_signals(sb_client)
                    st.success(
                        f"채점 완료 — 총 {res_sum['checked']}건 | "
                        f"익절 도달 {res_sum['wins']}건 | 손절 {res_sum['losses']}건 | "
                        f"만기 청산 {res_sum['expired']}건 | 대기중 {res_sum['still_pending']}건"
                    )
                except Exception:
                    st.error("자동 채점 실행 중 오류가 발생했습니다.")
    with act_c2:
        if st.button("DB 최신 데이터 새로고침", width="stretch", key="standalone_refresh_btn"):
            st.rerun()

    st.write("")
    render_feedback_results(sb_client, theme)


def main() -> None:
    st.session_state.setdefault("entered_app", False)
    st.session_state.setdefault("app_view", "scanner")
    st.session_state.setdefault("theme_name", list(THEMES.keys())[0])
    st.session_state.setdefault("scan_mode", "빠른 분석 (시총 상위 50대 종목)")

    constituents_df = cached_constituents()
    all_sectors = sorted(constituents_df["Sector"].dropna().unique().tolist())

    if not st.session_state["entered_app"]:
        theme = THEMES.get(st.session_state["theme_name"], THEMES[list(THEMES.keys())[0]])
        apply_theme_css(theme)
        render_landing_page(theme, total_sp500=len(constituents_df["Symbol"].unique()))
        return

    if st.session_state.get("app_view") == "feedback":
        with st.sidebar:
            if st.button("메인 화면으로 돌아가기", width="stretch", key="btn_back_home_fb"):
                st.session_state["entered_app"] = False
                st.rerun()
            selected_theme_name = st.radio(
                "화면 테마 선택",
                options=list(THEMES.keys()),
                key="theme_name",
                horizontal=True,
            )
            theme = THEMES[selected_theme_name]
            apply_theme_css(theme)
        render_standalone_feedback_view(theme)
        return

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        if st.button("메인 화면으로 돌아가기", width="stretch", key="btn_back_home"):
            st.session_state["entered_app"] = False
            st.rerun()

        st.header("화면 테마 및 분석 설정")

        selected_theme_name = st.radio(
            "화면 테마 선택",
            options=list(THEMES.keys()),
            key="theme_name",
            horizontal=True,
            help="라이트 테마와 다크 테마를 즉시 전환할 수 있습니다.",
        )
        theme = THEMES[selected_theme_name]
        apply_theme_css(theme)

        st.divider()
        st.subheader("1. 분석 대상 종목 범위")
        scan_mode = st.radio(
            "스캔 모드 선택",
            options=[
                "빠른 분석 (시총 상위 50대 종목)",
                "섹터별 집중 분석 (GICS 섹터)",
                "S&P 500 전체 종목 스캔 (~500개)",
                "직접 관심 티커 입력",
            ],
            key="scan_mode",
        )

        selected_tickers: List[str] = []
        if scan_mode.startswith("빠른"):
            selected_tickers = TOP_50_TICKERS
        elif scan_mode.startswith("섹터별"):
            chosen_sectors = st.multiselect(
                "분석할 섹터 선택",
                options=all_sectors,
                default=all_sectors[:2] if len(all_sectors) >= 2 else all_sectors,
            )
            selected_tickers = (
                constituents_df[constituents_df["Sector"].isin(chosen_sectors)]["Symbol"]
                .unique()
                .tolist()
            )
            st.caption(f"선택된 섹터 종목 수: **{len(selected_tickers)}개**")
        elif scan_mode.startswith("S&P 500"):
            selected_tickers = constituents_df["Symbol"].unique().tolist()
            st.caption(f"S&P 500 전체 대상: **{len(selected_tickers)}개** 종목")
        else:
            raw_input = st.text_area(
                "티커 입력 (쉼표 또는 공백 구분)",
                value="NVDA, AAPL, MSFT, AMZN, META, GOOGL, TSLA, AMD, AVGO, NFLX, JPM, LLY",
            )
            selected_tickers = [
                normalize_ticker(t)
                for t in raw_input.replace("\n", ",").replace(" ", ",").split(",")
                if t.strip()
            ]

        st.divider()
        st.subheader("2. 익절선·손절선 산출 방식")
        tpsl_mode = st.radio(
            "익절/손절 기준 선택",
            options=[
                "매매기법별 맞춤 동적 익절·손절 (권장)",
                "모든 기법 동일 고정 비율(%) 적용",
            ],
            index=0,
        )
        use_dynamic_tpsl = tpsl_mode.startswith("매매기법별")

        period = st.selectbox(
            "백테스팅 과거 데이터 기간",
            options=["1y", "2y", "3y", "5y"],
            index=2,
            format_func=lambda x: {"1y": "최근 1년", "2y": "최근 2년", "3y": "최근 3년 (권장)", "5y": "최근 5년"}[x],
        )
        hold_days = st.slider(
            "스윙 최대 보유 기간 (거래일)",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
        )

        if use_dynamic_tpsl:
            rr_multiplier = st.slider(
                "기법별 익절 목표치 배율 조정 (기본 1.0배)",
                min_value=0.7,
                max_value=1.5,
                value=1.0,
                step=0.1,
            )
            target_pct = 4.0
            stop_pct = -4.0
        else:
            rr_multiplier = 1.0
            target_pct = st.slider("고정 목표 익절률 (+%)", 1.5, 15.0, 4.0, 0.5)
            stop_pct_abs = st.slider("고정 손절 기준 (-%)", 1.5, 15.0, 4.0, 0.5)
            stop_pct = -float(stop_pct_abs)

        with st.expander("3. 승률 향상 실전 필터 (VIX·상대강도·실적·피드백)", expanded=False):
            vix_max = st.slider("VIX 공포지수 상한선", min_value=18.0, max_value=40.0, value=26.0, step=1.0)
            use_rs_filter = st.checkbox(
                "상대강도(RS) 우수 종목만 선별 (SPY 60일 수익률 초과)",
                value=True,
                help="S&P 500 지수보다 강한 주도주에서만 매수 타점을 찾습니다.",
            )
            exclude_earnings = st.checkbox(
                "실적 발표(어닝) 임박 종목 제외 (보유기간 내 갭락 방지)",
                value=True,
                help="향후 보유 기간 이내에 실적 발표가 있는 종목을 자동 제외합니다.",
            )
            use_feedback_weights = st.checkbox(
                "실전 자가 피드백 가중치 자동 반영",
                value=True,
                help="실전 채점 결과 승률이 높은 기법 조합에 가산점을 부여합니다.",
            )
            use_breakeven_stop = st.checkbox(
                "본절 보호 스탑 적용 (목표치 50% 도달 시 본절 방어)",
                value=True,
                help="진입 후 익절 목표의 절반 이상 수익이 나면 손절선을 진입가(+0.1%)로 끌어올려 수익 반납을 막습니다.",
            )
            require_multi_in_high_vix = st.checkbox(
                "고변동성(VIX 25↑) 시 2개 이상 복합 시그널만 추천",
                value=True,
                help="변동성 경계장에서는 단일 지표 속임수를 막기 위해 2~3중 확인 시그널만 추천합니다.",
            )
            use_sector_bonus = st.checkbox(
                "주도 섹터(Leading Sector) 강도 가산점 반영",
                value=True,
                help="최근 60일 수익률이 강한 상위 GICS 섹터 종목에 우선순위 가산점을 부여합니다.",
            )
            use_sma200_filter = st.checkbox(
                "개별 종목 200일선(SMA200) 장기 상승추세만 선별",
                value=False,
                help="장기 하락 추세 종목을 제외하고 200일선 위 상승 추세 종목에서만 매수 타점을 찾습니다.",
            )
            use_walk_forward = st.checkbox(
                "전진 분석(OOS) 최근 구간 검증 가중치 반영",
                value=True,
                help="과거 65% 이후 최근 35% 검증 구간(Out-of-Sample) 적중률까지 함께 평가해 과최적화를 방지합니다.",
            )
            use_partial_tp = st.checkbox(
                "분할 익절 적용 (1차 60% 도달 시 절반 익절 + 본절 방어)",
                value=False,
                help="목표가의 60%에 도달하면 절반을 먼저 수익 확정하고 나머지는 본절 스탑으로 끝까지 보유합니다.",
            )
            next_open_entry = st.checkbox(
                "익일 시가(Open) 진입 + 왕복 수수료(0.1%) 보수적 검증",
                value=False,
                help="당일 종가 대신 다음 거래일 시가에 매수하고 수수료·슬리피지 0.1%를 차감해 실전 괴리율을 검증합니다.",
            )
            slippage_pct = 0.1 if next_open_entry else 0.0
            max_per_sector = st.selectbox(
                "동일 섹터 최대 추천 종목 수 (분산 투자)",
                options=[0, 1, 2, 3],
                index=2,
                format_func=lambda v: "제한 없음" if v == 0 else f"섹터당 상위 {v}개만 추천",
                help="특정 섹터(예: 반도체·빅테크)에 추천 종목이 편중되는 것을 방지합니다.",
            )
            use_strategy_hold_days = st.checkbox(
                "기법 성격별 맞춤 보유 기간 자동 차등 적용 (7~14일)",
                value=True,
                help="단기 반등 기법(볼밴하단·스토캐스틱)은 7일, 추세 돌파 기법(이평선·상단돌파)은 12~14일로 맞춤 적용합니다.",
            )
            auto_tune_rr = st.checkbox(
                "종목별 최적 손익비 배율(0.85~1.2x) 자동 튜닝",
                value=False,
                help="과거 In-Sample 구간에서 기대수익이 가장 높은 익절 목표치 배율을 자동 선택합니다.",
            )
            use_dynamic_sizing = st.checkbox(
                "시장 국면별 동적 포지션 비중 조절 (경계장 0.5% 축소)",
                value=True,
                help="VIX 25 이상이거나 SPY가 200일선 아래일 때 1회 리스크 한도를 1%에서 0.5%로 자동 축소합니다.",
            )
            fast_active_only = st.checkbox(
                "초고속 시그널 전용 스캔 (최근 신호 없는 조합 건너뛰기)",
                value=False,
                help="오늘 매수 가능한 시그널만 초고속으로 찾고 싶을 때 활성화하세요.",
            )

        with st.expander("4. 매매기법 조합 세부 필터 (선택 사항)", expanded=False):
            max_comb = st.select_slider(
                "최대 조합 기법 개수",
                options=[1, 2, 3],
                value=3,
                format_func=lambda v: f"최대 {v}개 조합 ({ {1:8, 2:36, 3:92}[v] }종)",
            )
            signal_window = st.slider("조합 시그널 시차 허용 (일)", 1, 4, 2)
            min_trades = st.slider("최소 과거 발생 횟수 (표본)", 3, 20, 5)
            recent_signal_days = st.slider("최근 시그널 탐지 범위 (거래일)", 1, 5, 2)

        force_refresh = st.button("주가 데이터 새로고침 (캐시 갱신)", width="stretch")

    sb_url, sb_key = load_saved_credentials()

    if force_refresh:
        st.session_state["refresh_nonce"] = int(st.session_state.get("refresh_nonce", 0)) + 1
    refresh_nonce = int(st.session_state.get("refresh_nonce", 0))

    # ---------------- MARKET CONTEXT & FEEDBACK WEIGHTS ----------------
    market_ctx_df, market_summary = cached_market_context(period=period, refresh_nonce=refresh_nonce)

    feedback_bonus_map: Dict[str, float] = (
        cached_feedback_bonus(sb_url, sb_key) if (use_feedback_weights and sb_url and sb_key) else {}
    )

    # ---------------- HEADER & MARKET REGIME TOGGLE ----------------
    col_hdr1, col_hdr2 = st.columns([5, 1.3])
    with col_hdr1:
        st.title("S&P 500 매매기법 조합 확률 및 익절·손절 분석기")
    with col_hdr2:
        st.write("")
        if st.button("사이드바 보이기 / 숨기기", width="stretch"):
            components.html(
                """
                <script>
                const doc = window.parent.document;
                const btn = doc.querySelector(
                    '[data-testid="stSidebarCollapseButton"] button, ' +
                    '[data-testid="stExpandSidebarButton"] button, ' +
                    '[data-testid="collapsedControl"] button'
                );
                if (btn) btn.click();
                </script>
                """,
                height=0,
            )

    regime_choice = st.radio(
        "시장국면 필터 버전 선택 (ON / OFF 비교 가능)",
        options=[
            "시장국면 필터 ON (SPY 200일선 위 + VIX 안정장만 매수)",
            "시장국면 필터 OFF (모든 장세에서 시그널 포착)",
        ],
        index=0,
        horizontal=True,
    )
    use_regime_filter = regime_choice.startswith("시장국면 필터 ON")
    tpsl_label = "기법별 맞춤 익절·손절" if use_dynamic_tpsl else f"고정 익절 +{target_pct}% / 손절 {stop_pct}%"
    regime_text = (
        f"SPY &#36;{market_summary['spy_close']} / VIX {market_summary['vix_close']} ({market_summary['regime_label']})"
        if market_summary.get("available")
        else "시장 데이터 로드 대기"
    )
    fb_text = f"자가 피드백 반영중 ({len(feedback_bonus_map)}개 조합)" if feedback_bonus_map else "기본 백테스트 가중치"
    st.markdown(
        f"""
        <div class="pill-bar">
            <span class="info-pill">대상 종목: {len(selected_tickers)}개</span>
            <span class="info-pill">시장 국면: {regime_text}</span>
            <span class="info-pill">스윙 보유: 최대 {hold_days}거래일</span>
            <span class="info-pill">익절·손절: {tpsl_label}</span>
            <span class="info-pill">피드백: {fb_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not selected_tickers:
        st.warning("좌측 사이드바에서 최소 1개 이상의 분석 대상 종목/섹터를 선택해 주세요.")
        return

    tickers_tuple = tuple(selected_tickers)
    fb_items = tuple(sorted(feedback_bonus_map.items()))
    with st.spinner(f"{len(tickers_tuple)}개 종목 데이터 및 매매기법 분석 중... (캐시 적용 시 즉시 로드)"):
        stock_data = cached_stock_data(
            tickers=tickers_tuple,
            period=period,
            refresh_nonce=refresh_nonce,
        )
        active_signals_df, best_per_stock_df, leaderboard_df, excluded_earnings_tickers = cached_scan_and_filter(
            tickers=tickers_tuple,
            period=period,
            hold_days=hold_days,
            target_pct=target_pct,
            stop_pct=stop_pct,
            signal_window=signal_window,
            min_trades=min_trades,
            recent_signal_days=recent_signal_days,
            max_comb=max_comb,
            use_dynamic_tpsl=use_dynamic_tpsl,
            rr_multiplier=rr_multiplier,
            use_regime_filter=use_regime_filter,
            vix_max=vix_max,
            use_rs_filter=use_rs_filter,
            exclude_earnings=exclude_earnings,
            feedback_items=fb_items,
            use_sma200_filter=use_sma200_filter,
            use_breakeven_stop=use_breakeven_stop,
            require_multi_in_high_vix=require_multi_in_high_vix,
            use_sector_bonus=use_sector_bonus,
            use_walk_forward=use_walk_forward,
            next_open_entry=next_open_entry,
            slippage_pct=slippage_pct,
            use_partial_tp=use_partial_tp,
            max_per_sector=max_per_sector,
            fast_active_only=fast_active_only,
            use_strategy_hold_days=use_strategy_hold_days,
            auto_tune_rr=auto_tune_rr,
            use_dynamic_sizing=use_dynamic_sizing,
            refresh_nonce=refresh_nonce,
        )

    # ---------------- TABS ----------------
    tab_signals, tab_stock, tab_leaderboard, tab_feedback, tab_guide = st.tabs(
        [
            "오늘의 매수 시그널",
            "종목별 차트 및 익절·손절",
            "전체 기법 랭킹",
            "자가 피드백 및 자동 채점",
            "매매기법 설명서",
        ]
    )

    # =========================================================================
    # TAB 1: TODAY'S HIGH-PROBABILITY SIGNALS
    # =========================================================================
    with tab_signals:
        total_analyzed = len(best_per_stock_df)
        total_active = len(active_signals_df)
        high_prob_active = (
            int((active_signals_df["목표 도달 확률(%)"] >= 70.0).sum())
            if not active_signals_df.empty
            else 0
        )
        top_active_prob = (
            f"{active_signals_df['목표 도달 확률(%)'].iloc[0]:.1f}%"
            if not active_signals_df.empty
            else "-"
        )
        top_active_ticker = (
            f"{active_signals_df['Ticker'].iloc[0]} ({active_signals_df['매매기법 조합'].iloc[0]})"
            if not active_signals_df.empty
            else "해당 없음"
        )

        render_kpi_row(
            [
                {"label": "분석 완료 종목", "value": f"{total_analyzed}개", "sub": "최소 표본 조건 충족 종목"},
                {"label": "현재 매수 시그널 포착", "value": f"{total_active}개", "sub": f"전체 확률 기준 (최근 {recent_signal_days}거래일)", "color": theme["accent"]},
                {"label": "익절 도달 70% 이상", "value": f"{high_prob_active}개", "sub": "과거 적중률 70% 이상 종목", "color": theme["tp_color"]},
                {"label": "현재 최고 적중률", "value": top_active_prob, "sub": top_active_ticker, "color": theme["tp_color"]},
            ]
        )

        st.write("")
        if excluded_earnings_tickers:
            st.info(
                f"실적 발표(어닝) 임박 종목 자동 제외({hold_days + 3}일 이내): "
                f"{', '.join(excluded_earnings_tickers)}"
            )

        if active_signals_df.empty:
            st.info(
                "현재 조건에서 최근 매수 시그널이 발생한 종목이 없습니다. "
                "좌측 사이드바에서 '최근 시그널 탐지 범위'를 늘리거나 '스캔 모드'를 넓혀보세요."
            )
        else:
            f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
            with f_col1:
                min_prob_filter = st.slider(
                    "최소 익절 도달 확률(%) 필터",
                    min_value=0,
                    max_value=95,
                    value=70,
                    step=5,
                )
            with f_col2:
                avail_sectors = ["전체"] + sorted(active_signals_df["섹터"].unique().tolist())
                sec_filter = st.selectbox("섹터 필터", options=avail_sectors, index=0)
            with f_col3:
                st.write("")
                st.write("")
                only_today = st.checkbox("마지막 거래일 당일 시그널만 보기", value=False)

            filtered_active = active_signals_df[
                active_signals_df["목표 도달 확률(%)"] >= min_prob_filter
            ].copy()
            if sec_filter != "전체":
                filtered_active = filtered_active[filtered_active["섹터"] == sec_filter]
            if only_today:
                filtered_active = filtered_active[filtered_active["당일 시그널"] == True]

            if filtered_active.empty:
                max_avail_prob = float(active_signals_df["목표 도달 확률(%)"].max())
                st.warning(
                    f"현재 포착된 시그널 {len(active_signals_df)}개 중 설정하신 '{min_prob_filter}% 이상' 조건을 "
                    f"충족하는 종목이 없습니다. (현재 최고 도달 확률: {max_avail_prob:.1f}%) "
                    "위 슬라이더를 낮추거나 아래에서 전체 포착 종목을 확인해 보세요."
                )
                with st.expander(f"참고: 필터 미만 포착 시그널 {len(active_signals_df)}개 보기", expanded=True):
                    st.dataframe(
                        active_signals_df[
                            [
                                "Ticker",
                                "기업명",
                                "매매기법 조합",
                                "목표 도달 확률(%)",
                                "순수 승률(%)",
                                "권장 진입가",
                                "익절가($)",
                                "손절가($)",
                                "마지막 발생일",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "목표 도달 확률(%)": st.column_config.ProgressColumn(
                                "익절 도달 확률", format="%.1f%%", min_value=0, max_value=100
                            ),
                            "순수 승률(%)": st.column_config.ProgressColumn(
                                "순수 승률", format="%.1f%%", min_value=0, max_value=100
                            ),
                            "익절가($)": st.column_config.NumberColumn("익절선", format="$%.2f"),
                            "손절가($)": st.column_config.NumberColumn("손절선", format="$%.2f"),
                        },
                    )
            else:
                st.markdown("#### 오늘의 TOP 3 추천 종목")
                top_cards = filtered_active.head(3)
                card_cols = st.columns(len(top_cards))
                for idx_c, (_, row) in enumerate(top_cards.iterrows()):
                    entry_zone_html = str(row["권장 진입가"]).replace("$", "&#36;")
                    max_chase_html = str(row["추격금지 상한"]).replace("$", "&#36;")
                    with card_cols[idx_c]:
                        st.markdown(
                            f"""
                            <div class="signal-card">
                                <span class="signal-badge">{row['매매기법 조합']}</span>
                                <div class="signal-title">{row['Ticker']} · {row['기업명']}</div>
                                <div class="prob-highlight">익절 도달 확률 {row['목표 도달 확률(%)']:.1f}%</div>
                                <div style="font-size:0.84rem; color:{theme['text_sub']};">
                                    순수 승률 <b>{row['순수 승률(%)']:.1f}%</b> · {row['매매 방법']}
                                </div>
                                <div class="card-grid">
                                    <div class="grid-item">
                                        <div class="grid-label">권장 진입가격대</div>
                                        <div class="grid-val">{entry_zone_html}</div>
                                        <div style="font-size:0.74rem; color:{theme['text_muted']};">추격금지: {max_chase_html}↑</div>
                                    </div>
                                    <div class="grid-item">
                                        <div class="grid-label">권장 비중 · 손익비</div>
                                        <div class="grid-val">1 : {row['손익비(R:R)']:.2f}</div>
                                        <div style="font-size:0.74rem; color:{theme['text_muted']};">{row['권장 비중']}</div>
                                    </div>
                                    <div class="grid-item-tp">
                                        <div class="grid-label">맞춤 익절선</div>
                                        <div class="grid-val-tp">&#36;{row['익절가($)']:.2f}</div>
                                        <div style="font-size:0.75rem; font-weight:700; color:{theme['tp_color']};">+{row['익절폭(+%)']:.2f}%</div>
                                    </div>
                                    <div class="grid-item-sl">
                                        <div class="grid-label">맞춤 손절선</div>
                                        <div class="grid-val-sl">&#36;{row['손절가($)']:.2f}</div>
                                        <div style="font-size:0.75rem; font-weight:700; color:{theme['sl_color']};">{row['손절폭(-%)']:.2f}%</div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"{row['Ticker']} 차트 보기",
                            key=f"btn_chart_top_{row['Ticker']}_{idx_c}",
                            width="stretch",
                        ):
                            st.session_state["selected_signal_ticker"] = str(row["Ticker"])
                            st.session_state["selected_signal_combo"] = str(row["ComboID"])
                            st.session_state["tab_stock_ticker"] = str(row["Ticker"])

                with st.expander("TOP 추천 종목 상세 매매 방법 (주문 순서) 보기", expanded=False):
                    for _, row in top_cards.iterrows():
                        st.markdown(f"**{row['Ticker']} ({row['기업명']}) — {row['매매기법 조합']} | {row['매매 방법']}**")
                        st.text(row["매매 실행 가이드"])

                st.write("")
                show_extra_cols = st.checkbox("표에서 상세 지표(추격금지·비중·상대강도) 더 보기", value=False)
                st.markdown("#### 포착 종목 전체 요약 표")
                st.caption("표에서 종목 행을 클릭하거나 위 추천 카드의 '차트 보기' 버튼을 누르면 아래에 해당 종목의 차트가 즉시 표시됩니다.")
                core_cols = [
                    "Ticker",
                    "기업명",
                    "매매기법 조합",
                    "권장 진입가",
                    "익절가($)",
                    "손절가($)",
                    "목표 도달 확률(%)",
                    "순수 승률(%)",
                    "실적발표(D-day)",
                ]
                extra_cols = [
                    "최근검증(OOS) 확률(%)",
                    "매매 방법",
                    "추격금지 상한",
                    "익절폭(+%)",
                    "손절폭(-%)",
                    "상대강도(RS%)",
                    "권장 비중",
                    "마지막 발생일",
                ]
                display_cols = core_cols + extra_cols if show_extra_cols else core_cols
                table_event = st.dataframe(
                    filtered_active[display_cols],
                    width="stretch",
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="signals_table_sel",
                    column_config={
                        "Ticker": st.column_config.TextColumn("티커", width="small"),
                        "기업명": st.column_config.TextColumn("기업명", width="medium"),
                        "매매기법 조합": st.column_config.TextColumn("매매기법 조합", width="medium"),
                        "기준 진입가($)": st.column_config.NumberColumn("진입가", format="$%.2f"),
                        "익절가($)": st.column_config.NumberColumn("익절선", format="$%.2f"),
                        "익절폭(+%)": st.column_config.NumberColumn("익절폭", format="+%.2f%%"),
                        "손절가($)": st.column_config.NumberColumn("손절선", format="$%.2f"),
                        "손절폭(-%)": st.column_config.NumberColumn("손절폭", format="%.2f%%"),
                        "손익비(R:R)": st.column_config.NumberColumn("손익비", format="1:%.2f"),
                        "목표 도달 확률(%)": st.column_config.ProgressColumn(
                            "익절 도달 확률",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "순수 승률(%)": st.column_config.ProgressColumn(
                            "순수 승률",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "기대 수익률(%)": st.column_config.NumberColumn("기대수익", format="%+.2f%%"),
                    },
                )

                sel_rows = table_event.selection.rows if table_event and table_event.selection else []
                if sel_rows != st.session_state.get("_last_sig_table_rows", []):
                    st.session_state["_last_sig_table_rows"] = list(sel_rows)
                    if sel_rows and sel_rows[0] < len(filtered_active):
                        clicked_row = filtered_active.iloc[sel_rows[0]]
                        st.session_state["selected_signal_ticker"] = str(clicked_row["Ticker"])
                        st.session_state["selected_signal_combo"] = str(clicked_row["ComboID"])
                        st.session_state["tab_stock_ticker"] = str(clicked_row["Ticker"])

                csv_bytes = filtered_active[display_cols].to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "오늘의 시그널 및 익절·손절가 리스트 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"sp500_signals_tpsl_{hold_days}d.csv",
                    mime="text/csv",
                )

                # -------- Inline Quick Chart Preview for Clicked Signal --------
                preview_ticker = st.session_state.get("selected_signal_ticker")
                if preview_ticker and preview_ticker in filtered_active["Ticker"].values:
                    st.divider()
                    preview_row = filtered_active[filtered_active["Ticker"] == preview_ticker].iloc[0]
                    preview_combo_id = str(preview_row["ComboID"])

                    hdr_c1, hdr_c2, hdr_c3 = st.columns([4, 1.3, 1])
                    with hdr_c1:
                        st.markdown(
                            f"#### {preview_ticker} ({preview_row['기업명']}) — 매수 시그널 및 익절·손절 차트"
                        )
                    with hdr_c2:
                        preview_lookback = st.selectbox(
                            "차트 기간",
                            options=[120, 250, 500],
                            index=1,
                            format_func=lambda x: f"최근 {x}일",
                            key="preview_chart_lookback",
                            label_visibility="collapsed",
                        )
                    with hdr_c3:
                        if st.button("차트 닫기", key="btn_close_preview_chart", width="stretch"):
                            st.session_state.pop("selected_signal_ticker", None)
                            st.session_state.pop("selected_signal_combo", None)
                            st.rerun()

                    render_kpi_row(
                        [
                            {
                                "label": "매매기법 조합",
                                "value": str(preview_row["매매기법 조합"]),
                                "sub": f"익절 도달 확률 {preview_row['목표 도달 확률(%)']:.1f}% | 순수 승률 {preview_row['순수 승률(%)']:.1f}%",
                                "color": theme["accent"],
                            },
                            {
                                "label": "권장 진입가",
                                "value": str(preview_row["권장 진입가"]).replace("$", "&#36;"),
                                "sub": f"추격금지: {str(preview_row['추격금지 상한']).replace('$', '&#36;')}↑",
                            },
                            {
                                "label": "맞춤 익절선",
                                "value": f"&#36;{preview_row['익절가($)']:.2f}",
                                "sub": f"+{preview_row['익절폭(+%)']:.2f}%",
                                "color": theme["tp_color"],
                            },
                            {
                                "label": "맞춤 손절선",
                                "value": f"&#36;{preview_row['손절가($)']:.2f}",
                                "sub": f"{preview_row['손절폭(-%)']:.2f}% (손익비 1:{preview_row['손익비(R:R)']:.2f})",
                                "color": theme["sl_color"],
                            },
                        ]
                    )

                    p_df = stock_data.get(preview_ticker)
                    if p_df is None or p_df.empty:
                        p_df = fetch_single_stock(preview_ticker, period=period, use_cache=not force_refresh)
                    if p_df is not None and len(p_df) >= 60:
                        _, p_df_ind, p_combo_signals, p_outcomes_ctx = analyze_single_stock(
                            df=p_df,
                            ticker=preview_ticker,
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
                            min_rs_pct=0.0,
                            feedback_bonus_map=feedback_bonus_map,
                        )
                        p_trades_df = get_trade_history_for_combo(
                            df_ind=p_df_ind,
                            combo_signals=p_combo_signals,
                            outcomes_ctx=p_outcomes_ctx,
                            combo_id=preview_combo_id,
                        )
                        p_fig = build_candlestick_chart(
                            df_ind=p_df_ind,
                            trades_df=p_trades_df,
                            ticker=preview_ticker,
                            combo_name=str(preview_row["매매기법 조합"]),
                            entry_price=float(preview_row["기준 진입가($)"]),
                            tp_price=float(preview_row["익절가($)"]),
                            tp_pct=float(preview_row["익절폭(+%)"]),
                            sl_price=float(preview_row["손절가($)"]),
                            sl_pct=float(preview_row["손절폭(-%)"]),
                            theme=theme,
                            lookback_bars=preview_lookback,
                        )
                        st.plotly_chart(p_fig, width="stretch")

    # =========================================================================
    # TAB 2: INDIVIDUAL STOCK DEEP-DIVE & INTERACTIVE TP/SL CHART
    # =========================================================================
    with tab_stock:
        st.subheader("개별 종목 최적 매매기법 및 익절·손절 차트 분석")

        col_sel1, col_sel2, col_sel3 = st.columns([2, 2, 1])
        avail_tickers = sorted(list(stock_data.keys()))
        synced_ticker = st.session_state.get("tab_stock_ticker")
        if synced_ticker and synced_ticker in avail_tickers:
            default_ticker = synced_ticker
        elif not active_signals_df.empty:
            default_ticker = active_signals_df["Ticker"].iloc[0]
        else:
            default_ticker = "NVDA" if "NVDA" in avail_tickers else avail_tickers[0]

        with col_sel1:
            chosen_ticker = st.selectbox(
                "분석할 종목 선택",
                options=avail_tickers,
                index=avail_tickers.index(default_ticker) if default_ticker in avail_tickers else 0,
                format_func=lambda t: f"{t} ({constituents_df[constituents_df['Symbol']==t]['Security'].values[0]})"
                if not constituents_df[constituents_df["Symbol"] == t].empty
                else t,
            )
        with col_sel2:
            custom_ticker_input = st.text_input(
                "다른 미국 주식 티커 직접 검색 (예: PLTR, COIN, SOFI)",
                value="",
            )
        with col_sel3:
            lookback_bars = st.selectbox(
                "차트 기간",
                options=[120, 250, 500, 750],
                index=1,
                format_func=lambda x: f"최근 {x}일",
            )

        target_ticker = (
            normalize_ticker(custom_ticker_input)
            if custom_ticker_input.strip()
            else chosen_ticker
        )

        if target_ticker in stock_data:
            t_df = stock_data[target_ticker]
        else:
            with st.spinner(f"{target_ticker} 주가 데이터를 가져오는 중..."):
                t_df = fetch_single_stock(target_ticker, period=period, use_cache=not force_refresh)

        if t_df.empty or len(t_df) < 60:
            st.error(f"'{target_ticker}' 종목의 데이터를 불러올 수 없거나 상장 기간이 너무 짧습니다.")
        else:
            stats_df, df_ind, combo_signals, outcomes_ctx = analyze_single_stock(
                df=t_df,
                ticker=target_ticker,
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
                min_rs_pct=0.0,
                feedback_bonus_map=feedback_bonus_map,
            )

            if stats_df.empty:
                st.warning(
                    f"{target_ticker} 종목에서 최소 발생 횟수({min_trades}회) 이상 발생한 매매기법 조합이 없습니다. "
                    "좌측 사이드바의 '매매기법 조합 세부 필터'에서 최소 발생 횟수를 낮춰보세요."
                )
            else:
                combo_options = stats_df["ComboID"].tolist()
                combo_labels = {
                    r["ComboID"]: (
                        f"#{i+1} [{r['매매기법 조합']}] — 익절도달 {r['목표 도달 확률(%)']:.1f}% | "
                        f"순수승률 {r['순수 승률(%)']:.1f}% | 익절 ${r['익절가($)']:.2f}(+{r['익절폭(+%)']:.1f}%) / "
                        f"손절 ${r['손절가($)']:.2f}({r['손절폭(-%)']:.1f}%) "
                        f"{'[최근 시그널]' if r['최근 시그널'] else ''}"
                    )
                    for i, (_, r) in enumerate(stats_df.iterrows())
                }

                selected_combo_id = st.selectbox(
                    "확인할 매매기법 조합 선택 (1위가 현재 종목 최적 조합)",
                    options=combo_options,
                    index=0,
                    format_func=lambda cid: combo_labels.get(cid, cid),
                )

                sel_row = stats_df[stats_df["ComboID"] == selected_combo_id].iloc[0]

                render_kpi_row(
                    [
                        {
                            "label": "기준 진입가",
                            "value": f"&#36;{sel_row['기준 진입가($)']:.2f}",
                            "sub": f"최근 시그널일: {sel_row['마지막 발생일']}",
                            "color": theme["accent"],
                        },
                        {
                            "label": "기법 맞춤 익절선",
                            "value": f"&#36;{sel_row['익절가($)']:.2f} (+{sel_row['익절폭(+%)']:.2f}%)",
                            "sub": f"목표 도달 확률 {sel_row['목표 도달 확률(%)']:.1f}%",
                            "color": theme["tp_color"],
                        },
                        {
                            "label": "기법 맞춤 손절선",
                            "value": f"&#36;{sel_row['손절가($)']:.2f} ({sel_row['손절폭(-%)']:.2f}%)",
                            "sub": f"손절 반영 순수 승률 {sel_row['순수 승률(%)']:.1f}%",
                            "color": theme["sl_color"],
                        },
                        {
                            "label": "손익비 및 기대수익",
                            "value": f"1 : {sel_row['손익비(R:R)']:.2f}",
                            "sub": f"평균 기대수익 {sel_row['기대 수익률(%)']:+.2f}% ({sel_row['발생 횟수']}회)",
                        },
                    ]
                )

                sel_entry_zone_md = str(sel_row["권장 진입가"]).replace("$", "\\$")
                sel_max_chase_md = str(sel_row["추격금지 상한"]).replace("$", "\\$")

                with st.expander("상세 매매 방법 및 익절·손절 산출 근거 보기", expanded=False):
                    st.markdown(
                        f"**매매 전략:** {sel_row['매매 방법']} | **권장 진입가:** {sel_entry_zone_md} "
                        f"(추격 금지: {sel_max_chase_md} 이상) | **권장 비중:** {sel_row['권장 비중']}"
                    )
                    st.text(sel_row["매매 실행 가이드"])
                    st.markdown(
                        f"- **익절선 기준:** {sel_row['익절선 기준']} (평균 +{sel_row['평균 익절폭(+%)']:.2f}%)\n"
                        f"- **손절선 기준:** {sel_row['손절선 기준']} (평균 {sel_row['평균 손절폭(-%)']:.2f}%)"
                    )

                if load_gemini_api_key():
                    if st.button(f"{target_ticker} · Gemini AI 종합 매매 브리핑 생성", key=f"ai_btn_{target_ticker}"):
                        with st.spinner("Gemini AI가 차트·뉴스·시장 국면을 종합 분석하는 중..."):
                            briefing_txt = generate_ai_briefing(
                                ticker=target_ticker,
                                row_data=sel_row.to_dict(),
                                market_summary=market_summary,
                            )
                            st.info(briefing_txt)

                trades_df = get_trade_history_for_combo(
                    df_ind=df_ind,
                    combo_signals=combo_signals,
                    outcomes_ctx=outcomes_ctx,
                    combo_id=selected_combo_id,
                )

                selected_combo_name = format_combo_name(parse_combo_id(selected_combo_id))
                st.markdown(
                    f"#### {target_ticker} 일봉 차트 · `{selected_combo_name}` 타점 및 익절·손절 라인"
                )
                fig = build_candlestick_chart(
                    df_ind=df_ind,
                    trades_df=trades_df,
                    ticker=target_ticker,
                    combo_name=selected_combo_name,
                    entry_price=float(sel_row["기준 진입가($)"]),
                    tp_price=float(sel_row["익절가($)"]),
                    tp_pct=float(sel_row["익절폭(+%)"]),
                    sl_price=float(sel_row["손절가($)"]),
                    sl_pct=float(sel_row["손절폭(-%)"]),
                    theme=theme,
                    lookback_bars=lookback_bars,
                )
                st.plotly_chart(fig, width="stretch")

                sub_t1, sub_t2 = st.tabs(
                    [
                        f"{target_ticker} 전체 매매기법 조합 비교 ({len(stats_df)}개)",
                        f"[{selected_combo_name}] 과거 시그널 발생 타점 상세 ({len(trades_df)}건)",
                    ]
                )

                with sub_t1:
                    st.dataframe(
                        stats_df[
                            [
                                "매매기법 조합",
                                "기법 수",
                                "목표 도달 확률(%)",
                                "순수 승률(%)",
                                "기대 수익률(%)",
                                "기준 진입가($)",
                                "익절가($)",
                                "익절폭(+%)",
                                "손절가($)",
                                "손절폭(-%)",
                                "손익비(R:R)",
                                "발생 횟수",
                                "최근 시그널",
                                "마지막 발생일",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "매매기법 조합": st.column_config.TextColumn("매매기법 조합", width="medium"),
                            "기준 진입가($)": st.column_config.NumberColumn("진입가", format="$%.2f"),
                            "익절가($)": st.column_config.NumberColumn("익절선", format="$%.2f"),
                            "익절폭(+%)": st.column_config.NumberColumn("익절폭", format="+%.2f%%"),
                            "손절가($)": st.column_config.NumberColumn("손절선", format="$%.2f"),
                            "손절폭(-%)": st.column_config.NumberColumn("손절폭", format="%.2f%%"),
                            "손익비(R:R)": st.column_config.NumberColumn("손익비", format="1:%.2f"),
                            "목표 도달 확률(%)": st.column_config.ProgressColumn(
                                "익절 도달 확률", format="%.1f%%", min_value=0, max_value=100
                            ),
                            "순수 승률(%)": st.column_config.ProgressColumn(
                                "순수 승률", format="%.1f%%", min_value=0, max_value=100
                            ),
                            "기대 수익률(%)": st.column_config.NumberColumn("기대수익", format="%+.2f%%"),
                        },
                    )

                with sub_t2:
                    if not trades_df.empty:
                        trades_show = trades_df.sort_values("시그널 일자", ascending=False).copy()
                        trades_show["시그널 일자"] = trades_show["시그널 일자"].dt.strftime("%Y-%m-%d")
                        st.dataframe(
                            trades_show[
                                [
                                    "시그널 일자",
                                    "진입 종가($)",
                                    "익절 목표가($)",
                                    "익절폭(+%)",
                                    "손절 기준가($)",
                                    "손절폭(-%)",
                                    "결과 상태",
                                    "기간내 최고(%)",
                                    "기간내 최저(%)",
                                    "실현 수익률(%)",
                                ]
                            ],
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "진입 종가($)": st.column_config.NumberColumn("진입 종가", format="$%.2f"),
                                "익절 목표가($)": st.column_config.NumberColumn("익절선", format="$%.2f"),
                                "익절폭(+%)": st.column_config.NumberColumn("익절폭", format="+%.2f%%"),
                                "손절 기준가($)": st.column_config.NumberColumn("손절선", format="$%.2f"),
                                "손절폭(-%)": st.column_config.NumberColumn("손절폭", format="%.2f%%"),
                                "기간내 최고(%)": st.column_config.NumberColumn("기간내 최고", format="%+.2f%%"),
                                "기간내 최저(%)": st.column_config.NumberColumn("기간내 최저", format="%+.2f%%"),
                                "실현 수익률(%)": st.column_config.NumberColumn("실현 수익률", format="%+.2f%%"),
                            },
                        )

    # =========================================================================
    # TAB 3: S&P 500 STRATEGY LEADERBOARD
    # =========================================================================
    with tab_leaderboard:
        st.subheader("S&P 500 전체 종목 통합 매매기법 조합 랭킹")
        st.caption(
            f"분석된 총 {len(best_per_stock_df)}개 종목에서 각 매매기법 조합이 기록한 평균 익절 도달 확률 및 기대수익률 순위입니다."
        )

        if leaderboard_df.empty:
            st.warning("리더보드를 생성할 데이터가 충분하지 않습니다.")
        else:
            top15 = leaderboard_df.head(15).iloc[::-1]
            max_prob = float(top15["평균_목표도달확률"].max()) if not top15.empty else 100.0
            fig_bar = px.bar(
                top15,
                x="평균_목표도달확률",
                y="매매기법 조합",
                color="평균_기대수익률",
                orientation="h",
                text="평균_목표도달확률",
                color_continuous_scale="Tealgrn",
                labels={
                    "평균_목표도달확률": "평균 익절 도달 확률 (%)",
                    "매매기법 조합": "",
                    "평균_기대수익률": "기대수익(%)",
                },
                title="S&P 500 평균 익절 도달 확률 TOP 15 매매기법 조합",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig_bar.update_layout(
                height=540,
                template=theme["plotly_template"],
                paper_bgcolor=theme["chart_bg"],
                plot_bgcolor=theme["chart_bg"],
                font=dict(color=theme["text_main"], family="Pretendard, Malgun Gothic, sans-serif"),
                margin=dict(l=20, r=70, t=55, b=30),
                xaxis=dict(range=[0, max(100.0, max_prob * 1.18)], gridcolor=theme["grid_color"]),
                yaxis=dict(automargin=True),
            )
            st.plotly_chart(fig_bar, width="stretch")

            st.markdown("#### 41개 매매기법 조합 전체 통계 비교")
            st.dataframe(
                leaderboard_df.rename(
                    columns={
                        "검증종목수": "적용 종목수",
                        "총발생횟수": "총 시그널수",
                        "평균_목표도달확률": "평균 익절확률(%)",
                        "평균_순수승률": "평균 순수승률(%)",
                        "평균_기대수익률": "평균 기대수익(%)",
                        "평균_익절폭": "평균 익절폭(+%)",
                        "평균_손절폭": "평균 손절폭(-%)",
                        "승률70이상_종목수": "적중률 70% 이상 종목",
                        "현재시그널_종목수": "현재 시그널 종목",
                    }
                ).drop(columns=["ComboID"]),
                width="stretch",
                hide_index=True,
            )

            st.divider()
            st.markdown("#### 종목별 1위 매매기법 조합 및 현재 맞춤 익절·손절가")
            if not best_per_stock_df.empty:
                st.dataframe(
                    best_per_stock_df[
                        [
                            "Ticker",
                            "기업명",
                            "섹터",
                            "현재가($)",
                            "매매기법 조합",
                            "목표 도달 확률(%)",
                            "순수 승률(%)",
                            "기대 수익률(%)",
                            "기준 진입가($)",
                            "익절가($)",
                            "익절폭(+%)",
                            "손절가($)",
                            "손절폭(-%)",
                            "손익비(R:R)",
                            "발생 횟수",
                            "최근 시그널",
                            "마지막 발생일",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "현재가($)": st.column_config.NumberColumn("현재가", format="$%.2f"),
                        "기준 진입가($)": st.column_config.NumberColumn("진입가", format="$%.2f"),
                        "익절가($)": st.column_config.NumberColumn("익절선", format="$%.2f"),
                        "익절폭(+%)": st.column_config.NumberColumn("익절폭", format="+%.2f%%"),
                        "손절가($)": st.column_config.NumberColumn("손절선", format="$%.2f"),
                        "손절폭(-%)": st.column_config.NumberColumn("손절폭", format="%.2f%%"),
                        "손익비(R:R)": st.column_config.NumberColumn("손익비", format="1:%.2f"),
                        "목표 도달 확률(%)": st.column_config.NumberColumn("익절확률", format="%.1f%%"),
                        "순수 승률(%)": st.column_config.NumberColumn("순수승률", format="%.1f%%"),
                        "기대 수익률(%)": st.column_config.NumberColumn("기대수익", format="%+.2f%%"),
                    },
                )

    # =========================================================================
    # TAB 4: SUPABASE SELF-FEEDBACK & AUTO-GRADING
    # =========================================================================
    with tab_feedback:
        st.subheader("자가 피드백 및 예측 자동 채점 시스템")
        st.caption(
            "포착된 매수 시그널을 데이터베이스에 저장하고, 실제 주가로 성공/손절 여부를 자동 채점하여 "
            "각 매매기법 조합의 실전 적중률과 피드백 가중치를 스스로 업데이트합니다."
        )

        sb_client = None
        if sb_url and sb_key:
            try:
                sb_client = get_supabase_client(sb_url, sb_key)
            except Exception:
                st.error("데이터베이스 연결 초기화 중 오류가 발생했습니다.")

        if sb_client is None:
            render_supabase_setup_guide()
        else:
            fb_min_prob = st.slider(
                "가상 매수 기록 최소 익절 도달 확률 (%)",
                min_value=50,
                max_value=95,
                value=70,
                step=5,
                help="설정한 확률 이상인 시그널만 진입가에 매수했다고 가정하여 DB에 기록하고 익절/손절 도달 여부를 추적합니다.",
            )
            is_cur_top50 = scan_mode.startswith("빠른")
            cur_scope_code = "TOP50" if is_cur_top50 else "ALL"
            cur_scope_name = "상위 50개" if is_cur_top50 else "전체 종목"
            cur_ver_label = "국면 ON" if use_regime_filter else "국면 OFF"

            col_fb_btn1, col_fb_btn2, col_fb_btn3 = st.columns(3)
            with col_fb_btn1:
                if st.button(f"현재({cur_scope_name} · {cur_ver_label}) {fb_min_prob}%↑ 기록", width="stretch"):
                    if active_signals_df.empty:
                        st.warning("현재 포착된 시그널이 없어 저장할 항목이 없습니다.")
                    else:
                        try:
                            saved_cnt = save_signals_to_db(
                                sb_client,
                                active_signals_df,
                                hold_days=hold_days,
                                min_prob=float(fb_min_prob),
                                regime_filter_on=use_regime_filter,
                                rs_filter_on=use_rs_filter,
                                universe_scope=cur_scope_code,
                            )
                            if saved_cnt == 0:
                                st.info(f"익절 도달 확률 {fb_min_prob}% 이상인 현재 시그널이 없습니다.")
                            else:
                                st.success(
                                    f"[{cur_scope_name} · {cur_ver_label}] {fb_min_prob}%↑ 시그널 {saved_cnt}건을 "
                                    "가상 매수(PENDING)로 기록했습니다."
                                )
                        except Exception as exc:
                            st.error(f"시그널 저장 오류: {exc}")

            with col_fb_btn2:
                if st.button(
                    f"4대 트랙 동시 기록 ({fb_min_prob}%↑)",
                    width="stretch",
                    help="상위 50 종목(국면 ON/OFF)과 전체 종목(국면 ON/OFF) 총 4개 트랙을 한 번에 스캔하여 DB에 기록합니다.",
                ):
                    with st.spinner("4개 트랙(상위 50 ON/OFF + 전체 종목 ON/OFF)을 스캔하여 DB에 기록하는 중..."):
                        try:
                            # 1) Top 50 Universe
                            stock_data_top50 = {t: stock_data[t] for t in TOP_50_TICKERS if t in stock_data}
                            if not stock_data_top50:
                                stock_data_top50 = fetch_multiple_stocks(TOP_50_TICKERS, period=period, use_cache=True)
                            const_top50 = constituents_df[constituents_df["Symbol"].isin(TOP_50_TICKERS)]

                            act_top50_on, _, _ = scan_market(
                                stock_data=stock_data_top50, constituents_df=const_top50,
                                hold_days=hold_days, target_pct=target_pct, stop_pct=stop_pct,
                                signal_window=signal_window, min_trades=min_trades,
                                recent_signal_days=recent_signal_days, max_comb=max_comb,
                                use_dynamic_tpsl=use_dynamic_tpsl, rr_multiplier=rr_multiplier,
                                market_ctx_df=market_ctx_df, use_regime_filter=True, vix_max=vix_max,
                                use_rs_filter=use_rs_filter, feedback_bonus_map=feedback_bonus_map,
                            )
                            act_top50_off, _, _ = scan_market(
                                stock_data=stock_data_top50, constituents_df=const_top50,
                                hold_days=hold_days, target_pct=target_pct, stop_pct=stop_pct,
                                signal_window=signal_window, min_trades=min_trades,
                                recent_signal_days=recent_signal_days, max_comb=max_comb,
                                use_dynamic_tpsl=use_dynamic_tpsl, rr_multiplier=rr_multiplier,
                                market_ctx_df=market_ctx_df, use_regime_filter=False, vix_max=vix_max,
                                use_rs_filter=use_rs_filter, feedback_bonus_map=feedback_bonus_map,
                            )

                            # 2) Full Universe (Current stock_data)
                            act_all_on, _, _ = scan_market(
                                stock_data=stock_data, constituents_df=constituents_df,
                                hold_days=hold_days, target_pct=target_pct, stop_pct=stop_pct,
                                signal_window=signal_window, min_trades=min_trades,
                                recent_signal_days=recent_signal_days, max_comb=max_comb,
                                use_dynamic_tpsl=use_dynamic_tpsl, rr_multiplier=rr_multiplier,
                                market_ctx_df=market_ctx_df, use_regime_filter=True, vix_max=vix_max,
                                use_rs_filter=use_rs_filter, feedback_bonus_map=feedback_bonus_map,
                            )
                            act_all_off, _, _ = scan_market(
                                stock_data=stock_data, constituents_df=constituents_df,
                                hold_days=hold_days, target_pct=target_pct, stop_pct=stop_pct,
                                signal_window=signal_window, min_trades=min_trades,
                                recent_signal_days=recent_signal_days, max_comb=max_comb,
                                use_dynamic_tpsl=use_dynamic_tpsl, rr_multiplier=rr_multiplier,
                                market_ctx_df=market_ctx_df, use_regime_filter=False, vix_max=vix_max,
                                use_rs_filter=use_rs_filter, feedback_bonus_map=feedback_bonus_map,
                            )

                            # Save all 4 tracks
                            cnt_t50_on = save_signals_to_db(
                                sb_client, act_top50_on, hold_days=hold_days,
                                min_prob=float(fb_min_prob), regime_filter_on=True,
                                rs_filter_on=use_rs_filter, universe_scope="TOP50",
                            )
                            cnt_t50_off = save_signals_to_db(
                                sb_client, act_top50_off, hold_days=hold_days,
                                min_prob=float(fb_min_prob), regime_filter_on=False,
                                rs_filter_on=use_rs_filter, universe_scope="TOP50",
                            )
                            cnt_all_on = save_signals_to_db(
                                sb_client, act_all_on, hold_days=hold_days,
                                min_prob=float(fb_min_prob), regime_filter_on=True,
                                rs_filter_on=use_rs_filter, universe_scope="ALL",
                            )
                            cnt_all_off = save_signals_to_db(
                                sb_client, act_all_off, hold_days=hold_days,
                                min_prob=float(fb_min_prob), regime_filter_on=False,
                                rs_filter_on=use_rs_filter, universe_scope="ALL",
                            )
                            st.success(
                                f"4대 트랙 동시 기록 완료 — "
                                f"상위50 국면ON: {cnt_t50_on}건 | 상위50 국면OFF: {cnt_t50_off}건 | "
                                f"전체 국면ON: {cnt_all_on}건 | 전체 국면OFF: {cnt_all_off}건"
                            )
                        except Exception as exc:
                            st.error(f"4대 트랙 동시 기록 오류: {exc}")

            with col_fb_btn3:
                if st.button("과거 기록 자동 채점 (익절·손절 판정)", width="stretch"):
                    with st.spinner("진입가 매수 후 익절선/손절선 도달 여부를 실제 주가로 채점하는 중..."):
                        try:
                            res_sum = grade_pending_signals(sb_client)
                            st.success(
                                f"채점 완료 — 총 {res_sum['checked']}건 | "
                                f"익절 도달 {res_sum['wins']}건 | 손절 {res_sum['losses']}건 | "
                                f"만기 청산 {res_sum['expired']}건 | 대기중 {res_sum['still_pending']}건"
                            )
                        except Exception as exc:
                            st.error(f"자동 채점 오류: {exc}")

            render_feedback_results(sb_client, theme)

    # =========================================================================
    # TAB 5: STRATEGY & TP/SL GUIDE
    # =========================================================================
    with tab_guide:
        st.subheader("8대 핵심 매매기법별 익절선(Take-Profit) · 손절선(Stop-Loss) 설계 원리")
        st.markdown(
            """
            단순히 모든 종목과 기법에 일괄 고정 비율(`+4% / -4%`)을 적용하면, **역추세(볼린저 하단 반등)** 기법과 **추세 돌파(볼린저 상단·거래량 돌파)** 기법의
            실제 차트 지지선·저항선 및 종목별 변동성(`ATR`)을 반영하지 못합니다.  
            본 프로그램은 **각 매매기법의 기술적 원리에 맞춘 동적 익절선·손절선**을 계산하며, 2~3개 기법을 조합할 때는 조합된 기법들의 목표 저항선과 핵심 지지선을 결합합니다.
            """
        )

        guide_rows = []
        for code, meta in STRATEGY_METADATA.items():
            guide_rows.append(
                {
                    "코드": code,
                    "기법 명칭": meta["name"],
                    "성격": meta["category"],
                    "매수 진입 조건": meta["desc"],
                    "맞춤 익절선 (Take-Profit)": meta["tp_rule"],
                    "맞춤 손절선 (Stop-Loss)": meta["sl_rule"],
                }
            )
        st.dataframe(pd.DataFrame(guide_rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
