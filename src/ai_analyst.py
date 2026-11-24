"""
Google Gemini AI Stock & Strategy Briefing Module.
Reads GEMINI_API_KEY securely from environment, .env, or Streamlit secrets.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

from src.data_loader import BASE_DIR

logger = logging.getLogger(__name__)

ENV_FILE = BASE_DIR / ".env"
SECRETS_FILE = BASE_DIR / ".streamlit" / "secrets.toml"
DEFAULT_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def _read_config_value(key_name: str) -> str:
    """Read a config value from env, Streamlit st.secrets, .env, or secrets.toml."""
    val = os.environ.get(key_name, "").strip()
    if val:
        return val

    try:
        import streamlit as st

        if key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception as exc:
        logger.debug("st.secrets lookup failed for %s: %s", key_name, exc)

    for fpath in (ENV_FILE, SECRETS_FILE):
        if not fpath.exists():
            continue
        try:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key_name:
                    cleaned = v.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
        except OSError as exc:
            logger.warning("Failed to read config file %s: %s", fpath, exc)
    return ""


def load_gemini_api_key() -> str:
    """Load GEMINI_API_KEY from env, Streamlit st.secrets, .env, or secrets.toml."""
    return _read_config_value("GEMINI_API_KEY")


def load_gemini_models() -> List[str]:
    """Load comma-separated GEMINI_MODELS from config, falling back to defaults."""
    raw = _read_config_value("GEMINI_MODELS")
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    return list(DEFAULT_GEMINI_MODELS)


def fetch_recent_news_titles(ticker: str, max_items: int = 5) -> List[str]:
    """Fetch recent news headlines for a ticker via yfinance."""
    titles: List[str] = []
    try:
        news = yf.Ticker(ticker).news or []
        for item in news[:max_items]:
            title = ""
            if isinstance(item, dict):
                if "title" in item:
                    title = str(item["title"])
                elif isinstance(item.get("content"), dict):
                    title = str(item["content"].get("title", ""))
            if title:
                titles.append(title)
    except Exception as exc:
        logger.warning("Failed to fetch news headlines for %s: %s", ticker, exc)
    return titles


def generate_ai_briefing(
    ticker: str,
    row_data: Dict[str, Any],
    market_summary: Dict[str, Any],
    api_key: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> str:
    """
    Call Google Gemini API to generate a concise, practical Korean trading briefing
    combining strategy backtest stats, market regime, entry/TP/SL levels, and recent news.
    """
    key = (api_key or load_gemini_api_key()).strip()
    if not key:
        return "Gemini API Key가 설정되어 있지 않습니다. .env 또는 Streamlit Secrets에 GEMINI_API_KEY를 추가해 주세요."

    news_titles = fetch_recent_news_titles(ticker, max_items=5)
    news_text = "\n".join(f"- {t}" for t in news_titles) if news_titles else "- 최근 주요 뉴스 없음"

    prompt = (
        f"당신은 미국 주식 단기 스윙 트레이딩 전문 퀀트 애널리스트입니다.\n"
        f"아래 정량 백테스팅 결과와 시장 상황, 최근 뉴스를 바탕으로 한국어로 간결하고 실전적인 매매 브리핑을 작성해 주세요.\n\n"
        f"[종목 및 전략 정보]\n"
        f"- 종목 티커: {ticker}\n"
        f"- 매매기법 조합: {row_data.get('매매기법 조합')}\n"
        f"- 매매 방법: {row_data.get('매매 방법')}\n"
        f"- 과거 목표 도달 확률: {row_data.get('목표 도달 확률(%)')}% / 순수 승률: {row_data.get('순수 승률(%)')}% (표본 {row_data.get('발생 횟수')}회)\n"
        f"- 평균 기대수익률: {row_data.get('기대 수익률(%)')}% | 손익비: 1:{row_data.get('손익비(R:R)')}\n"
        f"- 권장 진입가격대: {row_data.get('권장 진입가')} (추격금지: {row_data.get('추격금지 상한')})\n"
        f"- 맞춤 익절선: ${row_data.get('익절가($)')} (+{row_data.get('익절폭(+%)')}%)\n"
        f"- 맞춤 손절선: ${row_data.get('손절가($)')} ({row_data.get('손절폭(-%)')}%)\n"
        f"- 지수 대비 상대강도(60일 RS): {row_data.get('상대강도(RS%)')}%\n\n"
        f"[전체 시장 국면]\n"
        f"- SPY: ${market_summary.get('spy_close', '-')} | VIX: {market_summary.get('vix_close', '-')} | 상태: {market_summary.get('regime_label', '-')}\n\n"
        f"[최근 뉴스 헤드라인]\n"
        f"{news_text}\n\n"
        f"[작성 형식 - 아래 3가지 항목으로 6~8줄 이내로 핵심만 작성]\n"
        f"1. 종합 매수 적합도 (상/중/하) 및 이유\n"
        f"2. 진입 및 리스크 관리 포인트 (뉴스/시장 국면 반영)\n"
        f"3. 주의해야 할 변수"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 650},
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }

    target_models = models if models else load_gemini_models()
    last_err = ""

    for model_name in target_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts).strip()
                        if text:
                            return text
                    last_err = f"{model_name}: 응답 본문이 비어 있습니다."
                    break

                if resp.status_code in (401, 403):
                    logger.error("Gemini API authentication error (%s)", resp.status_code)
                    return "Gemini API Key 인증에 실패했습니다. 키 유효성과 권한을 확인해 주세요."

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"{model_name} HTTP {resp.status_code}"
                    logger.warning("Transient Gemini error on %s (attempt %d): HTTP %s", model_name, attempt + 1, resp.status_code)
                    if attempt == 0:
                        time.sleep(0.8)
                        continue
                    break

                # Other 4xx errors (e.g. 404 model not found, 400 bad request): move to next model
                last_err = f"{model_name} HTTP {resp.status_code}: {resp.text[:160]}"
                logger.warning("Gemini request failed on %s: %s", model_name, last_err)
                break
            except requests.RequestException as exc:
                last_err = f"{model_name} 요청 오류: {type(exc).__name__}"
                logger.warning("Gemini network error on %s (attempt %d): %s", model_name, attempt + 1, exc)
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                break

    return f"Gemini AI 응답을 가져오지 못했습니다. ({last_err})"
