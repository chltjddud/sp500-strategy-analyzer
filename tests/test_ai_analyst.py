"""Unit tests for src/ai_analyst.py using mocked HTTP & news calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ai_analyst import generate_ai_briefing, load_gemini_models


def test_load_gemini_models_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODELS", "gemini-3.8-flash, gemini-3.6-flash")
    assert load_gemini_models() == ["gemini-3.8-flash", "gemini-3.6-flash"]


def test_generate_ai_briefing_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("src.ai_analyst.load_gemini_api_key", return_value=""):
        res = generate_ai_briefing("AAPL", {}, {})
    assert "Gemini API Key가 설정되어 있지 않습니다" in res


def test_generate_ai_briefing_uses_header_and_succeeds() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "1. 종합 매수 적합도: 상"}]}}]
    }

    with patch("src.ai_analyst.fetch_recent_news_titles", return_value=["Apple announces earnings"]), \
         patch("src.ai_analyst.requests.post", return_value=mock_resp) as mock_post:
        res = generate_ai_briefing(
            ticker="AAPL",
            row_data={"매매기법 조합": "RSI + MACD"},
            market_summary={"spy_close": 500, "vix_close": 15, "regime_label": "상승 안정장"},
            api_key="test-secret-key",
            models=["gemini-3.8-flash"],
        )

    assert "종합 매수 적합도: 상" in res
    called_url = mock_post.call_args.args[0]
    called_headers = mock_post.call_args.kwargs["headers"]
    assert "test-secret-key" not in called_url
    assert called_headers["x-goog-api-key"] == "test-secret-key"


def test_generate_ai_briefing_stops_immediately_on_401() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    with patch("src.ai_analyst.fetch_recent_news_titles", return_value=[]), \
         patch("src.ai_analyst.requests.post", return_value=mock_resp) as mock_post:
        res = generate_ai_briefing(
            ticker="AAPL",
            row_data={},
            market_summary={},
            api_key="bad-key",
            models=["gemini-3.8-flash", "gemini-3.6-flash"],
        )

    assert "인증에 실패했습니다" in res
    assert mock_post.call_count == 1


def test_generate_ai_briefing_retries_429_and_falls_back() -> None:
    resp_429 = MagicMock(status_code=429, text="Rate limit")
    resp_200 = MagicMock(status_code=200)
    resp_200.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "대체 모델 응답 성공"}]}}]
    }

    with patch("src.ai_analyst.fetch_recent_news_titles", return_value=[]), \
         patch("src.ai_analyst.time.sleep") as mock_sleep, \
         patch("src.ai_analyst.requests.post", side_effect=[resp_429, resp_429, resp_200]) as mock_post:
        res = generate_ai_briefing(
            ticker="NVDA",
            row_data={},
            market_summary={},
            api_key="valid-key",
            models=["gemini-3.8-flash", "gemini-3.6-flash"],
        )

    assert res == "대체 모델 응답 성공"
    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 1
