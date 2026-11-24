"""Unit tests for scripts/daily_feedback_job.py holiday detection."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.daily_feedback_job import is_us_market_trading_day


def test_is_us_market_trading_day_open_and_holiday() -> None:
    et = ZoneInfo("America/New_York")
    ctx = pd.DataFrame(
        {"SPY_CLOSE": [580.0, 582.0]},
        index=pd.to_datetime(["2026-11-23", "2026-11-24"]),
    )

    is_open, _ = is_us_market_trading_day(ctx, now_et=datetime(2026, 11, 24, 17, 0, tzinfo=et))
    assert is_open is True

    # Thanksgiving holiday (no bar for 2026-11-26)
    is_holiday, msg = is_us_market_trading_day(ctx, now_et=datetime(2026, 11, 26, 17, 0, tzinfo=et))
    assert is_holiday is False
    assert "휴장일" in msg
