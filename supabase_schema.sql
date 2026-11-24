-- S&P 500 매매기법 조합 분석기 & 자가 피드백 DB 초기 스키마
-- Supabase 대시보드 -> SQL Editor -> New Query 에 전체 내용을 붙여넣고 Run 을 클릭하세요.
-- 기존 테이블을 4대 트랙 스키마로 업그레이드할 때는 migrations/001_add_4way_tracks.sql 을 실행하세요.

-- 1. Colab 대용량 초반 분석 결과 저장 테이블
CREATE TABLE IF NOT EXISTS strategy_baseline (
    ticker TEXT NOT NULL,
    combo_id TEXT NOT NULL,
    combo_name TEXT NOT NULL,
    num_strategies INTEGER NOT NULL DEFAULT 1,
    sample_count INTEGER NOT NULL DEFAULT 0,
    hit_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    expected_return DOUBLE PRECISION NOT NULL DEFAULT 0,
    composite_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, combo_id)
);

-- 2. 매수 시그널 가상 매매 기록 및 자동 채점 테이블
CREATE TABLE IF NOT EXISTS signal_logs (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    combo_id TEXT NOT NULL,
    combo_name TEXT NOT NULL,
    signal_date DATE NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    tp_price DOUBLE PRECISION NOT NULL,
    sl_price DOUBLE PRECISION NOT NULL,
    hold_days INTEGER NOT NULL DEFAULT 10,
    predicted_hit_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    regime_filter_on BOOLEAN NOT NULL DEFAULT TRUE,
    rs_filter_on BOOLEAN NOT NULL DEFAULT TRUE,
    universe_scope TEXT NOT NULL DEFAULT 'ALL', -- 'TOP50' | 'ALL'
    is_top50 BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING | WIN | LOSS | EXPIRED_WIN | EXPIRED_LOSS
    actual_return DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    exit_date DATE,
    days_held INTEGER,
    graded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4대 트랙 중복 방지 유니크 인덱스 & 자주 조회되는 컬럼 인덱스
CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_logs_unique_4way
    ON signal_logs (ticker, combo_id, signal_date, regime_filter_on, universe_scope);
CREATE INDEX IF NOT EXISTS idx_signal_logs_status
    ON signal_logs (status);
CREATE INDEX IF NOT EXISTS idx_signal_logs_signal_date
    ON signal_logs (signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_logs_ticker
    ON signal_logs (ticker);

-- 3. 기법 조합별 실전 채점 기반 자가 피드백 가중치 테이블
CREATE TABLE IF NOT EXISTS strategy_feedback (
    combo_id TEXT PRIMARY KEY,
    combo_name TEXT NOT NULL,
    total_graded INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    live_win_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_live_return DOUBLE PRECISION NOT NULL DEFAULT 0,
    feedback_bonus DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- RLS (Row Level Security) 설정
-- 개인 프로젝트에서는 anon key로 읽기/쓰기할 수 있도록 비활성화되어 있습니다.
-- 멀티 유저/공개 배포 시에는 아래처럼 RLS를 활성화하고 정책을 적용하세요:
--   ALTER TABLE signal_logs ENABLE ROW LEVEL SECURITY;
--   CREATE POLICY "public_read" ON signal_logs FOR SELECT USING (true);
--   CREATE POLICY "service_write" ON signal_logs FOR ALL USING (auth.role() = 'service_role');
-- ============================================================================
ALTER TABLE strategy_baseline DISABLE ROW LEVEL SECURITY;
ALTER TABLE signal_logs DISABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_feedback DISABLE ROW LEVEL SECURITY;
