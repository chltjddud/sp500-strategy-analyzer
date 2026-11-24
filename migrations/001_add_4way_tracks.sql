-- Migration 001: Upgrade existing `signal_logs` table to 4-way feedback tracks & add query indexes.
-- Run this only if you already created `signal_logs` from an older schema version.

ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS regime_filter_on BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS rs_filter_on BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS universe_scope TEXT NOT NULL DEFAULT 'ALL';
ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS is_top50 BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS exit_date DATE;
ALTER TABLE signal_logs ADD COLUMN IF NOT EXISTS days_held INTEGER;

ALTER TABLE signal_logs DROP CONSTRAINT IF EXISTS signal_logs_ticker_combo_id_signal_date_key;
DROP INDEX IF EXISTS idx_signal_logs_unique_regime;
DROP INDEX IF EXISTS idx_signal_logs_unique_4way;

CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_logs_unique_4way
    ON signal_logs (ticker, combo_id, signal_date, regime_filter_on, universe_scope);

CREATE INDEX IF NOT EXISTS idx_signal_logs_status
    ON signal_logs (status);
CREATE INDEX IF NOT EXISTS idx_signal_logs_signal_date
    ON signal_logs (signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_signal_logs_ticker
    ON signal_logs (ticker);
