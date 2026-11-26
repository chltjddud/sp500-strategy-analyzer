-- Migration 002: Enable Row-Level Security (RLS) on all public tables.
-- Blocks external anon/publishable key access; server-side jobs and Streamlit
-- authenticate using Secret key (sb_secret_... / service_role) which bypasses RLS.

ALTER TABLE strategy_baseline ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_feedback ENABLE ROW LEVEL SECURITY;
