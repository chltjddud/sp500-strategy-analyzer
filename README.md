# S&P 500 매매기법 조합 확률 분석기

S&P 500 전 종목을 대상으로 8개 기본 매매 전략(총 92개 조합)의 과거 백테스트 승률·기대수익률·동적 익절/손절선을 분석하고, Supabase 기반 자가 피드백 채점 및 Google Gemini AI 브리핑을 제공하는 Streamlit 대시보드입니다.

- **배포 URL:** https://sp500-strategy-analyzer-gyoo6cshuvxzfuuzq5bdx9.streamlit.app/

## 주요 기능

- **다중 전략 조합 백테스팅:** 8개 기본 기법의 1~3중 조합(92개)에 대해 Look-ahead bias 없는 정량 백테스팅 및 Walk-forward OOS 검증
- **시장 국면 & 상대강도 필터:** SPY(SMA50/SMA200) 및 VIX 기반 시장 국면 필터 on/off, 60일 상대강도(RS) 필터
- **자동 피드백 & 4대 트랙 채점:** GitHub Actions로 미국 장 마감 후 매일 자동 채점 및 고확률 시그널 기록 (상위 50 / 전체 500 × 국면 필터 ON/OFF)
- **Gemini AI 매매 브리핑:** 종목별 백테스트 결과, 시장 국면, 최근 뉴스 헤드라인을 결합한 한국어 브리핑

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 복사해 `.env` 파일을 생성하고 값을 채웁니다. (Streamlit Cloud 배포 시 `.streamlit/secrets.toml` 사용)

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_api_key
GEMINI_API_KEY=your_gemini_api_key
# 선택: fallback 모델 순서 커스터마이징
# GEMINI_MODELS=gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-flash-latest
```

### 3. 대시보드 실행

```bash
python -m streamlit run app.py
```

Windows에서는 `run_dashboard.bat`을 더블클릭하거나 터미널에서 실행하면 바로 대시보드가 열립니다.

```bat
run_dashboard.bat
```

## Supabase DB 설정

1. Supabase 대시보드 → **SQL Editor** → **New Query**에서 `supabase_schema.sql` 전체를 실행합니다.
2. 기존 구버전 `signal_logs` 테이블을 사용 중이라면 `migrations/001_add_4way_tracks.sql`을 추가로 실행합니다.

## 일일 자동 채점 (GitHub Actions)

- `.github/workflows/daily_feedback.yml`이 매주 화~토 07:00 KST (UTC 월~금 22:00)에 `scripts/daily_feedback_job.py`를 실행합니다.
- 미국 증시 휴장일에는 신규 시그널 기록을 자동으로 건너뜁니다. (수동 강제 실행: `FORCE_RUN=1`)
- Repository Secrets: `SUPABASE_URL`, `SUPABASE_KEY` (선택: 실패 알림용 `SLACK_WEBHOOK_URL`). `GEMINI_API_KEY`는 필요하지 않습니다.

## 테스트 및 코드 품질

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ruff check src scripts tests
```

## 프로젝트 구조

```text
stock-prediction/
├── app.py                        # Streamlit 메인 대시보드
├── run_dashboard.bat             # Windows 실행 스크립트
├── supabase_schema.sql           # Supabase 초기 스키마 및 인덱스
├── migrations/                   # DB 마이그레이션 SQL
├── scripts/
│   └── daily_feedback_job.py     # 일일 자동 채점 및 4대 트랙 스캔 잡
├── src/
│   ├── ai_analyst.py             # Gemini AI 매매 브리핑 모듈
│   ├── backtester.py             # 정량 백테스팅 & 스캔 엔진
│   ├── data_loader.py            # S&P 500 구성종목·OHLCV 캐시·실적일 로더
│   ├── feedback_db.py            # Supabase 연동 및 4-way 피드백 채점
│   └── strategies.py             # 8개 기본 전략 및 92개 조합 정의
└── tests/                        # 단위/통합 테스트
```
