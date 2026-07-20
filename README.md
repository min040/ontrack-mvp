# 궤도 ON-TRACK — 실행 가이드

청년 자산 형성 AI 코칭 MVP. 소비 데이터를 목표 궤도 지수(GTI)로 환산하고,
AI 상담사가 재무 결정을 숫자로 답합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 앱 (온보딩 + 대시보드 + AI 상담) |
| `gti_engine.py` | 확정적 계산 엔진 (tool 5개) |
| `classifier.py` | LLM 거래 분류기 + 규칙 폴백 |
| `persona_sumin.csv` | 시연용 샘플 페르소나 (실사용 데이터 익명화본) |
| `requirements.txt` | 의존성 |
| `.streamlit/config.toml` | 테마 |

## 로컬 실행 (Windows)

```
cd 프로젝트폴더
pip install -r requirements.txt
set ANTHROPIC_API_KEY=본인키
streamlit run app.py
```

브라우저가 자동으로 열립니다 (http://localhost:8501).
API 키가 없어도 대시보드는 동작하고, AI 상담 탭만 비활성화됩니다.

## 배포 (Streamlit Community Cloud)

1. GitHub에 이 폴더를 저장소로 올리기 — **주의: API 키는 절대 커밋 금지.**
   `.gitignore`에 `.streamlit/secrets.toml` 포함할 것.
2. share.streamlit.io 접속 → GitHub 연동 → 저장소 선택 → `app.py` 지정.
3. 앱 설정(Settings) → Secrets에 아래 한 줄 추가:
   ```
   ANTHROPIC_API_KEY = "본인키"
   ```
4. 배포 완료 후 나오는 URL이 대회 제출용 웹서비스 URL.

## 심사 기간 접속 유지 (결격 방지 필수)

Streamlit Cloud는 트래픽이 없으면 앱을 재웁니다. 심사 기간(9/7 11:00 ~
9/11 23:59) 동안 다음 조치 필수:

1. cron-job.org 무료 계정 생성
2. 배포 URL을 대상으로 10분 간격 GET 요청 잡 등록
3. 9/6에 미리 켜서 정상 동작 확인

## 비용 참고

- 채팅 1턴당 Sonnet 기준 약 $0.01~0.03 수준 (tool 호출 횟수에 따라 변동)
- 심사 기간 전체 트래픽을 넉넉히 잡아도 $5 크레딧으로 충분
- 크레딧 소진 시 채팅이 멈추므로 9/6에 잔액 확인 권장
