"""flex_ingest — 임의 형식 가계부 CSV를 서비스 표준 형식으로 자동 변환
================================================================
표준 형식: date(YYYY-MM-DD), description(str), amount(int, 지출 양수)

3단 인식 구조:
1. 인코딩 자동 감지 (utf-8 / utf-8-sig / cp949·euc-kr)
2. 휴리스틱 컬럼 매핑
   - 날짜: 날짜로 파싱되는 비율이 가장 높은 컬럼
   - 금액: 숫자화 가능한 컬럼 (쉼표·'원' 제거). '출금/입금' 분리형이면
     출금 − 입금으로 상계 (입금은 환불로 음수 처리)
   - 내역: 남은 컬럼 중 텍스트 다양성이 가장 높은 컬럼
3. 휴리스틱 실패 시 AI(Haiku)가 헤더+샘플 행을 보고 매핑 (temperature 0)
"""
import io
import json
import re

import pandas as pd

DATE_HINTS = ["date", "날짜", "거래일", "일시", "이용일", "사용일"]
AMT_OUT_HINTS = ["출금", "지출", "사용금액", "이용금액", "결제금액", "amount",
                 "금액", "원화"]
AMT_IN_HINTS = ["입금", "환불", "취소"]
DESC_HINTS = ["내용", "적요", "내역", "가맹점", "상호", "거래처", "메모",
              "description", "사용처", "이용하신곳"]


def _read_any_encoding(raw: bytes) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last = e
    raise ValueError(f"CSV를 읽을 수 없어요 (인코딩 인식 실패: {last})")


def _to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[,원\s₩+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _date_score(series: pd.Series) -> float:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.notna().mean()


def _pick_by_hint(cols: list[str], hints: list[str]) -> str | None:
    for h in hints:
        for c in cols:
            if h in str(c).lower():
                return c
    return None


def _heuristic_map(df: pd.DataFrame) -> dict | None:
    cols = list(df.columns)
    # 날짜
    date_col = _pick_by_hint(cols, DATE_HINTS)
    if date_col is None or _date_score(df[date_col]) < 0.7:
        scored = sorted(cols, key=lambda c: _date_score(df[c]), reverse=True)
        date_col = scored[0] if _date_score(df[scored[0]]) >= 0.7 else None
    if date_col is None:
        return None
    rest = [c for c in cols if c != date_col]
    # 금액 (출금/입금 분리형 우선)
    out_col = _pick_by_hint(rest, AMT_OUT_HINTS)
    in_col = _pick_by_hint([c for c in rest if c != out_col], AMT_IN_HINTS)
    if out_col is None:
        numeric = [c for c in rest if _to_number(df[c]).notna().mean() > 0.7]
        out_col = numeric[0] if numeric else None
    if out_col is None:
        return None
    rest = [c for c in rest if c not in (out_col, in_col)]
    # 내역: 힌트 우선, 없으면 텍스트 다양성 최대 컬럼
    desc_col = _pick_by_hint(rest, DESC_HINTS)
    if desc_col is None and rest:
        desc_col = max(rest, key=lambda c: df[c].astype(str).nunique())
    if desc_col is None:
        return None
    return {"date": date_col, "description": desc_col,
            "amount_out": out_col, "amount_in": in_col}


def _ai_map(df: pd.DataFrame, api_key: str) -> dict | None:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    sample = df.head(3).to_csv(index=False)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=300, temperature=0,
        system=("가계부 CSV의 컬럼 역할을 판별해 JSON만 출력: "
                '{"date": 날짜컬럼명, "description": 내역컬럼명, '
                '"amount_out": 지출금액컬럼명, "amount_in": 입금컬럼명또는null}'
                " 다른 텍스트 금지."),
        messages=[{"role": "user", "content": sample}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    lb, rb = text.find("{"), text.rfind("}")
    m = json.loads(text[lb:rb + 1])
    for k in ("date", "description", "amount_out"):
        if m.get(k) not in df.columns:
            return None
    if m.get("amount_in") not in df.columns:
        m["amount_in"] = None
    return m


def ingest(raw: bytes, api_key: str | None = None) -> tuple[pd.DataFrame, str]:
    """임의 CSV 바이트 → 표준 DataFrame. (df, 사용된 인식 경로) 반환."""
    df = _read_any_encoding(raw)
    df.columns = [str(c).strip() for c in df.columns]

    # 이미 표준 형식이면 그대로
    if {"date", "description", "amount"}.issubset(df.columns):
        out = df.copy()
        out["amount"] = _to_number(out["amount"]).fillna(0).astype(int)
        return out, "standard"

    mapping, path = _heuristic_map(df), "heuristic"
    if mapping is None and api_key:
        try:
            mapping, path = _ai_map(df, api_key), "ai"
        except Exception:
            mapping = None
    if mapping is None:
        raise ValueError(
            "컬럼 구조를 인식하지 못했어요. 날짜·내역·금액 컬럼이 포함된 "
            "CSV인지 확인해 주세요.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce",
                                 format="mixed").dt.strftime("%Y-%m-%d")
    out["description"] = df[mapping["description"]].astype(str).str.strip()
    amt = _to_number(df[mapping["amount_out"]]).fillna(0)
    if mapping.get("amount_in"):
        amt = amt - _to_number(df[mapping["amount_in"]]).fillna(0)
    out["amount"] = amt.round().astype(int)

    out = out[out["date"].notna() & (out["amount"] != 0)
              & (out["description"] != "")].reset_index(drop=True)
    if out.empty:
        raise ValueError("변환 결과가 비어 있어요. 파일 내용을 확인해 주세요.")
    return out, path
