"""classify_transactions — 거래 텍스트 자동 분류 (LLM + 규칙 폴백)
================================================================
역할: "H&B 스토어 29,600원" 같은 텍스트를
      카테고리 / 필수·재량 / 월정기 여부로 판정.

구조:
- classify_llm()  : Claude Haiku 배치 분류 (메인 경로)
- classify_rule() : 키워드 규칙 폴백 (API 장애 시에도 서비스 유지)
- classify()      : LLM 시도 → 실패 시 규칙 폴백 (MVP 안정성 요건 대응)
- evaluate()      : 정답 라벨이 있는 CSV로 정확도 측정

실행 예:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 classifier.py persona_sumin.csv        # LLM 평가
    python3 classifier.py persona_sumin.csv --rule # 규칙 폴백만 평가
"""
__version__ = "classifier-v10"

import json
import os
import sys
import pandas as pd

CATEGORIES = ["식비", "카페", "뷰티", "패션", "교통", "주거", "통신",
              "구독", "여가", "의료", "교육", "환불", "미상", "기타"]

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = f"""당신은 개인 지출 내역 분류기입니다. 각 거래를 다음 기준으로 분류하세요.

카테고리 (반드시 이 중 하나): {", ".join(CATEGORIES)}

카테고리 세부 기준:
- 뷰티: 스킨케어, 메이크업, 헤어·바디 케어, 뷰티 소도구(고데기·퍼프 등),
  피부관리샵·에스테틱(예약금 포함), 향수·홈프래그런스
- 의료: 병원, 약국, 건강검진만 해당 (피부관리샵은 의료가 아니라 뷰티)
- 식비: 식사, 간식, 장보기, 외식 모두 포함

필수/재량 판정 (is_discretionary):
- 필수(false): 기본 식사·간편식·장보기, 대중교통 정기권, 주거, 통신, 구독료, 의료
- 재량(true): 카페·음료, 뷰티, 패션, 택시, 여가, 미상 지출
- 식비 판정: "식사"라고만 표기된 일상 식사는 필수. 주점·술·레스토랑·배달이
  명시적으로 드러난 경우에만 재량(외식)으로 판정

월정기 판정 (is_recurring):
- true: 월 1회 청구되는 지출 — 월세, 통신요금, 구독 서비스, 30일 교통 정기권, 보험료
- false: 그 외 전부

환불(음수 금액)은 카테고리 "환불", is_discretionary true, is_recurring false.

입력은 JSON 배열이며, 반드시 같은 길이·같은 순서의 JSON 배열만 출력하세요.
각 원소: {{"category": "...", "is_discretionary": true/false, "is_recurring": true/false}}
JSON 외 텍스트, 마크다운 백틱, 설명을 절대 포함하지 마세요."""


# ---------------------------------------------------------------
# 규칙 기반 폴백 (오프라인 테스트 + API 장애 대비)
# ---------------------------------------------------------------
_RULES = [  # (키워드 목록, 카테고리, 재량, 월정기) — 위에서부터 우선 적용
    (["환불", "반품"],                          "환불", True,  False),
    (["정기권", "월세", "관리비"],               None,   False, True),
    (["구독", "스트리밍", "저장공간", "멤버십"],   "구독", False, True),
    (["통신", "요금제", "알뜰폰"],               "통신", False, True),
    (["카페", "라떼", "커피", "콜드브루", "티 ", "홍차", "디저트"],
                                               "카페", True,  False),
    (["외식", "주점", "레스토랑", "배달"],        "식비", True,  False),
    (["스킨", "메이크업", "쿠션", "립", "블러셔", "헤어", "바디", "뷰티",
      "피부관리", "고데기", "퍼프", "팩트", "앰플", "프래그런스", "샴푸",
      "슬리밍"],                                "뷰티", True,  False),
    (["의류", "패션", "속옷", "잡화", "샌들", "가디건"],
                                               "패션", True,  False),
    (["택시"],                                  "교통", True,  False),
    (["교통", "지하철", "버스"],                 "교통", False, False),
    (["식사", "간식", "샌드위치", "쿠키", "간편식", "단백질", "에너지바",
      "닭가슴살", "베이커리", "우유"],            "식비", False, False),
    (["미상"],                                  "미상", True,  False),
]

def classify_rule(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        desc, amt = it["description"], it.get("amount", 0)
        result = {"category": "기타", "is_discretionary": True,
                  "is_recurring": False}
        for keywords, cat, disc, rec in _RULES:
            if any(k in desc for k in keywords):
                result = {"category": cat or _infer_cat(desc),
                          "is_discretionary": disc, "is_recurring": rec}
                break
        if amt < 0:
            result = {"category": "환불", "is_discretionary": True,
                      "is_recurring": False}
        out.append(result)
    return out

def _infer_cat(desc: str) -> str:
    if "정기권" in desc or "교통" in desc:
        return "교통"
    if "월세" in desc or "관리비" in desc:
        return "주거"
    return "기타"


# ---------------------------------------------------------------
# LLM 분류 (배치 1회 호출)
# ---------------------------------------------------------------
def classify_llm(items: list[dict], api_key: str | None = None) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
    payload = json.dumps(
        [{"description": i["description"], "amount": i.get("amount")} for i in items],
        ensure_ascii=False,
    )
    msg = client.messages.create(
        model=MODEL, max_tokens=4000, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    text = msg.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)

    # 출력 검증 — 길이·필드·값 도메인 체크. 하나라도 어긋나면 예외 → 폴백
    assert len(parsed) == len(items), f"길이 불일치 {len(parsed)} != {len(items)}"
    for r in parsed:
        assert r["category"] in CATEGORIES, f"미정의 카테고리 {r['category']}"
        assert isinstance(r["is_discretionary"], bool)
        assert isinstance(r["is_recurring"], bool)
    return parsed


def classify(items: list[dict], api_key: str | None = None) -> tuple[list[dict], str]:
    """메인 진입점. LLM 시도 → 실패 시 규칙 폴백. (결과, 사용된 경로) 반환."""
    try:
        return classify_llm(items, api_key), "llm"
    except Exception as e:
        print(f"[경고] LLM 분류 실패 → 규칙 폴백 사용: {e}", file=sys.stderr)
        return classify_rule(items), "rule"


# ---------------------------------------------------------------
# 평가 하네스
# ---------------------------------------------------------------
def evaluate(csv_path: str, use_rule_only: bool = False) -> None:
    df = pd.read_csv(csv_path)
    items = df[["description", "amount"]].to_dict("records")

    if use_rule_only:
        preds, path = classify_rule(items), "rule"
    else:
        preds, path = classify(items)

    n = len(df)
    cat_ok = disc_ok = rec_ok = all_ok = 0
    errors = []
    for i, (_, row) in enumerate(df.iterrows()):
        p = preds[i]
        c = p["category"] == row["category"]
        d = p["is_discretionary"] == bool(row["is_discretionary"])
        r = p["is_recurring"] == bool(row["is_recurring"])
        cat_ok += c; disc_ok += d; rec_ok += r; all_ok += (c and d and r)
        if not (c and d and r):
            errors.append(f"  '{row['description']}' 정답=({row['category']},"
                          f"재량={row['is_discretionary']},정기={row['is_recurring']}) "
                          f"예측=({p['category']},재량={p['is_discretionary']},"
                          f"정기={p['is_recurring']})")

    print(f"분류 경로: {path} / 평가 건수: {n}")
    print(f"카테고리 정확도 : {cat_ok/n*100:5.1f}% ({cat_ok}/{n})")
    print(f"필수·재량 정확도: {disc_ok/n*100:5.1f}% ({disc_ok}/{n})")
    print(f"월정기 정확도   : {rec_ok/n*100:5.1f}% ({rec_ok}/{n})")
    print(f"완전 일치      : {all_ok/n*100:5.1f}% ({all_ok}/{n})")
    if errors:
        print("\n오분류 내역:")
        print("\n".join(errors))


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "persona_sumin.csv"
    evaluate(csv, use_rule_only="--rule" in sys.argv)
