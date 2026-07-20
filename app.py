"""궤도(ON-TRACK) v3 — 청년 자산 형성 AI 코칭 MVP
====================================================
v3 변경 (2차 실사용자 테스트 피드백 반영):
- 컨셉 색 선택 → UI 전체 자동 테마 적용
- 지출 차트: 금액 라벨 표시, 필수/재량 혼합 카테고리 설명, 분류 근거 열람
- 목표 설정: 입력→계산 관계를 명시한 UI, 여력 음수여도 잠재 페이스로 계산
- 절감 플래너: 단축 효과(일) 표시, AI 품목 추천
- 이벤트 예산: 항목별 개별 절감률(슬라이더+직접 입력), 항목별 절감액, AI 품목 추천
- 가속 추천 → 절감 순위 전체 공개: 순위별 이유·최대 한도·비율 조절·단축 일수·AI 품목 추천
"""
__version__ = "app-v10"

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from classifier import classify
from flex_ingest import ingest as flex_ingest
from gti_engine import (CUT_CAPS, DEFAULT_CAP, UserProfile, calc_gti,
                        cut_rankings, detect_anomalies, event_budget,
                        get_spending_summary, plan_cuts, purchase_impact,
                        required_cut, simulate_scenario,
                        smart_cut_allocation, solve_goal, suggest_items,
                        uniform_cut_for_track)

st.set_page_config(page_title="궤도 ON-TRACK", page_icon="🛰️", layout="wide")

CHAT_MODEL = "claude-sonnet-4-6"
ITEM_MODEL = "claude-haiku-4-5-20251001"
ON_TRACK_GTI = round(100 / 1.2, 1)
NEUTRAL = "#8A8887"


def get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def won(x) -> str:
    return f"{round(x):,}원"


# ---------- 테마 색 유틸 ----------
def _shade(hex_color: str, factor: float) -> str:
    """factor>0 밝게, <0 어둡게."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def inject_theme(accent: str) -> None:
    """선택한 컨셉 색으로 UI 전체 테마 자동 적용 + 타이포 다양화."""
    soft = _shade(accent, 0.88)
    dark = _shade(accent, -0.30)
    st.markdown(f"""<style>
      h1 {{ color: {dark}; font-weight: 800; }}
      h2, h3 {{ color: {dark}; font-weight: 700; }}
      [data-testid="stMetricValue"] {{ color: {dark}; font-weight: 800; }}
      [data-testid="stMetricLabel"] {{ color: {NEUTRAL}; font-size: 0.85rem; }}
      [data-testid="stSidebar"] {{ background: {soft}; }}
      .stTabs [aria-selected="true"] {{ color: {dark} !important;
        border-bottom-color: {accent} !important; font-weight: 700; }}
      .stButton > button {{ border-color: {accent}; color: {dark}; }}
      .stButton > button:hover {{ background: {soft}; border-color: {dark}; }}
      div[data-testid="stSlider"] [role="slider"] {{ background: {accent}; }}
      .big-note {{ font-size: 1.05rem; color: {dark}; font-weight: 600; }}
      .sub-note {{ font-size: 0.85rem; color: {NEUTRAL}; }}
      .pill {{ display:inline-block; padding:2px 10px; border-radius:12px;
        background:{soft}; color:{dark}; font-size:0.82rem; font-weight:700;
        margin:1px; }}
      .pill.out {{ background:{dark}; color:#fff; }}
      .mode-arrow {{ color:{dark}; font-weight:800; margin:0 4px; }}
      span[data-baseweb="tag"] {{ background:{soft} !important;
        color:{dark} !important; }}
      span[data-baseweb="tag"] svg {{ fill:{dark} !important; }}
    </style>""", unsafe_allow_html=True)


def pct_control(label: str, cap_pct: int, default_pct: int, key: str) -> int:
    """슬라이더 + 직접 입력이 동기화된 퍼센트 컨트롤 (0 ~ 한도)."""
    sl, num = f"sl_{key}", f"num_{key}"
    if sl not in st.session_state:
        st.session_state[sl] = min(default_pct, cap_pct)
        st.session_state[num] = min(default_pct, cap_pct)
    # 한도 토글로 cap이 줄어든 경우 기존 값을 한도 안으로 클램프
    if st.session_state[sl] > cap_pct:
        st.session_state[sl] = cap_pct
        st.session_state[num] = cap_pct

    def _from_sl():
        st.session_state[num] = st.session_state[sl]

    def _from_num():
        st.session_state[sl] = st.session_state[num]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(label, 0, cap_pct, key=sl, on_change=_from_sl, format="%d%%")
    with c2:
        st.number_input("직접 입력(%)", 0, cap_pct, key=num,
                        on_change=_from_num, label_visibility="collapsed")
    return st.session_state[sl]


def monthly_scale_factor() -> float:
    """관측 기간 → 30일 환산 배율. 품목 플랜의 금액 단위를 대시보드와
    동일한 '월 환산'으로 통일하기 위함 (단위 불일치 버그 수정)."""
    dates = pd.to_datetime(tx["date"])
    days = (dates.max() - dates.min()).days + 1
    return 30.0 / max(days, 1)


def fmt_months(m: float) -> str:
    """1개월 미만은 일 단위로 표시 (며칠짜리 목표가 0.1개월로 뭉개지던 문제)."""
    if m is None:
        return "계산 불가"
    if m < 1:
        return f"약 {max(round(m * 30), 1)}일"
    return f"약 {round(m, 1)}개월"


VERDICT_META = {
    "hold": ("🛑", "이번 달 보류"),
    "reduce": ("⏸", "횟수·빈도 줄이기"),
    "substitute": ("🔁", "대안으로 대체"),
    "keep": ("✅", "유지"),
}


def _get_item_verdicts(category: str, target: int, items: list[dict],
                       min_secured: int | None = None) -> list[dict]:
    """AI가 각 품목의 처분을 JSON으로 판정. 품목은 번호(id)로만 지목하게
    해서 이름·금액 왜곡을 차단. 실패 시 확정적 폴백(큰 금액 순 보류).
    min_secured: 추가 절감 플랜에서 '직전 플랜보다 반드시 더 확보' 강제."""
    key = f"vrd_{category}_{target}_{min_secured or 0}"
    if key in st.session_state:
        return st.session_state[key]

    def fallback() -> list[dict]:
        out, acc = [], 0
        order = sorted(range(len(items)), key=lambda j: -items[j]["amount"])
        for idx in order:
            if acc >= target:
                break
            out.append({"item_ids": [idx], "verdict": "hold", "ratio": 1.0,
                        "action": "이번 달 구매 보류"})
            acc += items[idx]["amount"]
        return out

    api_key = get_api_key()
    verdicts = None
    last_err = None
    for _attempt in range(2 if api_key else 0):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            items_txt = "\n".join(
                f"{i}. {it['description']} — {it['amount']}원"
                for i, it in enumerate(items))
            corrective = ("" if _attempt == 0 else
                          "\n\n[중요] 직전 응답이 JSON 파싱에 실패했습니다. "
                          "이번에는 반드시 '['로 시작해 ']'로 끝나는 JSON "
                          "배열만, 다른 문자 없이 출력하세요.")
            pool = sum(it["amount"] for it in items)
            effective = min(target, pool)
            msg = client.messages.create(
                model=ITEM_MODEL, max_tokens=1600, temperature=0,
                system=(
                    "당신은 지출 코칭 판정기입니다. JSON 배열만 출력하세요. "
                    "마크다운 백틱, 설명, 다른 텍스트 금지.\n"
                    "각 원소: {\"item_ids\": [번호들], \"verdict\": "
                    "\"hold|reduce|keep\", \"ratio\": 0~1 숫자, "
                    "\"action\": \"15자 이내 행동 문구\"}\n"
                    "규칙:\n"
                    "1. 모든 품목 번호를 정확히 한 번씩 어느 그룹에든 배정\n"
                    "2. hold(대형 단발 구매 보류)는 ratio 1.0 / "
                    "reduce(습관성 반복의 횟수·양 축소)는 줄어드는 비율 / "
                    "keep(꼭 필요)은 ratio 0. 특정 대체품·대체처를 "
                    "제안하는 판정은 하지 말 것\n"
                    "3. 확보 합계(품목 금액 합 × ratio)가 목표 절감액의 "
                    "90~115%가 되도록 판정을 구성\n"
                    "4. 비슷한 품목은 한 그룹으로 묶기 (그룹 4~6개 이내)\n"
                    "5. action에 '40% 감소' 같은 퍼센트·비율 표현 금지. "
                    "구체적 행동만: hold는 '다음 달로 미루기'류, reduce는 "
                    "무엇을 덜 하는지(예: '주말에만 방문')를 명시\n"
                    "6. 확보 합계가 목표에 못 미칠 것 같으면 keep을 "
                    "최소화해서라도 목표에 최대한 근접시키기"),
                messages=[{"role": "user", "content":
                           f"카테고리: {category}\n"
                           f"월 절감 목표: {target}원\n"
                           f"품목 총액: {pool}원 → 달성 가능한 최대 확보액도 "
                           f"{pool}원입니다. 확보 합계를 "
                           f"{round(effective*0.95)}원 이상으로 최대한 "
                           f"끌어올리세요.\n"
                           + (f"중요: 직전 완화 플랜의 확보 합계는 "
                              f"{min_secured}원이었습니다. 이번 플랜의 확보 "
                              f"합계는 반드시 이보다 커야 합니다 — keep을 "
                              f"줄이고, reduce의 ratio를 높이고, hold를 "
                              f"늘리세요.\n" if min_secured else "")
                           + f"품목:\n{items_txt}" + corrective}])
            if msg.stop_reason == "max_tokens":
                raise ValueError("출력이 토큰 한도에서 잘림")
            text = "".join(b.text for b in msg.content if b.type == "text")
            # 관대한 추출: 첫 '['부터 마지막 ']'까지만 파싱
            lb, rb = text.find("["), text.rfind("]")
            if lb == -1 or rb == -1:
                raise ValueError("응답에 JSON 배열이 없음")
            parsed = json.loads(text[lb:rb + 1])
            assert isinstance(parsed, list) and parsed
            for g in parsed:
                if g.get("verdict") == "substitute":  # 방어: 대체 판정 흡수
                    g["verdict"] = "reduce"
                assert g["verdict"] in VERDICT_META
                g["item_ids"] = [int(i) for i in g["item_ids"]]
                assert all(0 <= i < len(items) for i in g["item_ids"])
                g["ratio"] = min(max(float(g.get("ratio", 0)), 0.0), 1.0)
                if g["verdict"] == "hold":
                    g["ratio"] = 1.0
                if g["verdict"] == "keep":
                    g["ratio"] = 0.0
            if min_secured is not None:
                sec = sum(round(sum(items[i]["amount"] for i in g["item_ids"])
                                * g["ratio"]) for g in parsed)
                if sec <= min_secured:
                    raise ValueError(
                        f"확보액 {sec}원이 기본 플랜({min_secured}원) 이하")
            verdicts = parsed
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            verdicts = None
    used_fallback = verdicts is None
    if used_fallback and last_err:
        st.session_state[f"vrd_err_{category}_{target}"] = last_err
    if used_fallback:
        verdicts = fallback()
        for g in verdicts:
            g["_fallback"] = True
    st.session_state[key] = verdicts
    return verdicts


def render_item_plan(targets: dict[str, int],
                     min_secured: dict[str, int] | None = None) -> dict:
    """구조화된 절감 실행 플랜 렌더링 — 금액 큰 행동 순, 진행 바 포함.
    (판정은 AI, 금액 합산·정렬·진행률은 전부 코드가 확정 계산)
    반환: {카테고리: 확보액} — 추가 절감 플랜의 에스컬레이션 기준."""
    secured_map = {}
    for category, target in targets.items():
        base = suggest_items(tx, category, max(target, 1))
        scale = monthly_scale_factor()
        items = [{**it, "amount": round(it["amount"] * scale)}
                 for it in base["all_items"]]
        if not items:
            st.info(f"{category}에 조절 가능한 구매 내역이 없어요.")
            continue
        pool_total = sum(it["amount"] for it in items)
        clamped = target > pool_total
        target = min(target, pool_total)  # 목표는 카테고리 월 지출을 넘을 수 없음
        ms = (min_secured or {}).get(category)
        if ms is not None and ms >= pool_total * 0.99:
            st.info(f"[{category}] 기본 플랜이 이미 이 카테고리에서 확보 "
                    f"가능한 최대치({won(pool_total)})예요 — 추가 절감은 "
                    f"다른 카테고리에서 찾아보세요.")
            secured_map[category] = ms
            continue
        verdicts = _get_item_verdicts(category, target, items,
                                      min_secured=ms)

        with st.expander("아이콘이 뭘 뜻하나요?"):
            st.markdown("🛑 **보류** — 이번 달엔 이 구매를 하지 않기  \n"
                        "⏸ **횟수 줄이기** — 반복 구매의 횟수를 낮추기 "
                        "(옆에 '관측 N회 → M회'로 표시)  \n"
                        "✅ **유지** — 지금처럼 써도 괜찮은 것")
        if any(g.get("_fallback") for g in verdicts):
            err = st.session_state.get(f"vrd_err_{category}_{target}", "")
            st.warning("AI 판정 연결이 잠시 안 돼서 '금액 큰 순 보류' 기본 "
                       "플랜을 보여드리고 있어요."
                       + (f" (원인: {err[:120]})" if err else ""))
            if st.button("🔄 AI 플랜 다시 생성",
                         key=f"regen_{category}_{target}"):
                st.session_state.pop(f"vrd_{category}_{target}", None)
                st.rerun()

        rows, secured = [], 0
        for g in verdicts:
            amt = round(sum(items[i]["amount"] for i in g["item_ids"])
                        * g["ratio"])
            names = " · ".join(items[i]["description"]
                               for i in g["item_ids"][:3])
            if len(g["item_ids"]) > 3:
                names += f" 외 {len(g['item_ids']) - 3}건"
            icon, label = VERDICT_META[g["verdict"]]
            verdict_txt = f"{icon} {g.get('action') or label}"
            if g["verdict"] == "reduce":
                # 퍼센트 대신 실제 구매 횟수로 번역 (코드가 데이터에서 계산)
                n = len(g["item_ids"])
                m = max(round(n * (1 - g["ratio"])), 0)
                verdict_txt += f" · 관측 {n}회 → {m}회"
            rows.append({"판정": verdict_txt,
                         "품목": names, "_amt": amt,
                         "확보 금액": won(amt) if amt else "—"})
            secured += amt
        act = sorted([r for r in rows if r["_amt"] > 0],
                     key=lambda r: -r["_amt"])[:5]

        pct = min(secured / target, 1.0) if target else 0
        st.markdown(f"**[{category}] 월 {won(target)} 만들기** — 아래 "
                    f"{len(act)}개 행동으로 목표의 **{pct*100:.0f}%** 확보")
        if clamped:
            st.caption(f"ℹ️ 요청한 목표가 이 카테고리 월 지출을 넘어서, "
                       f"확보 가능한 최대치({won(pool_total)})로 조정했어요.")
        table_rows = [{k: r[k] for k in ("판정", "품목", "확보 금액")}
                      for r in act]
        table_rows.append({"판정": "Σ 합계",
                           "품목": f"{len(act)}개 행동",
                           "확보 금액": won(secured)})
        st.dataframe(pd.DataFrame(table_rows), hide_index=True,
                     width='stretch')
        st.progress(pct, text=f"확보 {won(secured)} / 목표 {won(target)}")
        st.caption("🤖 AI 품목 추천은 목표 전액 커버를 보장하지 않아요 — "
                   "막대가 '이 플랜대로 하면 확보되는 비율'이에요.")

        # 유지 항목은 별도 표로 — '안 줄여도 되는 것'도 명확한 정보.
        # AI가 판정에서 빠뜨린 품목은 자동으로 '유지'로 분류해 항상 표시.
        keep_rows = []
        covered = {i for g in verdicts for i in g["item_ids"]}
        uncovered = [i for i in range(len(items)) if i not in covered]
        keep_groups = [g for g in verdicts
                       if g["ratio"] == 0 or g["verdict"] == "keep"]
        if uncovered:
            keep_groups.append({"item_ids": uncovered,
                                "action": "절감 대상에서 제외 — 유지"})
        for g in keep_groups:
            k_names = " · ".join(items[i]["description"]
                                 for i in g["item_ids"][:3])
            if len(g["item_ids"]) > 3:
                k_names += f" 외 {len(g['item_ids']) - 3}건"
            k_amt = sum(items[i]["amount"] for i in g["item_ids"])
            keep_rows.append({"품목": k_names,
                              "지출 (월 환산)": won(k_amt),
                              "이유": g.get("action") or "생활 유지 필요"})
        st.markdown("**✅ 그대로 둬도 되는 것**")
        if keep_rows:
            st.dataframe(pd.DataFrame(keep_rows), hide_index=True,
                         width='stretch')
        else:
            st.caption("이번 플랜은 목표 달성을 위해 모든 품목을 절감 "
                       "대상으로 판정했어요 — 유지 항목이 없습니다.")
        st.caption("금액은 대시보드와 같은 30일(월) 환산 기준이에요.")
        secured_map[category] = secured
    return secured_map


def rec_with_boost(targets: dict[str, int], key_prefix: str) -> None:
    """목표 딱 맞춤 플랜 + 슬라이더를 따라가는 추가 절감 플랜."""
    base_secured = render_item_plan(targets)
    st.markdown('<p class="sub-note">더 공격적으로 줄여보고 싶다면 아래에서 '
                '추가 비율을 정해보세요.</p>', unsafe_allow_html=True)
    boost = pct_control("목표 대비 추가 절감", 100, 0, f"{key_prefix}_boost")
    if boost > 0 and not st.session_state.get(f"{key_prefix}_boost_on"):
        if st.button("🤖 이 조건으로 추가 플랜 보기",
                     key=f"{key_prefix}_boost_btn"):
            st.session_state[f"{key_prefix}_boost_on"] = True
    if st.session_state.get(f"{key_prefix}_boost_on") and boost > 0:
        boosted = {c: round(t * (1 + boost / 100))
                   for c, t in targets.items()}
        st.markdown(f"---\n**➕ 추가 {boost}% 절감 플랜** (슬라이더를 "
                    f"움직이면 자동으로 갱신돼요)")
        render_item_plan(boosted, min_secured=base_secured)
    elif st.session_state.get(f"{key_prefix}_boost_on") and boost == 0:
        st.session_state[f"{key_prefix}_boost_on"] = False


# ================================================================
# 사이드바
# ================================================================
with st.sidebar:
    st.title("🛰️ 궤도 ON-TRACK")
    st.caption("자산 목표의 내비게이션 — 모든 소비를 '목표까지의 "
               "시간'으로 번역합니다")

    accent = st.color_picker("🎨 나만의 컨셉 색", "#0F6E56",
                             help="고른 색에 맞춰 화면 전체가 자동으로 꾸며져요.")
    # 내장 위젯(슬라이더·라디오·체크박스 등)은 Streamlit 테마 색을 따르므로
    # 테마 자체를 컨셉 색으로 교체. 색이 바뀐 직후엔 한 번 리프레시해
    # 모든 위젯에 즉시 반영.
    try:
        st._config.set_option("theme.primaryColor", accent)
    except Exception:
        pass
    if st.session_state.get("_accent") != accent:
        st.session_state["_accent"] = accent
        st.rerun()
    inject_theme(accent)

    st.subheader("1. 지출 데이터")
    if st.button("샘플 체험 (대학생 수민)", width='stretch'):
        st.session_state.tx = pd.read_csv("persona_sumin.csv")
        st.session_state.data_label = "샘플 페르소나 '수민' (21일)"
        st.session_state.profile_defaults = (870000, 3480000, 12, 0)
        st.session_state.messages = []

    uploaded = st.file_uploader(
        "내 CSV 업로드", type="csv",
        help="어떤 형식이든 OK — 토스·뱅크샐러드·카드사 내보내기 등 컬럼명이 "
             "달라도 자동으로 인식해요. (날짜·내역·금액 정보만 있으면 됩니다)")
    if uploaded is not None and st.session_state.get("uploaded_name") != uploaded.name:
        try:
            df, ingest_path = flex_ingest(uploaded.getvalue(), get_api_key())
            if ingest_path != "standard":
                st.info(f"컬럼 형식을 자동 인식해 변환했어요 "
                        f"({'AI 매핑' if ingest_path == 'ai' else '자동 매핑'}, "
                        f"{len(df)}건)")
        except ValueError as e:
            st.error(str(e))
            df = None
        if df is not None:
            if "category" not in df.columns or "is_discretionary" not in df.columns:
                with st.spinner("AI가 거래 내역을 분류하는 중..."):
                    items = df[["description", "amount"]].to_dict("records")
                    preds, path = classify(items, get_api_key())
                    df["category"] = [p["category"] for p in preds]
                    df["is_discretionary"] = [p["is_discretionary"] for p in preds]
                    df["is_recurring"] = [p["is_recurring"] for p in preds]
                    st.info(f"분류 완료 ({'AI' if path == 'llm' else '규칙 기반'})")
            st.session_state.tx = df
            st.session_state.data_label = f"업로드: {uploaded.name}"
            st.session_state.uploaded_name = uploaded.name
            st.session_state.messages = []

    st.subheader("2. 목표 설정")
    d = st.session_state.get("profile_defaults", (1000000, 4000000, 12, 0))
    st.markdown('<p class="sub-note">🔴 필수 입력 — 계산의 기준이 되는 '
                '값이에요.</p>', unsafe_allow_html=True)
    income = st.number_input("월 소득 (원) 🔴", 0, 100_000_000, d[0], 50000)
    assets = st.number_input("현재 자산 (원) 🔴", 0, 1_000_000_000, d[3],
                             100000)
    st.markdown('<p class="sub-note">⚪ 선택 입력 — 아래 3번 계산 모드에 '
                '따라 필요한 것만 쓰면 돼요.</p>', unsafe_allow_html=True)
    goal_raw = st.number_input("목표 금액 (원) ⚪", 0, 10**9, d[1], 100000)
    months_raw = st.number_input("목표 기한 (개월) ⚪", 1, 120, d[2])

    st.subheader("3. 계산 모드")
    MODE_BOTH, MODE_WHEN, MODE_HOW = "목표 점검", "기한 계산", "금액 계산"
    goal_mode = st.radio(
        "어떤 계산을 해드릴까요?", [MODE_BOTH, MODE_WHEN, MODE_HOW],
        captions=["금액과 기한을 쓸 테니 → 달성 가능한지 알려줘",
                  "금액만 쓸 테니 → 언제 모이는지 알려줘",
                  "기한만 쓸 테니 → 얼마 모을 수 있는지 알려줘"])

    if goal_mode == MODE_BOTH:
        goal_in, months_in = goal_raw, months_raw
        io_inputs = [("목표 금액", won(goal_raw)),
                     ("목표 기한", f"{months_raw}개월")]
    elif goal_mode == MODE_WHEN:
        goal_in, months_in = goal_raw, None
        io_inputs = [("목표 금액", won(goal_raw))]
    else:
        goal_in, months_in = None, months_raw
        io_inputs = [("목표 기한", f"{months_raw}개월")]

    # 세로 방향 입력란 → ⬇ → 출력란 다이어그램
    pills = " ".join(f'<span class="pill">{k}: {v}</span>'
                     for k, v in io_inputs)
    st.markdown(f'<p class="sub-note" style="margin-bottom:2px">✍️ 입력란'
                f'</p><p>{pills}</p>'
                f'<p style="text-align:center;margin:2px 0" '
                f'class="mode-arrow">⬇</p>'
                f'<p class="sub-note" style="margin-bottom:2px">🤖 출력란</p>',
                unsafe_allow_html=True)
    goal_output_slot = st.container()  # 계산 결과가 여기 채워짐

    with st.expander("'조절 가능한 지출을 전부 줄이면'이 무슨 뜻이에요?"):
        st.markdown(
            "카페·뷰티·패션·외식처럼 **재량**으로 분류된 지출을 전부 0원으로 "
            "만든 극단적 가정이에요. 이때의 저축 여력 = 소득 − 필수 지출(주거·"
            "통신·기본 식비·정기권 등)로, 이론상 낼 수 있는 **최대 저축 "
            "속도**죠. 현실적인 목표가 아니라 '상한선'을 보여주는 기준값이고, "
            "실제 플랜은 🧮 플래너에서 항목별 한도 안에서 설계하는 걸 추천해요.")

# ================================================================
# 랜딩
# ================================================================
if "tx" not in st.session_state:
    with goal_output_slot:
        st.info("지출 데이터를 불러오면 계산 결과가 여기 표시돼요.")
    st.title("궤도는 당신의 모든 소비를 '목표까지의 시간'으로 번역합니다")
    st.markdown(
        "가계부는 **과거를 기록**해요 — \"지난달 카페에 8만원 썼어요\"까지. "
        "그 소비가 내 목표와 무슨 관계인지는 말해주지 않죠.\n\n"
        "궤도는 **자산 목표의 내비게이션**입니다. 내비가 \"이 길로 가면 "
        "도착이 10분 늦어져요\"라고 알려주듯, 궤도는 모든 재무 결정을 "
        "시간으로 답해요:\n"
        "- 🏁 **도착 예정** — 이 페이스면 목표에 언제 도달하나\n"
        "- 🧭 **경로 이탈 감지** — 이 소비가 목표를 며칠 미뤘나\n"
        "- 🔁 **경로 재탐색** — 어디서 줄이면 며칠을 되찾나\n\n"
        "👈 왼쪽에서 **샘플 체험**을 누르거나, 쓰던 가계부 CSV를 그대로 "
        "올려 시작하세요.")
    st.stop()

tx = st.session_state.tx

# ---- 목표 계산 (여력 음수여도 잠재 페이스로 계산) ----
_probe = calc_gti(UserProfile(income, assets, 10**9, 120), tx)
capacity = _probe["monthly_savings_capacity"]
potential = income - _probe["monthly_fixed_expense"]  # 재량 전액 절감 시 여력

if goal_in and months_in:
    goal, months = goal_in, months_in
    sv = solve_goal(capacity, assets, goal_amount=goal_in,
                    goal_months=months_in)
    with goal_output_slot:
        if sv["feasible"]:
            st.success(f"**달성 가능!** 매달 {won(sv['required_monthly'])}씩 "
                       f"모으면 되고, 현재 여력({won(capacity)})으로 "
                       f"충분해요.")
        else:
            st.warning(f"달성하려면 매달 **{won(sv['required_monthly'])}** "
                       f"저축이 필요한데, 현재 여력은 {won(capacity)}이에요. "
                       f"🧮 플래너에서 부족분을 채울 플랜을 만들어보세요.")
elif goal_in:
    sv = solve_goal(capacity, assets, goal_amount=goal_in,
                    potential_capacity=potential)
    with goal_output_slot:
        if sv["months"] is not None:
            months = max(int(-(-sv["months"] // 1)), 1)
            st.success(f"도달까지 걸리는 기간: **{fmt_months(sv['months'])}** "
                       f"(기한을 {months}개월로 설정했어요)")
        elif sv["potential_months"] is not None:
            months = max(int(-(-sv["potential_months"] // 1)), 1)
            st.warning(
                f"현재 페이스(여력 {won(capacity)})로는 도달할 수 없어요. "
                f"조절 가능한 지출을 전부 줄이면 최소 "
                f"**{fmt_months(sv['potential_months'])}** — 기한을 "
                f"{months}개월로 잡았으니 플래너에서 현실적인 플랜을 "
                f"만들어보세요.")
        else:
            months = 12
            st.error("필수 지출만으로 소득이 넘어서 기한 계산이 불가해요.")
    goal = goal_in
else:
    sv = solve_goal(capacity, assets, goal_months=months_in,
                    potential_capacity=potential)
    with goal_output_slot:
        if sv["amount"] > assets:
            goal = sv["amount"]
            st.success(f"{months_in}개월 뒤 모을 수 있는 금액: "
                       f"**{won(sv['amount'])}**")
        elif sv["potential_amount"]:
            goal = sv["potential_amount"]
            st.warning(
                f"현재 페이스로는 모이는 돈이 없어요. 조절 가능한 지출을 "
                f"전부 줄이면 {months_in}개월 뒤 최대 "
                f"**{won(sv['potential_amount'])}**까지 가능해요. 이 "
                f"최대치를 목표로 잡았으니 플래너에서 얼마나 줄일지 "
                f"정해보세요.")
        else:
            goal = 1
            st.error("필수 지출만으로 소득이 넘어서 계산이 불가해요.")
    months = months_in

profile = UserProfile(monthly_income=income, current_assets=assets,
                      goal_amount=max(goal, 1), goal_months=months)

st.caption(f"데이터: {st.session_state.data_label}")
tab_dash, tab_plan, tab_chat = st.tabs(
    ["🧭 진단 (대시보드)", "🔁 처방 (플래너)", "💬 상담 (AI)"])

# ================================================================
# 대시보드
# ================================================================
with tab_dash:
    st.markdown('<p class="sub-note">🧭 <b>진단</b> — 지금의 소비가 목표 '
                '도착 시간에 미치는 영향을 봅니다.</p>',
                unsafe_allow_html=True)
    r = calc_gti(profile, tx)

    if r["monthly_savings_capacity"] <= 0:
        st.error(f"### 지금은 버는 돈보다 쓰는 돈이 많아요\n매달 "
                 f"**{won(-r['monthly_savings_capacity'])}**씩 마이너스예요. "
                 f"저축을 시작하려면 지출 구조부터 바꿔야 해요 → 🧮 플래너 탭")
    elif r["track_ratio"] < 1.0:
        st.warning(f"### 이 페이스면 목표보다 약 {r['delay_months']}개월 늦어요\n"
                   f"매달 {won(r['monthly_required_savings'])}씩 모아야 하는데, "
                   f"지금 여력은 {won(r['monthly_savings_capacity'])}이에요.")
    else:
        early = round(-r["delay_months"], 1) if r["delay_months"] else 0
        st.success(f"### 궤도 위예요! 목표보다 약 {early}개월 빠른 페이스\n"
                   f"이대로면 {r['eta_months']}개월 뒤 목표에 도달해요.")

    anomalies = detect_anomalies(tx)
    if anomalies:
        with st.expander(f"📌 데이터 특이사항 {len(anomalies)}건 — 숫자 해석 "
                         f"전에 확인하세요", expanded=False):
            st.markdown('<p class="sub-note">가계부를 자동 분석해 월 환산이나 '
                        '코칭 해석에 영향을 줄 수 있는 패턴을 찾았어요. 같은 '
                        '데이터면 항상 같은 내용이 표시돼요.</p>',
                        unsafe_allow_html=True)
            for n in anomalies:
                st.markdown(f"- **{n['type']}** · {n['note']}")

    # ---- 히어로 지표: 내비게이션 언어(시간·행동)로 통일 ----
    rc = required_cut(profile, tx)
    m1, m2, m3 = st.columns(3)
    m1.metric("🏁 도착 예정",
              f"{r['eta_months']}개월 뒤" if r["eta_months"]
              else "도달 불가",
              help="지금 소비 페이스가 유지될 때 목표에 도달하는 시점")
    if r["monthly_savings_capacity"] <= 0:
        route_val, route_help = "저축 정지", "지출이 소득을 넘어 경로를 이탈했어요"
    elif r["track_ratio"] < 1.0:
        route_val = f"{r['delay_months']}개월 늦음"
        route_help = "목표 기한 대비 예상 지연"
    else:
        early = round(-r["delay_months"], 1) if r["delay_months"] else 0
        route_val, route_help = f"{early}개월 빠름", "목표 기한보다 앞선 페이스"
    m2.metric("🧭 경로 상태", route_val, help=route_help)
    if r["track_ratio"] >= 1.0:
        m3.metric("🔁 경로 재탐색", "필요 없음", help="이미 궤도 위예요")
    elif rc.get("feasible") and rc.get("cut_ratio") is not None:
        m3.metric("🔁 경로 재탐색", f"월 {won(rc['monthly_cut_amount'])} 절감",
                  help="목표 기한을 지키기 위해 되찾아야 하는 금액 — "
                       "🔁 처방 탭에서 설계")
    else:
        m3.metric("🔁 경로 재탐색", "목표 조정 필요",
                  help="지출 절감만으로는 어려워요 — 상담 탭에서 대안을 "
                       "논의해보세요")
    st.caption(f"참고 수치 — 월 저축 여력 {won(r['monthly_savings_capacity'])} "
               f"/ 월 요구 저축액 {won(r['monthly_required_savings'])}")

    st.divider()
    # ---- GTI: 위 상태를 압축한 '경로 준수율' 보조 지표 ----
    g1, g2 = st.columns([1, 1.5])
    with g1:
        g_color = (accent if r["gti"] >= ON_TRACK_GTI
                   else "#EF9F27" if r["gti"] >= 50 else "#E24B4A")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["gti"],
            title={"text": "GTI (경로 준수율)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": g_color},
                   "threshold": {"line": {"color": _shade(accent, -0.3),
                                          "width": 3},
                                 "value": ON_TRACK_GTI}}))
        fig.update_layout(height=240, margin=dict(t=55, b=10, l=30, r=30))
        st.plotly_chart(fig, width='stretch')
    with g2:
        st.markdown(
            '<p class="big-note">GTI는 위의 도착 예정·경로 상태·소비 안정성을 '
            '0~100 하나로 압축한 <b>경로 준수율</b>이에요 — 내비의 보조 '
            '계기판처럼, 매일의 변화를 점수 하나로 추적할 때 쓰세요.</p>',
            unsafe_allow_html=True)
        with st.expander("GTI 계산 방식이 궁금해요"):
            st.markdown(
                f"- **{ON_TRACK_GTI}점(굵은 선)** = 딱 목표 기한에 맞는 "
                f"페이스 (궤도 진입선)\n- 그 위 = 목표보다 빨리 도달 / "
                f"그 아래 = 이대로면 늦어요\n- 소비가 들쭉날쭉하면 계획을 "
                f"지키기 어렵다고 보고 점수를 살짝 깎아요. 지금 변동성 감점: "
                f"**-{round(r['gti_before_penalty'] - r['gti'], 1)}점**")

    st.divider()
    st.subheader("줄일 수 있는 돈은 어디에 있나")
    st.markdown(
        '<p class="big-note">컨셉 색 막대가 <b>재량</b>(조절 가능한 돈), '
        '회색 막대가 <b>필수</b>(줄이기 어려운 돈)예요. 저축 여력은 컨셉 색 '
        '영역에서 나옵니다.</p>',
        unsafe_allow_html=True)
    s = get_spending_summary(tx)
    cat_df = pd.DataFrame(s["categories"])
    cat_df = cat_df[cat_df["monthly_estimate"] > 0]
    cat_df["구분"] = cat_df["is_discretionary"].map(
        {True: "재량 (조절 가능)", False: "필수"})
    order = (cat_df.groupby("category")["monthly_estimate"].sum()
             .sort_values().index.tolist())
    fig2 = go.Figure()
    for label, color in [("재량 (조절 가능)", accent), ("필수", NEUTRAL)]:
        part = cat_df[cat_df["구분"] == label]
        fig2.add_trace(go.Bar(
            y=part["category"], x=part["monthly_estimate"], name=label,
            orientation="h", marker_color=color,
            text=[won(v) for v in part["monthly_estimate"]],
            textposition="auto"))
    fig2.update_layout(height=340, barmode="stack",
                       yaxis={"categoryorder": "array",
                              "categoryarray": order},
                       margin=dict(t=10, b=10, l=10, r=10),
                       xaxis_title="월 환산 지출 (원)")
    st.plotly_chart(fig2, width='stretch')
    st.caption(
        "식비·교통처럼 한 카테고리 안에 필수와 재량이 섞인 경우, 한 줄에 두 "
        "색이 이어 붙어 표시돼요 — 예: 식비 = 기본 식사(회색) + 외식(컨셉 색).")

    with st.expander("필수/재량은 어떻게 정해졌나요? — 내 거래 분류 근거 보기"):
        st.markdown(
            "AI 분류기가 거래 한 건 한 건을 판정했어요. 기준: 기본 식사·정기권·"
            "구독·주거·통신·의료 = **필수** / 카페·뷰티·패션·외식·택시 = "
            "**재량**. 아래 표에서 직접 확인하고, 잘못된 분류가 있으면 CSV의 "
            "`is_discretionary` 값을 고쳐 다시 올리면 돼요.")
        view = tx.copy()
        view["구분"] = view["is_discretionary"].map({True: "재량", False: "필수"})
        st.dataframe(view[["date", "description", "category", "구분", "amount"]]
                     .rename(columns={"date": "날짜", "description": "내역",
                                      "category": "카테고리", "amount": "금액"}),
                     hide_index=True, width='stretch', height=300)

# ================================================================
# 플래너
# ================================================================
with tab_plan:
    st.markdown('<p class="sub-note">🔁 <b>처방</b> — 어디서 줄이면 며칠을 '
                '되찾을 수 있는지 경로를 다시 계산합니다.</p>',
                unsafe_allow_html=True)
    s = get_spending_summary(tx)
    rankings = cut_rankings(profile, tx)
    disc_cats = [rk["category"] for rk in rankings]

    with st.expander("💡 먼저 읽어보세요 — '최대 한도(%)'는 어떻게 정한 "
                     "건가요?", expanded=False):
        st.markdown(
            "플래너 곳곳에 나오는 항목별 **최대 한도**는, 항목이 **일상 유지에 "
            "얼마나 밀착돼 있는지**를 기준으로 한 서비스 기본값이에요 — 전부 "
            "없애는 비현실적 플랜을 막기 위한 안전선이죠.\n"
            "- 출처 불명·일회성: 미상 100%, 교통(택시) 100%, 여가 90%\n"
            "- 기호·쇼핑성: 뷰티/패션/쇼핑/배달 80%\n"
            "- 일상 습관성: 카페 70%\n"
            "- 생활 유지 직결: 식비(외식) 60%\n"
            "정답이 있는 숫자가 아니라 조정 가능한 기본선이에요. 각 도구의 "
            "**'최대 한도 리밋 적용'** 토글로 켜고 끌 수 있어요.")

    # ---- (1) 절감 플래너 ----
    st.subheader("✂️ 절감 플래너")
    st.markdown("어떤 항목에서 얼마나 줄일지 설계해보세요. 기본값은 서비스가 "
                "계산한 **추천 배분** — 절감 여력이 큰 항목부터, 항목별 한도 "
                "안에서 궤도 진입에 필요한 만큼만 채우는 방식이에요.")
    # 순위 카드의 '플래너에 담기' 요청 처리 (위젯 생성 전에 상태 반영)
    if "_pending_plan_add" in st.session_state:
        p = st.session_state.pop("_pending_plan_add")
        cur = st.session_state.get("plan_sel", list(disc_cats))
        if p["cat"] not in cur:
            cur = cur + [p["cat"]]
        st.session_state["plan_sel"] = cur
        st.session_state["alloc_mode"] = "🎛️ 직접 설정"
        st.session_state[f"sl_plan_{p['cat']}"] = p["pct"]
        st.session_state[f"num_plan_{p['cat']}"] = p["pct"]
        st.toast(f"'{p['cat']}' {p['pct']}%를 절감 플래너에 담았어요!",
                 icon="✂️")
    if "plan_sel" not in st.session_state:
        st.session_state["plan_sel"] = list(disc_cats)
    sel = st.multiselect("줄여볼 항목 선택", disc_cats, key="plan_sel")
    alloc_mode = st.radio(
        "배분 방식", ["🤖 추천 배분 (기본)", "⚖️ 균등 배분", "🎛️ 직접 설정"],
        horizontal=True, key="alloc_mode",
        help="추천: 큰 항목 집중, 한도 준수 / 균등: 모든 항목 같은 비율 / "
             "직접: 슬라이더로 내 마음대로")
    cap_on_plan = st.checkbox(
        "최대 한도 리밋 적용", value=(alloc_mode == "🤖 추천 배분 (기본)"),
        key="cap_plan",
        help="켜면 항목별 생활 유지 최소선까지만 줄이도록 제한해요.")
    if sel:
        cut_map = {}
        if alloc_mode == "🎛️ 직접 설정":
            st.markdown('<p class="sub-note">직접 설정 모드에서는 목표를 '
                        '바꿔도 내가 정한 비율이 그대로 유지돼요 — 목표에 '
                        '맞춰 자동으로 따라가길 원하면 추천 배분을 쓰세요.</p>',
                        unsafe_allow_html=True)
            u = uniform_cut_for_track(profile, tx, sel)
            for cat in sel:
                cap = (int(CUT_CAPS.get(cat, DEFAULT_CAP) * 100)
                       if cap_on_plan else 100)
                cut_map[cat] = pct_control(
                    f"{cat} 절감률 (한도 {cap}%)", cap,
                    int((u or 0.3) * 100), f"plan_{cat}") / 100
        elif alloc_mode == "⚖️ 균등 배분":
            u = uniform_cut_for_track(profile, tx, sel)
            if u is None:
                st.error("선택 항목을 전부 없애도 궤도 진입이 어려워요. "
                         "항목을 더 선택하거나 목표를 조정해보세요.")
            else:
                st.markdown(f"궤도 진입에 필요한 균등 절감률: **{u*100:.0f}%**")
                if cap_on_plan:
                    over = [c for c in sel
                            if u > CUT_CAPS.get(c, DEFAULT_CAP)]
                    if over:
                        st.warning(f"{', '.join(over)}은(는) 균등 절감률이 "
                                   f"한도를 넘어요 — 추천 배분이나 직접 "
                                   f"설정을 써보세요.")
            cut_map = {c: (u or 0) for c in sel}
        else:  # 추천 배분 (기본)
            sa = smart_cut_allocation(profile, tx, sel, caps_on=cap_on_plan)
            cut_map = sa["cut_map"]
            st.markdown('<p class="sub-note">추천 배분은 위 <b>줄여볼 항목 '
                        '선택</b>에 담긴 항목 안에서만 이뤄져요 — 항목을 넣고 '
                        '빼면서 추천 배분을 직접 조정할 수 있어요. 0%인 '
                        '항목은 큰 항목만으로 목표가 채워져 건드리지 않아도 '
                        '된다는 뜻이에요.</p>', unsafe_allow_html=True)
            if sa["shortfall"] == 0:
                st.success("이미 궤도 위라 절감이 필요 없어요. 그래도 "
                           "실험해보고 싶다면 직접 설정 모드를 쓰세요.")
            elif not sa["covered"]:
                st.warning(f"선택 항목을 한도까지 줄여도 월 "
                           f"{won(sa['uncovered'])}이 부족해요. 항목을 더 "
                           f"선택하거나, 한도 리밋을 끄거나, 목표를 조정해 "
                           f"보세요.")
            else:
                st.markdown(f"궤도 진입에 필요한 월 **{won(sa['shortfall'])}**"
                            f"을 절감 여력 큰 항목부터 배분했어요.")

        if any(v > 0 for v in cut_map.values()):
            pc = plan_cuts(profile, tx, cut_map)
            if alloc_mode == "🤖 추천 배분 (기본)":
                # 선택한 항목 전체 표시 (0% 포함) — 배분 대상을 명확히
                by_cat = {c["category"]: c for c in s["categories"]
                          if c["is_discretionary"]}
                rows = [{"항목": cat,
                         "절감률": f"{int(round(cut_map.get(cat, 0)*100))}%",
                         "현재 (월)": won(by_cat[cat]["monthly_estimate"]),
                         "줄일 금액": won(by_cat[cat]["monthly_estimate"]
                                       * cut_map.get(cat, 0))}
                        for cat in sel if cat in by_cat]
            else:
                rows = [{"항목": c["category"],
                         "절감률": f"{int(round(c['ratio']*100))}%",
                         "현재 (월)": won(c["monthly_before"]),
                         "줄일 금액": won(c["cut_amount"])}
                        for c in pc["per_category"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            a, b = pc["before"], pc["after"]
            if pc["became_reachable"]:
                effect = (f"**도달 불가 → {b['eta_months']}개월 만에 도달**로 "
                          f"바뀌어요")
            elif pc["days_saved"]:
                effect = f"도달이 **{pc['days_saved']}일** 앞당겨져요"
            else:
                effect = "도달 시점 변화 없음"
            st.markdown(f"이 플랜 실행 시 — 월 절감 "
                        f"**{won(pc['total_monthly_cut'])}** / GTI "
                        f"{a['gti']} → **{b['gti']}** / {effect}")

            pick = st.selectbox("어떤 항목의 구체적 품목 추천을 받아볼까요?",
                                [c["category"] for c in pc["per_category"]])
            if st.button("🤖 AI 품목 추천 받기", key="plan_ai"):
                st.session_state["plan_show_rec"] = True
            if st.session_state.get("plan_show_rec"):
                target = next(c["cut_amount"] for c in pc["per_category"]
                              if c["category"] == pick)
                with st.spinner("가계부의 실제 구매 내역을 분석하는 중..."):
                    rec_with_boost({pick: target}, "plan")

    st.divider()

    # ---- (2) 이벤트 예산 계산기 ----
    st.subheader("🎒 이벤트 예산 계산기")
    st.markdown("여행, 사고 싶은 물건 같은 일회성 지출에 **최대 얼마까지** "
                "써도 되는지 계산해요. 다른 항목을 줄여 예산을 만들 수 있어요.")
    e1, e2 = st.columns(2)
    with e1:
        event_name = st.text_input("이벤트 이름", "제주도 여행")
        delay_ok = st.slider("목표가 며칠까지 밀려도 괜찮나요?", 0, 60, 14)
    with e2:
        cut_cats = st.multiselect("예산 마련을 위해 줄일 항목 (선택)",
                                  disc_cats, key="event_cuts")
    ev_cut_map = {}
    if cut_cats:
        cap_on_ev = st.checkbox("최대 한도 리밋 적용", value=False,
                                key="cap_event",
                                help="항목별 생활 유지 최소선까지만 줄이도록 "
                                     "제한. 끄면 100%까지 자유 설정.")
        st.markdown('<p class="sub-note">항목별 절감률을 각각 정하세요 — '
                    '슬라이더를 밀거나 숫자를 직접 입력할 수 있어요.</p>',
                    unsafe_allow_html=True)
        for cat in cut_cats:
            cap = (int(CUT_CAPS.get(cat, DEFAULT_CAP) * 100)
                   if cap_on_ev else 100)
            ev_cut_map[cat] = pct_control(f"{cat} 절감률 (한도 {cap}%)",
                                          cap, 30, f"event_{cat}") / 100
    eb = event_budget(profile, tx, ev_cut_map, max_delay_days=delay_ok)
    if eb["max_budget"] > 0:
        st.success(f"**{event_name}** 예산: 최대 **{won(eb['max_budget'])}** "
                   f"(목표 지연 {delay_ok}일 이내 기준)")
        if ev_cut_map:
            st.markdown("이 예산을 만들기 위해 항목별로 줄여야 하는 금액:")
            ev_rows = []
            for c in s["categories"]:
                ratio = ev_cut_map.get(c["category"], 0)
                if c["is_discretionary"] and ratio > 0 \
                        and c["monthly_estimate"] > 0:
                    after_amt = round(c["monthly_estimate"] * (1 - ratio))
                    ev_rows.append({
                        "항목": c["category"],
                        "현재 (월)": won(c["monthly_estimate"]),
                        "절감률": f"{int(ratio*100)}%",
                        "줄여야 하는 금액": won(c["monthly_estimate"] * ratio),
                        "절감 후 월 예산": won(after_amt)})
            st.dataframe(pd.DataFrame(ev_rows), hide_index=True,
                         width='stretch')
            pick2 = st.selectbox("품목 추천 받을 항목", cut_cats,
                                 key="event_pick")
            if st.button("🤖 AI 품목 추천 받기", key="event_ai"):
                st.session_state["event_show_rec"] = True
            if st.session_state.get("event_show_rec"):
                target2 = next(
                    round(c["monthly_estimate"] * ev_cut_map[pick2])
                    for c in s["categories"]
                    if c["category"] == pick2 and c["is_discretionary"])
                with st.spinner("가계부의 실제 구매 내역을 분석하는 중..."):
                    rec_with_boost({pick2: target2}, "event")
    else:
        st.error(f"지금 조건으로는 {event_name} 예산이 0원이에요 — "
                 f"{eb.get('reason', '저축 여력 부족')}. 절감 항목을 추가하거나 "
                 f"절감률을 높여보세요.")

    st.divider()

    # ---- (3) 절감 여력 순위 ----
    st.subheader("🚀 줄여볼 만한 항목, 순위로 보기")
    st.markdown("조절 가능한 모든 항목을 절감 여력이 큰 순서로 정리했어요. "
                "각 항목의 **왜**를 확인하고, 비율을 조절하면 그 항목 하나만 "
                "줄였을 때의 효과를 계산해드려요. 여러 항목을 함께 줄이는 "
                "플랜은 위 **✂️ 절감 플래너**에서 설계하세요.")
    with st.expander("최대 한도(%)는 어떻게 정한 건가요?"):
        st.markdown("탭 맨 위의 **'💡 먼저 읽어보세요'** 설명과 같아요 — "
                    "생활 밀착도 기반 서비스 기본값이고, 아래 토글로 켜고 끌 "
                    "수 있어요.")
    cap_on_rank = st.checkbox("최대 한도 리밋 적용", value=True,
                              key="cap_rank")

    base_r = calc_gti(profile, tx)
    for rk in rankings:
        cap_pct = (int(rk["max_cut_ratio"] * 100) if cap_on_rank else 100)
        with st.expander(
                f"{rk['rank']}위 · {rk['category']} — 월 "
                f"{won(rk['monthly_estimate'])} (한도 {cap_pct}%)",
                expanded=(rk["rank"] == 1)):
            st.markdown(
                f"**왜 {rk['rank']}위인가요?** 조절 가능한 지출의 "
                f"**{rk['share_of_discretionary']*100:.0f}%**를 차지하고, "
                f"{rk['count']}건 구매에 평균 {won(rk['avg_ticket'])}씩 썼어요. "
                f"가장 큰 지출은 '{rk['biggest_item']}' "
                f"({won(rk['biggest_amount'])}).")
            ratio = pct_control("절감률 조절", cap_pct,
                                min(30, cap_pct),
                                f"rank_{rk['category']}") / 100
            if ratio > 0:
                cut_amt = round(rk["monthly_estimate"] * ratio)
                sim = simulate_scenario(profile, tx,
                                        category_cut={rk["category"]: ratio})
                after = sim["after"]
                if base_r["eta_months"] and after["eta_months"]:
                    saved = round((base_r["eta_months"]
                                   - after["eta_months"]) * 30)
                    effect = f"목표 도달이 **{saved}일** 빨라져요"
                elif after["eta_months"]:
                    effect = (f"**도달 불가 → {after['eta_months']}개월** 만에 "
                              f"도달 가능으로 바뀌어요")
                else:
                    effect = ("이 항목 하나로는 저축 전환이 안 돼요 — ✂️ "
                              "절감 플래너에서 여러 항목을 함께 줄여보세요")
                st.markdown(f"이 항목에서만 월 **{won(cut_amt)}** 절감 시 → "
                            f"GTI {base_r['gti']} → **{after['gti']}** / "
                            f"{effect}")
                col_add, col_ai = st.columns(2)
                if col_add.button("⬆ 이 비율로 절감 플래너에 담기",
                                  key=f"rank_add_{rk['category']}"):
                    st.session_state["_pending_plan_add"] = {
                        "cat": rk["category"], "pct": int(round(ratio * 100))}
                    st.rerun()
                if col_ai.button("🤖 이 금액, 구체적으로 뭘 줄여서 만들까?",
                                 key=f"rank_ai_{rk['category']}"):
                    st.session_state[f"rank_show_{rk['category']}"] = True
                if st.session_state.get(f"rank_show_{rk['category']}"):
                    with st.spinner("가계부의 실제 구매 내역을 분석하는 중..."):
                        rec_with_boost({rk["category"]: cut_amt},
                                       f"rank_{rk['category']}")

# ================================================================
# AI 상담사
# ================================================================
TOOLS = [
    {"name": "calc_gti",
     "description": "현재 GTI 지수, 저축 여력, 요구 저축액, 도달 예상 시점을 계산",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "purchase_impact",
     "description": "특정 금액의 지출 1건이 목표 도달을 며칠 미루는지 환산",
     "input_schema": {"type": "object",
                      "properties": {"amount": {"type": "integer"}},
                      "required": ["amount"]}},
    {"name": "simulate_scenario",
     "description": "지출 절감, 일회성 지출, 목표 변경 가정 시 GTI 변화를 전후 비교",
     "input_schema": {"type": "object", "properties": {
         "category_cut": {"type": "object"},
         "one_time_expense": {"type": "integer"},
         "new_goal_amount": {"type": "integer"},
         "new_goal_months": {"type": "integer"}}}},
    {"name": "required_cut",
     "description": "목표 궤도 진입에 필요한 재량 지출 절감률과 절감액을 역산",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_spending_summary",
     "description": "카테고리별 지출 합계·건수·월 환산액과 재량 상위 항목 조회",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "event_budget",
     "description": "여행·구매 등 일회성 이벤트 최대 예산 계산 (절감 가정, 허용 지연일)",
     "input_schema": {"type": "object", "properties": {
         "category_cut": {"type": "object"},
         "max_delay_days": {"type": "integer"}}}},
    {"name": "cut_rankings",
     "description": "조절 가능한 지출 카테고리를 절감 여력 순으로 순위화 (통계·최대 한도 포함)",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "suggest_items",
     "description": "특정 카테고리의 실제 구매 품목 중 목표 절감액을 만들 후보 목록",
     "input_schema": {"type": "object", "properties": {
         "category": {"type": "string"},
         "target_amount": {"type": "integer"}},
         "required": ["category", "target_amount"]}},
]

SYSTEM = """당신은 청년 자산 형성 코칭 서비스 '궤도'의 AI 재무 상담사입니다.

원칙:
1. 모든 숫자는 반드시 도구(tool) 호출 결과에서 가져옵니다. 직접 계산하거나 추정하지 마세요.
2. 재무 결정 질문에는 simulate_scenario, purchase_impact, event_budget을 호출해
   구체적 숫자로 답합니다. "뭘 줄일까" 질문에는 cut_rankings와 suggest_items로
   실제 구매 품목 기반의 추천을 합니다.
3. 판단은 사용자 몫입니다. "안 됩니다" 대신 선택지를 제시하세요.
4. 소비를 도덕적으로 평가하지 마세요. 숫자와 트레이드오프만 전달합니다.
5. 데이터 관측 기간이 짧아 월 환산에 왜곡이 있을 수 있음을 필요할 때 언급하세요.
6. 답변은 간결하게, 한국어로. 핵심 숫자를 먼저 말하고 설명을 붙이세요.
7. 투자 상품 추천, 세무·법률 자문은 하지 않습니다. 요청받으면 전문가 상담을 안내하세요."""


def run_tool(name: str, args: dict) -> object:
    if name == "calc_gti":
        return calc_gti(profile, tx)
    if name == "purchase_impact":
        return purchase_impact(profile, tx, args["amount"])
    if name == "simulate_scenario":
        return simulate_scenario(
            profile, tx, category_cut=args.get("category_cut"),
            one_time_expense=args.get("one_time_expense", 0) or 0,
            new_goal_amount=args.get("new_goal_amount"),
            new_goal_months=args.get("new_goal_months"))
    if name == "required_cut":
        return required_cut(profile, tx)
    if name == "get_spending_summary":
        return get_spending_summary(tx)
    if name == "event_budget":
        return event_budget(profile, tx, args.get("category_cut"),
                            args.get("max_delay_days", 14) or 14)
    if name == "cut_rankings":
        return cut_rankings(profile, tx)
    if name == "suggest_items":
        return suggest_items(tx, args["category"], args["target_amount"])
    return {"error": f"알 수 없는 tool: {name}"}


with tab_chat:
    st.markdown('<p class="sub-note">💬 <b>상담</b> — 결정이 필요한 순간, '
                '무엇이든 물어보면 숫자로 답합니다.</p>',
                unsafe_allow_html=True)
    api_key = get_api_key()
    if not api_key:
        st.warning("ANTHROPIC_API_KEY가 설정되지 않아 AI 상담을 사용할 수 "
                   "없어요. 대시보드와 플래너는 정상 동작합니다.")
        st.stop()

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("**이렇게 물어보세요**")
        examples = ["나 지금 목표 잘 가고 있어?",
                    "다음 달에 제주도 여행 30만원 써도 돼?",
                    "구체적으로 뭘 안 사면 목표를 앞당길 수 있어?"]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, width='stretch'):
                st.session_state.pending = ex
                st.rerun()

    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant") and isinstance(m["content"], str):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    user_input = st.chat_input("소비, 목표, 뭐든 물어보세요")
    if "pending" in st.session_state:
        user_input = st.session_state.pop("pending")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        api_messages = [{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                        if isinstance(m["content"], str)]

        with st.chat_message("assistant"):
            with st.spinner("계산 중..."):
                try:
                    while True:
                        resp = client.messages.create(
                            model=CHAT_MODEL, max_tokens=1500,
                            system=SYSTEM, tools=TOOLS, messages=api_messages)
                        if resp.stop_reason != "tool_use":
                            break
                        api_messages.append(
                            {"role": "assistant", "content": resp.content})
                        results = []
                        for block in resp.content:
                            if block.type == "tool_use":
                                out = run_tool(block.name, block.input)
                                results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(
                                        out, ensure_ascii=False, default=str)})
                        api_messages.append({"role": "user", "content": results})
                    answer = "".join(b.text for b in resp.content
                                     if b.type == "text")
                except Exception as e:
                    answer = ("죄송해요, 지금 상담 응답에 문제가 생겼어요. "
                              "잠시 후 다시 시도해 주세요.")
                    st.caption(f"오류: {type(e).__name__}")
            st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
