"""
GTI (Goal Track Index) 계산 엔진
================================
설계 원칙: "계산은 수식, 해석은 AI"
- 이 모듈의 모든 함수는 확정적(deterministic)입니다. LLM은 여기 없습니다.
- Claude API에서는 이 함수들을 tool로 등록해 호출만 합니다.

거래 데이터 형식 (DataFrame):
    date            : 날짜 (YYYY-MM-DD)
    description     : 거래 내역 텍스트 (예: "올리브영 강남점")
    amount          : 지출 금액 (원, 양수)
    category        : 카테고리 (식비/카페/뷰티/교통/주거/통신/배달/쇼핑/기타)
    is_discretionary: 재량 지출 여부 (True=재량, False=필수)
    ※ category, is_discretionary는 MVP에서 LLM 분류 결과가 채워줌
"""

from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# 사용자 프로필 (온보딩에서 입력받는 값)
# ---------------------------------------------------------------
@dataclass
class UserProfile:
    monthly_income: int      # 월 소득 I (원)
    current_assets: int      # 현재 자산 A (원)
    goal_amount: int         # 목표 금액 G (원)
    goal_months: int         # 남은 개월 수 T

# 변동성 페널티 가중치 (튜닝 대상)
LAMBDA = 0.25
# 궤도율 상한 캡 (120% 초과분은 점수에 반영 안 함)
CAP = 1.2
# 월 1회 청구되는 고정비 카테고리 — 관측 기간이 짧아도 월 환산(30/n일 배율)
# 을 적용하면 안 되고 금액 그대로 월 비용으로 취급해야 함.
# (검증 중 발견: 18일 데이터의 월세를 30/18배 하면 고정비가 과대평가됨)
RECURRING_CATEGORIES = {"주거", "통신", "구독", "보험"}


# ---------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------
def _month_periods(df: pd.DataFrame) -> int:
    """관측 기간이 몇 '달'에 해당하는지 (30일 단위 반올림, 최소 1).
    월정기 지출이 여러 달치 관측되면 그 수로 나눠 월 비용을 구함.
    (검증에서 발견: 60일 데이터에서 월세가 2회 합산돼 고정비 2배 버그)"""
    dates = pd.to_datetime(df["date"])
    days = (dates.max() - dates.min()).days + 1
    return max(round(days / 30), 1)


def _monthly_scale(df: pd.DataFrame) -> float:
    """관측 기간이 한 달 미만이면 30일 기준으로 환산하는 배율."""
    dates = pd.to_datetime(df["date"])
    observed_days = (dates.max() - dates.min()).days + 1
    return 30.0 / max(observed_days, 1)


def _weekly_discretionary_cv(df: pd.DataFrame) -> float:
    """재량 지출의 주 단위 변동계수(CV = 표준편차/평균).

    - 7일 단위로 잘라 완전한 주만 사용
    - 완전한 주가 2개 미만이면 일 단위 CV로 대체 (18일 데이터도 동작하도록)
    - 지출이 없으면 0
    """
    disc = df[df["is_discretionary"]].copy()
    if disc.empty:
        return 0.0
    disc["date"] = pd.to_datetime(disc["date"])
    start = pd.to_datetime(df["date"]).min()
    disc["week_idx"] = ((disc["date"] - start).dt.days // 7)

    total_days = (pd.to_datetime(df["date"]).max() - start).days + 1
    full_weeks = total_days // 7
    weekly = (
        disc[disc["week_idx"] < full_weeks]
        .groupby("week_idx")["amount"].sum()
        .reindex(range(full_weeks), fill_value=0)
    )

    if full_weeks >= 2 and weekly.mean() > 0:
        # 환불 등으로 CV가 음수가 될 수 없도록 하한 0 (버그 수정:
        # 극단 절감 시나리오에서 음수 CV가 페널티를 보너스로 뒤집던 문제)
        return float(max(weekly.std(ddof=0) / weekly.mean(), 0.0))

    # 폴백: 일 단위 CV (0원인 날 포함해 전체 기간으로 계산)
    daily = (
        disc.groupby(disc["date"].dt.date)["amount"].sum()
        .reindex(pd.date_range(start, pd.to_datetime(df["date"]).max()).date,
                 fill_value=0)
    )
    if daily.mean() <= 0:
        return 0.0
    # 일 단위 CV는 원래 크게 나오므로 주 단위 스케일로 완화 (sqrt(7)로 나눔)
    return float(max(daily.std(ddof=0) / daily.mean() / np.sqrt(7), 0.0))


# ---------------------------------------------------------------
# Tool 1: calc_gti — 목표 궤도 지수
# ---------------------------------------------------------------
def _recurring_mask(df: pd.DataFrame) -> pd.Series:
    """월 1회 지출(환산 제외) 판별.

    - 'is_recurring' 컬럼이 있으면 그것을 우선 사용 (행 단위 제어)
    - 없으면 카테고리 기반 폴백
    실데이터 검증에서 발견: 30일 교통 정기권처럼 카테고리만으로는
    구분 불가능한 월정기 지출이 존재함 → 행 단위 플래그 필요.
    """
    if "is_recurring" in df.columns:
        return df["is_recurring"].fillna(False).astype(bool)
    return df["category"].isin(RECURRING_CATEGORIES)


def calc_gti(profile: UserProfile, transactions: pd.DataFrame,
             lam: float = LAMBDA) -> dict:
    """GTI 3층 계산. Claude tool use의 메인 함수.

    반환값의 모든 숫자는 Claude가 코칭 문장을 만들 때 근거로 사용.
    """
    scale = _monthly_scale(transactions)

    fixed = transactions[~transactions["is_discretionary"]]
    recurring_mask = _recurring_mask(fixed)
    # 월정기 고정비: 금액 그대로 / 일반 고정비(식비 등): 관측기간 → 월 환산
    periods = _month_periods(transactions)
    e_fix = (fixed.loc[recurring_mask, "amount"].sum() / periods
             + fixed.loc[~recurring_mask, "amount"].sum() * scale)
    e_var = transactions.loc[transactions["is_discretionary"], "amount"].sum() * scale

    # 1층: 저축 여력
    savings_capacity = profile.monthly_income - e_fix - e_var

    # 2층: 요구 저축액
    required = (profile.goal_amount - profile.current_assets) / profile.goal_months

    # 3층: 궤도율 → 지수 → 변동성 페널티
    if required <= 0:            # 이미 목표 달성
        track_ratio = CAP
    elif savings_capacity <= 0:  # 저축 여력 없음
        track_ratio = 0.0
    else:
        track_ratio = savings_capacity / required

    gti_raw = 100 * min(track_ratio, CAP) / CAP
    cv = _weekly_discretionary_cv(transactions)
    gti_final = gti_raw * (1 - lam * min(cv, 1.0))  # CV 폭주 방지 위해 1.0 캡

    # 파생: 목표 도달 예상 개월
    if savings_capacity > 0:
        eta_months = (profile.goal_amount - profile.current_assets) / savings_capacity
        delay_months = eta_months - profile.goal_months
    else:
        eta_months, delay_months = None, None

    return {
        "gti": round(gti_final, 1),
        "gti_before_penalty": round(gti_raw, 1),
        "track_ratio": round(track_ratio, 3),
        "volatility_cv": round(cv, 3),
        "monthly_savings_capacity": round(savings_capacity),
        "monthly_required_savings": round(required),
        "monthly_fixed_expense": round(e_fix),
        "monthly_discretionary_expense": round(e_var),
        "eta_months": round(eta_months, 1) if eta_months else None,
        "delay_months": round(delay_months, 1) if delay_months is not None else None,
    }


# ---------------------------------------------------------------
# Tool 2: purchase_impact — 소비 1건 → 목표 지연일 환산 (킬러 기능)
# ---------------------------------------------------------------
def purchase_impact(profile: UserProfile, transactions: pd.DataFrame,
                    amount: int) -> dict:
    """지출 1건이 목표 도달일을 며칠 미루는지 계산.

    논리: 저축 여력 S로 매달 저축한다고 할 때, amount만큼 자산이 줄면
    그만큼 모으는 데 걸리는 시간 = amount / S 개월 = amount / S * 30 일
    """
    base = calc_gti(profile, transactions)
    s = base["monthly_savings_capacity"]
    if s <= 0:
        return {"delay_days": None, "message_hint": "저축 여력이 0 이하라 환산 불가",
                "current_gti": base["gti"]}
    delay_days = amount / s * 30
    return {
        "amount": amount,
        "delay_days": round(delay_days, 1),
        "delay_hours": round(delay_days * 24, 1),
        "current_gti": base["gti"],
        "monthly_savings_capacity": s,
    }


# ---------------------------------------------------------------
# Tool 3: simulate_scenario — 가정 변경 시뮬레이션
# ---------------------------------------------------------------
def simulate_scenario(profile: UserProfile, transactions: pd.DataFrame,
                      category_cut: dict | None = None,
                      one_time_expense: int = 0,
                      new_goal_amount: int | None = None,
                      new_goal_months: int | None = None) -> dict:
    """조건을 바꿨을 때 GTI가 어떻게 변하는지 전/후 비교.

    category_cut     : {"카페": 0.3} → 카페 지출 30% 절감 가정
    one_time_expense : 일회성 지출 (예: 여행 300000)
    new_goal_*       : 목표 변경 가정
    """
    before = calc_gti(profile, transactions)

    tx = transactions.copy()
    tx["amount"] = tx["amount"].astype(float)
    if category_cut:
        for cat, ratio in category_cut.items():
            # 버그 수정: 절감은 해당 카테고리의 '재량' 지출에만 적용.
            # (식비 절감이 기본 식사(필수)까지 깎던 문제)
            mask = (tx["category"] == cat) & tx["is_discretionary"] \
                   & (tx["amount"] > 0)
            tx.loc[mask, "amount"] = tx.loc[mask, "amount"] * (1 - ratio)

    new_profile = UserProfile(
        monthly_income=profile.monthly_income,
        current_assets=profile.current_assets - one_time_expense,
        goal_amount=new_goal_amount or profile.goal_amount,
        goal_months=new_goal_months or profile.goal_months,
    )
    after = calc_gti(new_profile, tx)

    result = {"before": before, "after": after,
              "gti_change": round(after["gti"] - before["gti"], 1)}
    if one_time_expense > 0:
        impact = purchase_impact(profile, transactions, one_time_expense)
        result["one_time_expense_delay_days"] = impact["delay_days"]
    return result


# ---------------------------------------------------------------
# Tool 5: required_cut — 궤도 진입에 필요한 재량 절감률 역산
# ---------------------------------------------------------------
def required_cut(profile: UserProfile, transactions: pd.DataFrame) -> dict:
    """궤도율 1.0(GTI 만점 구간)에 도달하려면 재량 지출을 몇 % 줄여야
    하는지 역산. GTI가 0인 사용자에게 '그래서 뭘 해야 하는데?'를
    구체적 숫자로 답하는 기능.
    """
    base = calc_gti(profile, transactions)
    e_fix = base["monthly_fixed_expense"]
    e_var = base["monthly_discretionary_expense"]
    required = base["monthly_required_savings"]

    # 궤도율 1.0에 필요한 재량 지출 상한
    allowed_var = profile.monthly_income - e_fix - required

    if e_var <= 0:
        return {"feasible": allowed_var >= 0, "cut_ratio": 0.0, "base": base}
    if allowed_var < 0:
        # 재량을 전부 없애도 불가능 → 목표 조정 필요
        max_goal = (profile.monthly_income - e_fix) * profile.goal_months \
                   + profile.current_assets
        return {
            "feasible": False,
            "cut_ratio": None,
            "reachable_goal_amount": round(max(max_goal, 0)),
            "message_hint": "재량 지출 전액 절감으로도 목표 불가. 목표 금액 또는 기한 조정 필요",
            "base": base,
        }
    cut = 1 - allowed_var / e_var
    return {
        "feasible": True,
        "cut_ratio": round(max(cut, 0.0), 3),
        "allowed_monthly_discretionary": round(allowed_var),
        "current_monthly_discretionary": round(e_var),
        "monthly_cut_amount": round(e_var - allowed_var),
        "base": base,
    }


# ---------------------------------------------------------------
# Tool 7: plan_cuts — 선택 카테고리 절감 플랜 계산
# ---------------------------------------------------------------
def plan_cuts(profile: UserProfile, transactions: pd.DataFrame,
              cut_map: dict[str, float]) -> dict:
    """카테고리별 절감률(cut_map)을 적용했을 때 카테고리별 절감액과
    전/후 GTI를 계산. 절감 플래너 UI의 백엔드.
    """
    summary = get_spending_summary(transactions)
    per_cat = []
    for c in summary["categories"]:
        if not c["is_discretionary"]:
            continue  # 절감은 재량 지출만 대상 (필수 침범 버그 수정)
        ratio = cut_map.get(c["category"], 0.0)
        if ratio > 0 and c["monthly_estimate"] > 0:
            per_cat.append({
                "category": c["category"],
                "monthly_before": c["monthly_estimate"],
                "cut_amount": round(c["monthly_estimate"] * ratio),
                "monthly_after": round(c["monthly_estimate"] * (1 - ratio)),
                "ratio": ratio,
            })
    sim = simulate_scenario(profile, transactions, category_cut=cut_map)

    # 도달 단축 효과 계산 (일 단위)
    b_eta, a_eta = sim["before"]["eta_months"], sim["after"]["eta_months"]
    if b_eta and a_eta:
        days_saved = round((b_eta - a_eta) * 30)
    elif a_eta and not b_eta:
        days_saved = None  # '도달 불가 → 가능'으로 전환된 경우
    else:
        days_saved = 0
    return {"per_category": per_cat,
            "total_monthly_cut": sum(p["cut_amount"] for p in per_cat),
            "days_saved": days_saved,
            "became_reachable": bool(a_eta and not b_eta),
            "before": sim["before"], "after": sim["after"]}


def uniform_cut_for_track(profile: UserProfile, transactions: pd.DataFrame,
                          categories: list[str]) -> float | None:
    """선택한 카테고리들만 '동일 비율'로 줄여 궤도(궤도율 1.0)에 진입하려면
    필요한 비율. 전액(100%) 절감으로도 불가능하면 None.
    """
    base = calc_gti(profile, transactions)
    shortfall = base["monthly_required_savings"] - base["monthly_savings_capacity"]
    if shortfall <= 0:
        return 0.0
    summary = get_spending_summary(transactions)
    pool = sum(c["monthly_estimate"] for c in summary["categories"]
               if c["category"] in categories and c["is_discretionary"]
               and c["monthly_estimate"] > 0)
    if pool <= 0 or shortfall > pool:
        return None
    return round(shortfall / pool, 3)


# ---------------------------------------------------------------
# Tool 8: event_budget — 이벤트 최대 예산 계산
# ---------------------------------------------------------------
def event_budget(profile: UserProfile, transactions: pd.DataFrame,
                 category_cut: dict[str, float] | None = None,
                 max_delay_days: int = 14) -> dict:
    """"이 이벤트(여행·구매 등)에 최대 얼마까지 써도 되는가"를 계산.

    기준: 선택한 절감을 실행한다고 가정한 저축 여력 S'에서,
    일회성 지출 X가 목표를 X/S'×30일 미루므로
    허용 지연일(max_delay_days) 이내가 되는 최대 X를 역산.
    """
    sim = simulate_scenario(profile, transactions,
                            category_cut=category_cut or {})
    s_after = sim["after"]["monthly_savings_capacity"]
    if s_after <= 0:
        return {"max_budget": 0, "reason": "절감 후에도 저축 여력이 없음",
                "savings_after_cut": s_after, "after": sim["after"]}
    max_budget = int(s_after * max_delay_days / 30)
    return {"max_budget": max_budget,
            "max_delay_days": max_delay_days,
            "savings_after_cut": s_after,
            "on_track_after_cut": sim["after"]["track_ratio"] >= 1.0,
            "after": sim["after"]}


# ---------------------------------------------------------------
# Tool 9: acceleration_options — 절감 여력 큰 항목의 가속 효과
# ---------------------------------------------------------------
def acceleration_options(profile: UserProfile,
                         transactions: pd.DataFrame) -> dict:
    """재량 지출 1위 카테고리를 단계별(10~50%)로 줄일 때
    목표 도달 시점이 얼마나 앞당겨지는지 계산.
    """
    summary = get_spending_summary(transactions)
    if not summary["top_discretionary"]:
        return {"target_category": None, "steps": []}
    target = summary["top_discretionary"][0]["category"]

    base = calc_gti(profile, transactions)
    steps = []
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        sim = simulate_scenario(profile, transactions,
                                category_cut={target: ratio})
        a = sim["after"]
        saved_months = (round(base["eta_months"] - a["eta_months"], 1)
                        if base["eta_months"] and a["eta_months"] else None)
        steps.append({"ratio": ratio, "gti": a["gti"],
                      "eta_months": a["eta_months"],
                      "months_saved": saved_months})
    return {"target_category": target,
            "target_monthly": summary["top_discretionary"][0]["monthly_estimate"],
            "base_eta_months": base["eta_months"], "steps": steps}


# ---------------------------------------------------------------
# 목표 3요소 역산 (금액·기한·저축여력 중 2개 → 나머지 1개)
# ---------------------------------------------------------------
def solve_goal(monthly_savings_capacity: float, current_assets: int,
               goal_amount: int | None = None,
               goal_months: int | None = None,
               potential_capacity: float | None = None) -> dict:
    """온보딩 '2개만 입력' 모드의 백엔드.
    - 금액+기한 입력 → 필요 저축액과 현재 여력 비교
    - 금액만 입력   → 도달 기한 계산
    - 기한만 입력   → 달성 가능 금액 계산
    potential_capacity: 재량 지출을 전부 줄였을 때의 최대 여력.
    현재 여력이 0 이하여도 이 값 기준의 계산을 함께 제공 (사용자 피드백:
    여력이 없어도 목표 계산은 보여줘야 함).
    """
    s = monthly_savings_capacity
    p = potential_capacity
    if goal_amount and goal_months:
        need = (goal_amount - current_assets) / goal_months
        return {"mode": "check", "required_monthly": round(need),
                "feasible": s >= need}
    if goal_amount and not goal_months:
        out = {"mode": "months", "months": None, "potential_months": None}
        if s > 0:
            out["months"] = (goal_amount - current_assets) / s
        if p and p > 0:
            out["potential_months"] = (goal_amount - current_assets) / p
        return out
    if goal_months and not goal_amount:
        out = {"mode": "amount",
               "amount": round(current_assets + max(s, 0) * goal_months),
               "potential_amount": None}
        if p and p > 0:
            out["potential_amount"] = round(current_assets + p * goal_months)
        return out
    return {"mode": "invalid"}


# ---------------------------------------------------------------
# Tool 10: cut_rankings — 절감 여력 전체 순위 분석
# ---------------------------------------------------------------
# 카테고리별 최대 절감 한도 (생활 밀착도 기반 휴리스틱):
# 일상 유지와 무관할수록 한도가 높음. MVP 기본값이며 조정 가능.
CUT_CAPS = {"미상": 1.0, "여가": 0.9, "패션": 0.8, "쇼핑": 0.8, "뷰티": 0.8,
            "배달": 0.8, "카페": 0.7, "식비": 0.6, "교통": 1.0}
DEFAULT_CAP = 0.8


def cut_rankings(profile: UserProfile, transactions: pd.DataFrame) -> list[dict]:
    """모든 재량 카테고리를 절감 여력 순으로 순위화하고,
    선정 이유가 될 통계와 최대 절감 한도를 함께 반환.
    """
    disc = transactions[transactions["is_discretionary"]
                        & (transactions["amount"] > 0)]
    if disc.empty:
        return []
    summary = get_spending_summary(transactions)
    disc_total = sum(c["monthly_estimate"] for c in summary["categories"]
                     if c["is_discretionary"] and c["monthly_estimate"] > 0)
    out = []
    for c in summary["categories"]:
        if not c["is_discretionary"] or c["monthly_estimate"] <= 0:
            continue
        g = disc[disc["category"] == c["category"]]
        if g.empty:
            continue
        biggest = g.loc[g["amount"].idxmax()]
        out.append({
            "category": c["category"],
            "monthly_estimate": c["monthly_estimate"],
            "share_of_discretionary": round(
                c["monthly_estimate"] / disc_total, 3) if disc_total else 0,
            "count": int(len(g)),
            "avg_ticket": round(g["amount"].mean()),
            "biggest_item": str(biggest["description"]),
            "biggest_amount": int(biggest["amount"]),
            "max_cut_ratio": CUT_CAPS.get(c["category"], DEFAULT_CAP),
        })
    out.sort(key=lambda x: x["monthly_estimate"], reverse=True)
    for i, o in enumerate(out):
        o["rank"] = i + 1
    return out


# ---------------------------------------------------------------
# Tool 11: suggest_items — 목표 절감액을 만들 구체 품목 후보 (확정적 폴백)
# ---------------------------------------------------------------
def suggest_items(transactions: pd.DataFrame, category: str,
                  target_amount: int) -> dict:
    """해당 카테고리의 실제 구매 내역에서 큰 것부터 골라
    목표 절감액을 채우는 후보 목록. (AI 추천의 확정적 폴백이자,
    AI 추천 프롬프트에 넣을 근거 데이터)
    """
    g = transactions[(transactions["category"] == category)
                     & transactions["is_discretionary"]
                     & (transactions["amount"] > 0)].copy()
    g = g.sort_values("amount", ascending=False)
    picked, acc = [], 0
    for _, row in g.iterrows():
        if acc >= target_amount:
            break
        picked.append({"description": str(row["description"]),
                       "amount": int(row["amount"]),
                       "date": str(row["date"])})
        acc += int(row["amount"])
    return {"category": category, "target_amount": target_amount,
            "picked_items": picked, "picked_total": acc,
            "reached": acc >= target_amount,
            "all_items": g[["description", "amount"]].to_dict("records")}
def get_spending_summary(transactions: pd.DataFrame) -> dict:
    """카테고리별 합계·건수, 재량 지출 상위 카테고리 등 요약."""
    scale = _monthly_scale(transactions)
    by_cat = (
        transactions.groupby(["category", "is_discretionary"])["amount"]
        .agg(["sum", "count"]).reset_index()
    )
    categories = [
        {
            "category": r["category"],
            "is_discretionary": bool(r["is_discretionary"]),
            "total": int(r["sum"]),
            "monthly_estimate": round(
                r["sum"] / _month_periods(transactions)
                if r["category"] in RECURRING_CATEGORIES
                else r["sum"] * scale),
            "count": int(r["count"]),
        }
        for _, r in by_cat.iterrows()
    ]
    disc_sorted = sorted(
        [c for c in categories if c["is_discretionary"]],
        key=lambda c: c["total"], reverse=True,
    )
    return {
        "observed_days": int((pd.to_datetime(transactions["date"]).max()
                              - pd.to_datetime(transactions["date"]).min()).days + 1),
        "total_spent": int(transactions["amount"].sum()),
        "categories": categories,
        "top_discretionary": disc_sorted[:3],
    }


# ---------------------------------------------------------------
# Tool 12: detect_anomalies — 가계부 특이사항 자동 감지 (확정적)
# ---------------------------------------------------------------
def detect_anomalies(transactions: pd.DataFrame) -> list[dict]:
    """월 환산·코칭 해석에 영향을 주는 특이 패턴을 규칙 기반으로 감지.
    (감지와 문구 모두 확정적 — 같은 데이터면 항상 같은 노트)
    """
    notes = []
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    days = (df["date"].max() - df["date"].min()).days + 1
    pos = df[df["amount"] > 0]

    if days < 30:
        notes.append({"type": "짧은 관측 기간",
                      "note": f"데이터가 {days}일치예요. 월 환산(30일 기준)에 "
                              f"왜곡이 있을 수 있고, 기록이 쌓일수록 정확해져요."})

    # 대량 구매일: 일 지출이 중앙값의 3배 이상 & 10만원 초과
    daily = pos.groupby(pos["date"].dt.date)["amount"].sum()
    med = daily.median()
    for d, v in daily.items():
        if v > max(3 * med, 100000):
            notes.append({"type": "대량 구매일",
                          "note": f"{d.strftime('%m/%d')}에 하루 {v:,.0f}원이 "
                                  f"집중됐어요. 시즌성 몰아 구매라면 월 환산 "
                                  f"지출이 평소보다 부풀려져 보일 수 있어요."})

    # 반복 습관성 소액: 재량 카테고리에서 10건 이상
    disc = pos[pos["is_discretionary"]]
    for cat, g in disc.groupby("category"):
        if len(g) >= 10:
            notes.append({"type": "반복 습관 지출",
                          "note": f"{cat}에서 {days}일간 {len(g)}건(평균 "
                                  f"{g['amount'].mean():,.0f}원)을 반복 구매"
                                  f"했어요. 건당은 작아도 합계 영향이 커서 "
                                  f"횟수 조절 효과가 좋은 항목이에요."})

    # 환불
    refunds = df[df["amount"] < 0]
    if not refunds.empty:
        notes.append({"type": "반품·환불",
                      "note": f"환불 {len(refunds)}건(총 "
                              f"{-refunds['amount'].sum():,.0f}원)이 있어 "
                              f"순지출 기준으로 계산했어요."})

    # 미상 지출
    unk = pos[pos["category"] == "미상"]
    if not unk.empty:
        notes.append({"type": "출처 불명 지출",
                      "note": f"출처를 특정하지 못한 지출 "
                              f"{unk['amount'].sum():,.0f}원이 있어요. "
                              f"재량 지출로 보수적으로 분류했어요."})
    return notes


# ---------------------------------------------------------------
# Tool 13: smart_cut_allocation — 추천 절감 배분 (확정적 알고리즘)
# ---------------------------------------------------------------
def smart_cut_allocation(profile: UserProfile, transactions: pd.DataFrame,
                         categories: list[str],
                         caps_on: bool = True) -> dict:
    """궤도 진입에 필요한 절감액을 '절감 여력이 큰 항목부터, 항목별 한도
    안에서' 채우는 추천 배분. 큰 항목에 집중해 여러 항목의 생활 충격을
    줄이는 그리디 방식.
    """
    base = calc_gti(profile, transactions)
    shortfall = max(base["monthly_required_savings"]
                    - base["monthly_savings_capacity"], 0)
    summary = get_spending_summary(transactions)
    cats = [c for c in summary["categories"]
            if c["is_discretionary"] and c["category"] in categories
            and c["monthly_estimate"] > 0]
    cats.sort(key=lambda c: c["monthly_estimate"], reverse=True)

    cut_map, remaining = {}, shortfall
    for c in cats:
        cap = CUT_CAPS.get(c["category"], DEFAULT_CAP) if caps_on else 1.0
        take = min(c["monthly_estimate"] * cap, remaining)
        cut_map[c["category"]] = (round(take / c["monthly_estimate"], 3)
                                  if take > 0 else 0.0)
        remaining -= take
    return {"cut_map": cut_map,
            "shortfall": round(shortfall),
            "covered": remaining <= 1,
            "uncovered": round(max(remaining, 0))}
