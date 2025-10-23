# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

from fetch_btc import fetch_btc_usdt_1h_last_days, resample_to_kst_daily
from bazi_qimen import pillars_for_date_kst
from agent_core import AgentState, score_row, prob_from_score, update_state, build_report

OUT_DIR = Path("Reports")
STATE_PATH = Path("state.json")
KST = timezone(timedelta(hours=9))

def main():
    # 1) BTC 시세(1h) → KST 일봉 재집계
    df1h = fetch_btc_usdt_1h_last_days(days=70)
    daily = resample_to_kst_daily(df1h)  # columns: [date_kst, close_kst, ret_kst, rsi14, rsi_sig]

    # 2) 날짜별 기문‧오행 (일주 기반; override csv 있으면 그 값 사용)
    daily["date_kst"] = pd.to_datetime(daily["date_kst"]).dt.date
    p_rows = []
    for d in daily["date_kst"]:
        p = pillars_for_date_kst(d)
        p_rows.append({
            "date_kst": d,
            "wood": p.wood, "fire": p.fire, "earth": p.earth, "metal": p.metal, "water": p.water,
            "gate": p.gate, "ganji_day": p.ganji_day
        })
    pillars_df = pd.DataFrame(p_rows)

    merged = pd.merge(
        pillars_df,
        daily[["date_kst", "ret_kst", "rsi14", "rsi_sig"]],
        on="date_kst",
        how="inner"
    ).sort_values("date_kst").reset_index(drop=True)

    if merged.empty or len(merged) < 2:
        raise RuntimeError("데이터가 부족합니다(최소 2일 이상 필요).")

    today_kst = datetime.now(tz=KST).date()
    verify_target = today_kst - timedelta(days=1)

    # 3) 상태 로드 & 최근 30일 온라인 로지스틱 보정
    st = AgentState.load(STATE_PATH)
    train = merged[(merged["date_kst"] < today_kst)].tail(30)
    for _, r in train.iterrows():
        actual = 1 if r["ret_kst"] > 0 else -1
        s = score_row(st, r)
        p = prob_from_score(s)
        update_state(st, r, p, actual)

    # 4) 오늘(예측행) 구성: pillars는 오늘값, RSI/rsi_sig는 최신값 보간
    row_today = pillars_df[pillars_df["date_kst"] == today_kst]
    if row_today.empty:
        # 아직 시세집계가 끝나기 전일 수 있으니, pillars만 생성
        p = pillars_for_date_kst(today_kst)
        row_today = pd.DataFrame([{
            "date_kst": today_kst,
            "wood": p.wood, "fire": p.fire, "earth": p.earth, "metal": p.metal, "water": p.water,
            "gate": p.gate, "ganji_day": p.ganji_day
        }])
    rsi_last = merged["rsi14"].dropna().iloc[-1] if merged["rsi14"].notna().any() else float("nan")
    rsisig_last = merged["rsi_sig"].dropna().iloc[-1] if merged["rsi_sig"].notna().any() else 0.0
    row_today = row_today.iloc[0].to_dict()
    row_today["rsi14"] = rsi_last
    row_today["rsi_sig"] = rsisig_last

    # 5) 예측 확률
    s_today = score_row(st, row_today)
    p_today = prob_from_score(s_today)

    # 6) 어제 실제값(검증)
    row_y = merged[merged["date_kst"] == verify_target]
    actual_label = 1 if (not row_y.empty and (row_y["ret_kst"].iloc[0] > 0)) else -1

    # 7) 리포트 출력/저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today_str = str(today_kst)
    report_txt = build_report(
        today_str, str(today_kst), str(verify_target),
        row_today, p_today, actual_label, st
    )
    (OUT_DIR / f"report_{today_str}.txt").write_text(report_txt, encoding="utf-8")
    st.save(STATE_PATH)
    print(report_txt)

if __name__ == "__main__":
    main()
