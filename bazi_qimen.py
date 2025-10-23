# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import os

HEAVENLY = list("甲乙丙丁戊己庚辛壬癸")
EARTHLY = list("子丑寅卯辰巳午未申酉戌亥")
ELEMENT_OF_HEAVENLY = {
    "甲":"wood","乙":"wood","丙":"fire","丁":"fire","戊":"earth",
    "己":"earth","庚":"metal","辛":"metal","壬":"water","癸":"water"
}
GATES = list("開休生傷杜景死驚")

@dataclass
class PillarDay:
    date_kst: date
    ganji_day: str
    gate: str
    wood: float
    fire: float
    earth: float
    metal: float
    water: float

# 앵커: 2025-10-20(KST) = 壬戌日(水土) 조건을 만족하도록 설정
ANCHOR_DATE = date(2025, 10, 20)
ANCHOR_HEAVENLY = "壬"
ANCHOR_EARTHLY  = "戌"

def _ganji_for_date(d: date) -> str:
    delta = (d - ANCHOR_DATE).days
    idx_h = (HEAVENLY.index(ANCHOR_HEAVENLY) + delta) % 10
    idx_e = (EARTHLY.index(ANCHOR_EARTHLY) + delta) % 12
    return HEAVENLY[idx_h] + EARTHLY[idx_e]

def _elements_from_ganji(ganji: str):
    stem = ganji[0]; branch = ganji[1]
    base = {"wood":0.18,"fire":0.18,"earth":0.18,"metal":0.18,"water":0.18}
    dom = ELEMENT_OF_HEAVENLY.get(stem)
    if dom: base[dom] += 0.12
    if branch in "子亥": base["water"] += 0.04
    if branch in "寅卯": base["wood"]  += 0.04
    if branch in "巳午": base["fire"]  += 0.04
    if branch in "申酉": base["metal"] += 0.04
    if branch in "辰戌丑未": base["earth"] += 0.04
    s = sum(base.values())
    for k in base: base[k] = round(base[k]/s, 4)
    return base

def _gate_from_ganji(ganji: str) -> str:
    idx = (ord(ganji[0]) + ord(ganji[1])) % len(GATES)
    return GATES[idx]

def pillars_for_date_kst(d: date) -> PillarDay:
    # 사용자 오버라이드(csv)가 있으면 우선 사용
    if os.path.exists("pillars_overrides.csv"):
        df = pd.read_csv("pillars_overrides.csv")
        df["date_kst"] = pd.to_datetime(df["date_kst"]).dt.date
        hit = df[df["date_kst"] == d]
        if not hit.empty:
            r = hit.iloc[0]
            return PillarDay(
                date_kst=d, ganji_day="(override)",
                gate=str(r["gate"]),
                wood=float(r["wood"]), fire=float(r["fire"]),
                earth=float(r["earth"]), metal=float(r["metal"]), water=float(r["water"])
            )
    ganji = _ganji_for_date(d)
    elems = _elements_from_ganji(ganji)
    gate = _gate_from_ganji(ganji)
    return PillarDay(
        date_kst=d, ganji_day=ganji, gate=gate,
        wood=elems["wood"], fire=elems["fire"], earth=elems["earth"],
        metal=elems["metal"], water=elems["water"]
    )
