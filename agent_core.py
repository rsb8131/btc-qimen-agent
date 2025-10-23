# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any
import json
import math

ELEMENTS = ["wood","fire","earth","metal","water"]
GATE_ADJ = {"開": +0.05, "休": +0.02, "生": +0.03, "傷": -0.03, "杜": -0.02, "景": +0.01, "死": -0.05, "驚": -0.04}

@dataclass
class AgentState:
    weights: Dict[str, float]
    bias: float
    lr: float
    ema_acc: float
    ema_alpha: float
    days_seen: int

    @staticmethod
    def load(path: Path) -> "AgentState":
        if path.exists():
            return AgentState(**json.loads(path.read_text()))
        return AgentState(weights={e:0.0 for e in ELEMENTS}, bias=0.0, lr=0.06, ema_acc=0.5, ema_alpha=0.2, days_seen=0)

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

def score_row(state: AgentState, row) -> float:
    s = state.bias
    for e in ELEMENTS:
        s += state.weights.get(e, 0.0) * float(row.get(e, 0.0))
    gate = str(row.get("gate","")).strip()
    s += GATE_ADJ.get(gate, 0.0)
    s += 0.1 * float(row.get("rsi_sig", 0.0))
    return s

def prob_from_score(s: float) -> float:
    return 1.0 / (1.0 + math.exp(-s))

def update_state(state: AgentState, row, prob: float, actual_label: int):
    y = 1.0 if actual_label > 0 else 0.0
    grad = (prob - y)
    # 가중치 업데이트(간단 온라인 로지스틱)
    state.bias -= state.lr * grad * 0.2
    for e in ELEMENTS:
        x = float(row.get(e, 0.0))
        state.weights[e] -= state.lr * grad * x

    # EMA 적중률 업데이트
    pred_label = 1.0 if prob >= 0.5 else 0.0
    hit = 1.0 if pred_label == y else 0.0
    state.ema_acc = (1 - state.ema_alpha) * state.ema_acc + state.ema_alpha * hit
    state.days_seen += 1

def build_report(today_str: str, predict_target: str, verify_target: str, merged_row, prob: float, actual_label: int, state: AgentState) -> str:
    lbl_pred  = "상승" if prob >= 0.5 else "하락"
    lbl_act   = "상승" if actual_label > 0 else "하락"
    hit = "적중 ✅" if ((prob>=0.5 and actual_label>0) or (prob<0.5 and actual_label<0)) else "빗나감 ❌"
    weights_str = " ".join([f"{k}{state.weights.get(k,0.0):+0.02f}" for k in ELEMENTS])

    lines = [
        f"[BTC × 기문‧오행 일일 리포트] {today_str} (KST)",
        f"🔹 예측 대상일: {predict_target}",
        f"🔹 검증 대상일: {verify_target}",
        f"🌿 오행지수: 木{merged_row['wood']:.2f} 火{merged_row['fire']:.2f} 土{merged_row['earth']:.2f} 金{merged_row['metal']:.2f} 水{merged_row['water']:.2f} · 門:{merged_row['gate']}",
        f"📈 RSI14:{float(merged_row.get('rsi14', float('nan'))):.1f} · 보조시그널:{int(merged_row.get('rsi_sig',0))}",
        f"🎯 오늘 예측: {lbl_pred} (확률 {prob*100:.1f}%)",
        f"🧪 전일 결과: {lbl_act} → {hit}",
        f"📊 EMA 적중률(최근 30일): {state.ema_acc*100:.1f}%",
        f"⚙️ 가중치: {weights_str} · bias:{state.bias:+0.02f} · lr:{state.lr}",
        "🕐 규칙: KST 00:00–24:00, UTC→KST 보정, 月=中氣, 年=立春"
    ]
    return "\n".join(lines)

