#!/usr/bin/env python3
"""Generate a clear static SVG for the agentic system flow."""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "agentic_system_flow_v2.svg"
LEGACY_OUT = ROOT / "docs" / "figures" / "agentic_system_flow.svg"

STROKE = "#334155"
TEXT = "#0f172a"
BG = "#ffffff"
NODE = "#f8fafc"
PANEL = "#f8fafc"
MODEL_NODE = "#dbeafe"
MODEL_NODE_ALT = "#dcfce7"
DECISION_NODE = "#e2e8f0"
SUCCESS_NODE = "#e2fbe8"
ERROR_NODE = "#fee2e2"
PLACEHOLDER_NODE = "#fff7ed"
ROUND_NODE = "#dbeafe"
ROUND_NEXT_NODE = "#dcfce7"
AGENT_QUANT = "#e0f2fe"
AGENT_FUND = "#dcfce7"
AGENT_SENT = "#fef3c7"
AGENT_RISK = "#fee2e2"
AGENT_MACRO = "#ede9fe"


def box(x: float, y: float, w: float, h: float, text: str, fill: str = NODE, fs: int = 13) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{STROKE}" stroke-width="1.4"/>'
    ]
    lines = text.split("\n")
    y0 = y + h / 2 - (len(lines) - 1) * (fs * 0.6)
    parts.append(f'<text x="{x + w/2}" y="{y0}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{fs}" fill="{TEXT}">')
    for i, t in enumerate(lines):
        dy = 0 if i == 0 else int(fs * 1.35)
        parts.append(f'<tspan x="{x + w/2}" dy="{dy}">{escape(t)}</tspan>')
    parts.append("</text>")
    return "\n".join(parts)


def _arrow_head(x1: float, y1: float, x2: float, y2: float) -> str:
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):
        if dx >= 0:
            pts = f"{x2},{y2} {x2-9},{y2-4.5} {x2-9},{y2+4.5}"
        else:
            pts = f"{x2},{y2} {x2+9},{y2-4.5} {x2+9},{y2+4.5}"
    else:
        if dy >= 0:
            pts = f"{x2},{y2} {x2-4.5},{y2-9} {x2+4.5},{y2-9}"
        else:
            pts = f"{x2},{y2} {x2-4.5},{y2+9} {x2+4.5},{y2+9}"
    return f'<polygon points="{pts}" fill="{STROKE}"/>'


def line_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "", dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    parts = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{STROKE}" stroke-width="1.6"{dash}/>']
    parts.append(_arrow_head(x1, y1, x2, y2))
    if label:
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 7
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{TEXT}">{escape(label)}</text>')
    return "\n".join(parts)


def poly_arrow(points: list[tuple[float, float]], label: str = "", dashed: bool = False) -> str:
    if len(points) < 2:
        return ""
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    parts = [f'<polyline points="{pts}" fill="none" stroke="{STROKE}" stroke-width="1.6"{dash}/>']
    (x1, y1), (x2, y2) = points[-2], points[-1]
    parts.append(_arrow_head(x1, y1, x2, y2))
    if label:
        # place label near first bend / midpoint
        mid = points[len(points)//2]
        parts.append(f'<text x="{mid[0]}" y="{mid[1]-8}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{TEXT}">{escape(label)}</text>')
    return "\n".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1900, 2200
    s: list[str] = []
    s.append('<?xml version="1.0" encoding="UTF-8"?>')
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    s.append(f'<rect width="100%" height="100%" fill="{BG}"/>')
    s.append(f'<text x="950" y="48" text-anchor="middle" font-family="Arial, sans-serif" font-size="40" font-weight="700" fill="{TEXT}">Agentic System Flow (Current Implementation)</text>')
    s.append(f'<text x="950" y="82" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="{TEXT}">Primary execution path is top-to-bottom</text>')

    # --- Top-to-bottom pipeline ---
    cx = 950
    main_w = 430
    main_h = 72
    y_post = 120
    y_redis = 220
    y_bg = 320
    y_r0 = 420
    y_dispatch = 520

    s.append(box(cx - main_w / 2, y_post, main_w, main_h, "POST /api/v1/analysis"))
    s.append(box(cx - main_w / 2, y_redis, main_w, main_h, "Create Redis state\nstatus=pending"))
    s.append(box(cx - main_w / 2, y_bg, main_w, main_h, "Background pipeline starts"))
    s.append(box(cx - main_w / 2, y_r0, main_w, main_h, "Round 0: predict_alpha\nstatus=predicting"))
    s.append(box(cx - main_w / 2, y_dispatch, main_w, 64, "Dispatch forecast_model branch"))

    s.append(line_arrow(cx, y_post + main_h, cx, y_redis))
    s.append(line_arrow(cx, y_redis + main_h, cx, y_bg))
    s.append(line_arrow(cx, y_bg + main_h, cx, y_r0))
    s.append(line_arrow(cx, y_r0 + main_h, cx, y_dispatch))

    # --- Forecast model branch (still top-to-bottom in each column) ---
    branch_w = 330
    branch_h = 64
    left_x = 490
    right_x = 1080
    y_run = 630
    y_ok = 710
    y_out = 790
    y_merge = 900
    y_round1 = 1020

    s.append(box(left_x, y_run, branch_w, branch_h, "run_chronos_inference", fill=MODEL_NODE))
    s.append(box(right_x, y_run, branch_w, branch_h, "run_fincast_lora_inference", fill=MODEL_NODE_ALT))
    s.append(box(left_x, y_ok, branch_w, branch_h, "chronos ok?", fill=DECISION_NODE))
    s.append(box(right_x, y_ok, branch_w, branch_h, "fincast ok?", fill=DECISION_NODE))

    s.append(poly_arrow([(cx, y_dispatch + 64), (cx, 605), (left_x + branch_w / 2, 605), (left_x + branch_w / 2, y_run)], "chronos"))
    s.append(poly_arrow([(cx, y_dispatch + 64), (cx, 605), (right_x + branch_w / 2, 605), (right_x + branch_w / 2, y_run)], "fincast_lora"))

    s.append(line_arrow(left_x + branch_w / 2, y_run + branch_h, left_x + branch_w / 2, y_ok))
    s.append(line_arrow(right_x + branch_w / 2, y_run + branch_h, right_x + branch_w / 2, y_ok))

    s.append(box(730, y_out, 440, 64, "alpha stored", fill=SUCCESS_NODE))
    s.append(box(350, y_out, 250, 64, "placeholder\nalpha", fill=PLACEHOLDER_NODE))
    s.append(box(1300, y_out, 250, 64, "status=error", fill=ERROR_NODE))

    s.append(poly_arrow([(left_x + branch_w / 2, y_ok + branch_h), (left_x + branch_w / 2, y_out - 16), (950, y_out - 16), (950, y_out)], "yes"))
    s.append(poly_arrow([(left_x + branch_w / 2, y_ok + branch_h), (left_x + branch_w / 2, y_out - 16), (475, y_out - 16), (475, y_out)], "no"))
    s.append(poly_arrow([(475, y_out + 64), (475, y_merge), (950, y_merge), (950, y_out + 64)]))

    s.append(poly_arrow([(right_x + branch_w / 2, y_ok + branch_h), (right_x + branch_w / 2, y_out - 16), (950, y_out - 16), (950, y_out)], "yes"))
    s.append(poly_arrow([(right_x + branch_w / 2, y_ok + branch_h), (right_x + branch_w / 2, y_out - 16), (1425, y_out - 16), (1425, y_out)], "no"))

    # Unhandled exception route
    s.append(poly_arrow([(cx, y_r0 + main_h), (cx, y_out - 36), (1425, y_out - 36), (1425, y_out)], "any unhandled exception", dashed=True))

    s.append(box(730, y_merge, 440, 64, "alpha payload ready"))
    s.append(line_arrow(950, y_out + 64, 950, y_merge))

    s.append(box(cx - main_w / 2, y_round1, main_w, main_h, "Round 1: agent analysis (parallel)\nstatus=round_1", fill=ROUND_NODE))
    s.append(line_arrow(950, y_merge + 64, 950, y_round1))

    # --- Committee rounds (top-to-bottom) ---
    panel_x, panel_y, panel_w, panel_h = 60, 980, 1780, 1180
    s.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="14" fill="{PANEL}" stroke="{STROKE}" stroke-width="1.2"/>')
    s.append(f'<text x="950" y="1060" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="{TEXT}">Committee Rounds (top-to-bottom)</text>')

    y_prefetch = 1120
    y_agentview = 1210
    s.append(box(cx - main_w / 2, y_prefetch, main_w, 64, "Prefetch tools + trace build"))
    s.append(box(cx - main_w / 2, y_agentview, main_w, 64, "LLM AgentView JSON + correction pass"))
    s.append(line_arrow(950, y_round1 + main_h, 950, y_prefetch))
    s.append(line_arrow(950, y_prefetch + 64, 950, y_agentview))

    # Agent role boxes
    agent_y = 1320
    aw = 320
    ah = 260
    gap = 25
    start = 90
    agent_centers: list[float] = []
    agent_boxes = [
        ("Quantitative\n- momentum & volatility\n- alpha signal strength\n- technical context", AGENT_QUANT),
        ("Fundamentals\n- valuation and peers\n- financial health\n- earnings context", AGENT_FUND),
        ("Sentiment\n- news narrative\n- social/web pulse\n- positioning tone", AGENT_SENT),
        ("Risk Guardian\n- tail risk / CVaR\n- options positioning\n- stress scenarios", AGENT_RISK),
        ("Macro Regime\n- rates and curve shape\n- risk-on/off regime\n- cross-asset backdrop", AGENT_MACRO),
    ]
    for i, (label, fill) in enumerate(agent_boxes):
        x = start + i * (aw + gap)
        agent_centers.append(x + aw / 2)
        s.append(box(x, agent_y, aw, ah, label, fill=fill, fs=12))
        s.append(poly_arrow([(950, y_agentview + 64), (950, 1300), (x + aw / 2, 1300), (x + aw / 2, agent_y)]))

    # Claims extraction + rounds 2/3 + persist
    y_extract = 1620
    y_r2 = 1720
    y_r3 = 1820
    y_persist = 1920
    s.append(box(cx - main_w / 2, y_extract, main_w, main_h, "extract_claims"))
    s.append(box(cx - main_w / 2, y_r2, main_w, main_h, "Round 2: debate\nstatus=round_2", fill=ROUND_NEXT_NODE))
    s.append(box(cx - main_w / 2, y_r3, main_w, main_h, "Round 3: memo synthesis\nstatus=round_3", fill=ROUND_NEXT_NODE))
    s.append(box(cx - main_w / 2, y_persist, main_w, main_h, "Persist memo\nstatus=complete", fill=SUCCESS_NODE))

    for c in agent_centers:
        s.append(poly_arrow([(c, agent_y + ah), (c, y_extract - 20), (950, y_extract - 20), (950, y_extract)]))

    s.append(line_arrow(950, y_extract + main_h, 950, y_r2))
    s.append(line_arrow(950, y_r2 + main_h, 950, y_r3))
    s.append(line_arrow(950, y_r3 + main_h, 950, y_persist))

    s.append(f'<text x="1800" y="2160" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="{TEXT}">Solid arrow: normal path   |   Dashed arrow: exception path</text>')

    s.append('</svg>')
    svg = "\n".join(s)
    OUT.write_text(svg, encoding="utf-8")
    # Keep legacy path in sync for older links and local references.
    LEGACY_OUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
