#!/usr/bin/env python3
"""Generate a clear static SVG for the agentic system flow."""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "agentic_system_flow_v2.svg"
LEGACY_OUT = ROOT / "docs" / "figures" / "agentic_system_flow.svg"

STROKE = "#111111"
TEXT = "#111111"
BG = "#ffffff"
NODE = "#ffffff"
NODE_MUTED = "#f2f2f2"
NODE_DARK = "#e8e8e8"
PANEL = "#fcfcfc"


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

    w, h = 2200, 1180
    s: list[str] = []
    s.append('<?xml version="1.0" encoding="UTF-8"?>')
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    s.append(f'<rect width="100%" height="100%" fill="{BG}"/>')
    s.append(f'<text x="1100" y="44" text-anchor="middle" font-family="Arial, sans-serif" font-size="42" font-weight="700" fill="{TEXT}">Agentic System Flow (Current Implementation)</text>')

    # --- Top pipeline ---
    y_top = 110
    s.append(box(40, y_top, 240, 72, "POST /api/v1/analysis"))
    s.append(box(330, y_top, 300, 72, "Create Redis state\nstatus=pending"))
    s.append(box(680, y_top, 300, 72, "Background pipeline starts"))
    s.append(box(1030, y_top, 300, 72, "Round 0: predict_alpha\nstatus=predicting"))

    s.append(line_arrow(280, y_top + 36, 330, y_top + 36))
    s.append(line_arrow(630, y_top + 36, 680, y_top + 36))
    s.append(line_arrow(980, y_top + 36, 1030, y_top + 36))

    # Model branch
    s.append(box(1400, 70, 280, 62, "run_chronos_inference", fill=NODE_MUTED))
    s.append(box(1400, 150, 280, 62, "run_fincast_lora_inference", fill=NODE_MUTED))

    s.append(line_arrow(1330, y_top + 20, 1400, 100, "chronos"))
    s.append(line_arrow(1330, y_top + 52, 1400, 181, "fincast_lora"))

    s.append(box(1730, 70, 220, 62, "chronos ok?", fill=NODE_DARK))
    s.append(box(1730, 150, 220, 62, "fincast ok?", fill=NODE_DARK))

    s.append(line_arrow(1680, 101, 1730, 101))
    s.append(line_arrow(1680, 181, 1730, 181))

    s.append(box(1990, 40, 170, 62, "alpha stored", fill=NODE_MUTED))
    s.append(box(1990, 180, 170, 62, "status=error", fill=NODE_DARK))
    s.append(box(1990, 110, 170, 62, "placeholder\nalpha", fill=NODE_MUTED))

    s.append(line_arrow(1950, 101, 1990, 71, "yes"))
    s.append(line_arrow(1950, 101, 1990, 141, "no"))
    s.append(line_arrow(2075, 172, 2075, 102))
    s.append(line_arrow(1950, 181, 1990, 211, "no"))
    s.append(line_arrow(1950, 181, 1990, 71, "yes"))

    # Unhandled exception routed outside to avoid overlaps
    s.append(poly_arrow([(980, y_top + 72), (980, 260), (2075, 260), (2075, 242)], "any unhandled exception", dashed=True))

    # --- Committee rounds panel ---
    panel_x, panel_y, panel_w, panel_h = 140, 320, 1920, 800
    s.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="14" fill="{PANEL}" stroke="{STROKE}" stroke-width="1.2"/>')
    s.append(f'<text x="1100" y="366" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="{TEXT}">Committee Rounds (top-to-bottom)</text>')

    # Round 1 chain
    s.append(box(220, 410, 420, 72, "Round 1: agent analysis (parallel)\nstatus=round_1", fill=NODE_MUTED))
    s.append(box(220, 500, 420, 64, "Prefetch tools + trace build"))
    s.append(box(220, 582, 420, 64, "LLM AgentView JSON + correction pass"))

    s.append(line_arrow(430, 482, 430, 500))
    s.append(line_arrow(430, 564, 430, 582))

    # Agent role boxes (what each agent does)
    agent_y = 690
    aw, ah, gap, start = 350, 250, 20, 180
    s.append(box(start + 0*(aw+gap), agent_y, aw, ah,
                 "Quantitative\n- momentum & volatility\n- alpha signal strength\n- technical context", fill=NODE, fs=12))
    s.append(box(start + 1*(aw+gap), agent_y, aw, ah,
                 "Fundamentals\n- valuation and peers\n- financial health\n- earnings context", fill=NODE, fs=12))
    s.append(box(start + 2*(aw+gap), agent_y, aw, ah,
                 "Sentiment\n- news narrative\n- social/web pulse\n- positioning tone", fill=NODE, fs=12))
    s.append(box(start + 3*(aw+gap), agent_y, aw, ah,
                 "Risk Guardian\n- tail risk / CVaR\n- options positioning\n- stress scenarios", fill=NODE, fs=12))
    s.append(box(start + 4*(aw+gap), agent_y, aw, ah,
                 "Macro Regime\n- rates and curve shape\n- risk-on/off regime\n- cross-asset backdrop", fill=NODE, fs=12))

    # Link round1 to agent roles
    for i in range(5):
        cx = start + i * (aw + gap) + aw / 2
        s.append(poly_arrow([(430, 646), (430, 668), (cx, 668), (cx, agent_y)]))

    # Right-side synthesis chain
    s.append(box(760, 430, 300, 72, "extract_claims"))
    s.append(box(760, 522, 300, 72, "Round 2: debate\nstatus=round_2", fill=NODE_MUTED))
    s.append(box(760, 614, 300, 72, "Round 3: memo synthesis\nstatus=round_3", fill=NODE_MUTED))
    s.append(box(1110, 614, 360, 72, "Persist memo\nstatus=complete", fill=NODE_MUTED))

    s.append(line_arrow(640, 446, 760, 466))
    s.append(line_arrow(910, 502, 910, 522))
    s.append(line_arrow(910, 594, 910, 614))
    s.append(line_arrow(1060, 650, 1110, 650))

    # Entry from alpha stored to round1
    s.append(poly_arrow([(2050, 71), (2050, 290), (430, 290), (430, 410)]))
    s.append(f'<text x="1740" y="1110" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="{TEXT}">Solid arrow: normal path   |   Dashed arrow: exception path</text>')

    s.append('</svg>')
    svg = "\n".join(s)
    OUT.write_text(svg, encoding="utf-8")
    # Keep legacy path in sync for older links and local references.
    LEGACY_OUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
