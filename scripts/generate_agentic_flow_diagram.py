#!/usr/bin/env python3
"""Generate a lightweight static SVG for the agentic system flow."""

from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "agentic_system_flow.svg"


def box(x, y, w, h, text, fill="#f8fafc"):
    lines = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#334155" stroke-width="1.4"/>'
    ]
    cy = y + h / 2 - (len(text.split("\n")) - 1) * 9
    lines.append(f'<text x="{x + w/2}" y="{cy}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#0f172a">')
    for i, t in enumerate(text.split("\n")):
        dy = 0 if i == 0 else 18
        lines.append(f'<tspan x="{x + w/2}" dy="{dy}">{escape(t)}</tspan>')
    lines.append('</text>')
    return "\n".join(lines)


def arrow(x1, y1, x2, y2, label=""):
    # plain line + small triangle arrowhead (no marker defs)
    parts = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="1.5"/>']
    # arrowhead at end
    if abs(x2 - x1) >= abs(y2 - y1):
        # horizontal-ish
        if x2 >= x1:
            pts = f"{x2},{y2} {x2-8},{y2-4} {x2-8},{y2+4}"
        else:
            pts = f"{x2},{y2} {x2+8},{y2-4} {x2+8},{y2+4}"
    else:
        # vertical-ish
        if y2 >= y1:
            pts = f"{x2},{y2} {x2-4},{y2-8} {x2+4},{y2-8}"
        else:
            pts = f"{x2},{y2} {x2-4},{y2+8} {x2+4},{y2+8}"
    parts.append(f'<polygon points="{pts}" fill="#334155"/>')
    if label:
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2 - 6
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#334155">{escape(label)}</text>')
    return "\n".join(parts)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    w, h = 1800, 760
    s = []
    s.append('<?xml version="1.0" encoding="UTF-8"?>')
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    s.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    s.append('<text x="900" y="42" text-anchor="middle" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#0f172a">Agentic System Flow (Current Implementation)</text>')

    # Top pipeline (left -> right)
    s.append(box(40, 100, 190, 62, 'POST /api/v1/analysis'))
    s.append(box(280, 100, 220, 62, 'Create Redis state\nstatus=pending'))
    s.append(box(550, 100, 220, 62, 'Background pipeline starts'))
    s.append(box(820, 100, 220, 62, 'Round 0: predict_alpha\nstatus=predicting'))
    s.append(box(1100, 60, 220, 56, 'run_chronos_inference', '#dbeafe'))
    s.append(box(1100, 148, 220, 56, 'run_fincast_lora_inference', '#dcfce7'))
    s.append(box(1380, 60, 180, 56, 'chronos ok?', '#e2e8f0'))
    s.append(box(1380, 148, 180, 56, 'fincast ok?', '#e2e8f0'))
    s.append(box(1610, 60, 150, 56, 'alpha stored', '#e2fbe8'))
    s.append(box(1610, 148, 150, 56, 'status=error', '#fee2e2'))

    s.append(arrow(230, 131, 280, 131))
    s.append(arrow(500, 131, 550, 131))
    s.append(arrow(770, 131, 820, 131))
    s.append(arrow(1040, 131, 1100, 88, 'chronos'))
    s.append(arrow(1040, 131, 1100, 176, 'fincast_lora'))
    s.append(arrow(1320, 88, 1380, 88))
    s.append(arrow(1320, 176, 1380, 176))
    s.append(arrow(1560, 88, 1610, 88, 'yes'))
    s.append(arrow(1560, 176, 1610, 176, 'no'))

    # Rounds panel (top -> bottom)
    s.append('<rect x="240" y="280" width="1320" height="430" rx="12" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2"/>')
    s.append('<text x="900" y="314" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">Committee Rounds (top-to-bottom)</text>')

    s.append(box(320, 350, 320, 62, 'Round 1: agent analysis (parallel)\nstatus=round_1', '#dbeafe'))
    s.append(box(320, 435, 320, 62, 'Prefetch tools + trace build'))
    s.append(box(320, 520, 320, 62, 'LLM AgentView + correction pass'))

    s.append(box(760, 350, 260, 62, 'extract_claims'))
    s.append(box(760, 435, 260, 62, 'Round 2: debate\nstatus=round_2', '#dcfce7'))
    s.append(box(760, 520, 260, 62, 'Round 3: memo synthesis\nstatus=round_3', '#dcfce7'))

    s.append(box(1120, 435, 320, 62, 'Persist memo\nstatus=complete', '#e2fbe8'))

    s.append(arrow(1610, 88, 480, 350))
    s.append(arrow(480, 412, 480, 435))
    s.append(arrow(480, 497, 480, 520))
    s.append(arrow(640, 381, 760, 381))
    s.append(arrow(890, 412, 890, 435))
    s.append(arrow(890, 497, 890, 520))
    s.append(arrow(1020, 466, 1120, 466))

    s.append(arrow(770, 162, 1610, 176, 'any unhandled exception'))

    s.append('</svg>')
    OUT.write_text("\n".join(s), encoding='utf-8')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
