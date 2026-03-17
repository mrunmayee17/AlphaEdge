"""Hallucination guardrail — verifies agent claims against tool call trace.

Every tool call result is stored in CommitteeState.trace as a TraceEvent.
This module checks agent Evidence items against the trace to detect fabricated data.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def check_agent_claims(
    agent_view: dict,
    trace_events: list[dict],
    agent_name: str,
) -> list[dict]:
    """Check an agent's evidence against the tool call trace.

    Returns a list of mismatches found. Each mismatch is a dict:
    {
        "metric": str,
        "claimed_value": str,
        "actual_value": str,
        "source_tool": str,
        "severity": "warning" | "error"
    }
    """
    mismatches = []

    evidence_items = agent_view.get("supporting_evidence", [])
    if not evidence_items:
        return mismatches

    # Build lookup from trace: tool_name → result
    agent_trace = [t for t in trace_events if t.get("agent") == agent_name]
    trace_by_tool: dict[str, str] = {}
    for t in agent_trace:
        tool = t.get("tool", "")
        result = t.get("result", "")
        trace_by_tool[tool] = result

    for ev in evidence_items:
        if not isinstance(ev, dict):
            continue

        source_tool = ev.get("source_tool", "")
        metric = ev.get("metric", "")
        claimed_value = str(ev.get("value", ""))

        if not source_tool or not claimed_value:
            continue

        # Find matching trace
        trace_result = trace_by_tool.get(source_tool, "")
        if not trace_result:
            # Tool wasn't called — evidence is fabricated
            mismatches.append({
                "metric": metric,
                "claimed_value": claimed_value,
                "actual_value": "NO TOOL CALL FOUND",
                "source_tool": source_tool,
                "severity": "error",
            })
            continue

        # Check if claimed value appears in trace result
        if not _value_in_result(claimed_value, trace_result):
            mismatches.append({
                "metric": metric,
                "claimed_value": claimed_value,
                "actual_value": trace_result[:200],
                "source_tool": source_tool,
                "severity": "warning",
            })

    if mismatches:
        logger.warning(
            f"Hallucination check for {agent_name}: {len(mismatches)} mismatch(es)"
        )

    return mismatches


def _value_in_result(claimed: str, result: str) -> bool:
    """Check if a claimed value appears in the tool result.

    Handles numeric comparisons with tolerance and string matching.
    """
    claimed = claimed.strip()
    result_lower = result.lower()

    # Direct string match
    if claimed.lower() in result_lower:
        return True

    # Try numeric comparison with tolerance
    try:
        claimed_num = float(re.sub(r"[%$,]", "", claimed))
        # Search for numbers in result that are close
        numbers_in_result = re.findall(r"-?\d+\.?\d*", result)
        for num_str in numbers_in_result:
            try:
                result_num = float(num_str)
                if abs(result_num) < 1e-10:
                    continue
                if abs(claimed_num - result_num) / max(abs(result_num), 1e-8) < 0.05:
                    return True  # Within 5% tolerance
            except ValueError:
                continue
    except ValueError:
        pass

    return False


def build_correction_prompt(mismatches: list[dict], agent_name: str) -> Optional[str]:
    """Build a correction prompt for the agent if mismatches are found.

    Returns None if no correction needed.
    """
    errors = [m for m in mismatches if m["severity"] == "error"]
    warnings = [m for m in mismatches if m["severity"] == "warning"]

    if not errors and not warnings:
        return None

    parts = [f"HALLUCINATION CHECK FAILED for {agent_name}. Corrections needed:"]

    for m in errors:
        parts.append(
            f"- ERROR: You cited {m['source_tool']} for {m['metric']}={m['claimed_value']}, "
            f"but that tool was never called. Remove this evidence."
        )

    for m in warnings:
        parts.append(
            f"- WARNING: For {m['metric']}, you claimed {m['claimed_value']} "
            f"but the tool returned: {m['actual_value']}. Correct your analysis."
        )

    parts.append("\nRevise your AgentView JSON with corrected evidence.")
    return "\n".join(parts)
