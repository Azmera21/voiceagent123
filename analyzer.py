"""
Transcript analysis module.

After each call the raw transcript (a list of (speaker, text) tuples) is
passed to the Analyzer, which uses OpenAI to detect bugs and quality issues
in the *agent's* responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

import config

logger = logging.getLogger(__name__)

# ── Analysis result schema ────────────────────────────────────────────────────
#
# The LLM is asked to return a JSON object matching this structure:
#
#   {
#     "overall_quality": "good" | "acceptable" | "poor",
#     "issues": [
#       {
#         "severity": "critical" | "high" | "medium" | "low",
#         "category": str,          # e.g. "incorrect_information", "unhelpful_response"
#         "description": str,       # plain English explanation
#         "agent_quote": str        # the problematic agent utterance
#       }
#     ],
#     "summary": str,
#     "recommendations": [str]
#   }

_ANALYSIS_SYSTEM_PROMPT = """
You are a quality-assurance specialist evaluating a healthcare voice agent.
You will receive:
1. A scenario description explaining what the patient was trying to accomplish.
2. The full transcript of the call (each line is "PATIENT: ..." or "AGENT: ...").
3. A list of expected outcomes and red-flag phrases for this scenario.

Analyse ONLY the AGENT's responses for the following issue categories:
- incorrect_information: Agent gave factually wrong or potentially dangerous information.
- unhelpful_response: Agent failed to progress the patient toward their goal.
- poor_empathy: Agent was robotic, dismissive, or insensitive.
- missing_verification: Agent failed to verify identity when required (e.g., DOB, name).
- escalation_failure: Agent did not escalate an urgent issue (e.g., chest pain → 911).
- red_flag_phrase: Agent used one of the listed red-flag phrases.
- process_error: Agent followed an incorrect process (e.g., wrong department transfer).
- incomplete_resolution: Scenario ended without the patient's goal being met.

Return a single valid JSON object with NO markdown fencing, matching this schema:
{
  "overall_quality": "good" | "acceptable" | "poor",
  "issues": [
    {
      "severity": "critical" | "high" | "medium" | "low",
      "category": "<one of the categories above>",
      "description": "<plain English explanation>",
      "agent_quote": "<exact agent utterance that caused the issue>"
    }
  ],
  "summary": "<2-3 sentence summary of the agent's performance>",
  "recommendations": ["<actionable improvement>"]
}
""".strip()


class Analyzer:
    """Analyses a call transcript for agent quality issues."""

    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def analyze(
        self,
        scenario: dict[str, Any],
        transcript: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """
        Analyse the transcript for the given scenario.

        Parameters
        ----------
        scenario:
            The scenario dict from scenarios.py.
        transcript:
            List of (speaker, text) tuples where speaker is "PATIENT" or "AGENT".

        Returns
        -------
        dict
            Parsed JSON analysis result (see schema above).
        """
        transcript_text = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)

        user_message = (
            f"SCENARIO: {scenario['description']}\n"
            f"PATIENT GOAL: {scenario['goal']}\n\n"
            f"EXPECTED OUTCOMES: {', '.join(scenario.get('expected_outcomes', []))}\n"
            f"RED-FLAG PHRASES: {', '.join(scenario.get('red_flags', []))}\n\n"
            f"TRANSCRIPT:\n{transcript_text}"
        )

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=1000,
                temperature=0.2,
            )
            raw = response.choices[0].message.content or "{}"
            result: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse analysis JSON: %s", exc)
            result = _error_result(f"JSON parse error: {exc}")
        except Exception as exc:
            logger.error("OpenAI analysis error: %s", exc)
            result = _error_result(str(exc))

        # Supplement with rule-based red-flag detection
        result = _apply_red_flag_rules(result, scenario, transcript)

        return result

    def format_report(
        self,
        scenario: dict[str, Any],
        analysis: dict[str, Any],
        call_sid: str = "",
    ) -> str:
        """Return a human-readable report string."""
        lines: list[str] = [
            "=" * 70,
            f"SCENARIO : {scenario['name']}",
            f"CALL SID : {call_sid or 'N/A'}",
            f"QUALITY  : {analysis.get('overall_quality', 'unknown').upper()}",
            "-" * 70,
            f"SUMMARY  : {analysis.get('summary', '')}",
            "",
        ]

        issues = analysis.get("issues", [])
        if issues:
            lines.append(f"ISSUES ({len(issues)}):")
            for issue in issues:
                lines.append(
                    f"  [{issue.get('severity', '?').upper()}] "
                    f"{issue.get('category', '?')} — "
                    f"{issue.get('description', '')}"
                )
                quote = issue.get("agent_quote", "")
                if quote:
                    lines.append(f'    Agent said: "{quote}"')
        else:
            lines.append("ISSUES  : None detected.")

        recs = analysis.get("recommendations", [])
        if recs:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in recs:
                lines.append(f"  • {rec}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_result(message: str) -> dict[str, Any]:
    return {
        "overall_quality": "unknown",
        "issues": [],
        "summary": f"Analysis could not be completed: {message}",
        "recommendations": [],
    }


def _apply_red_flag_rules(
    result: dict[str, Any],
    scenario: dict[str, Any],
    transcript: list[tuple[str, str]],
) -> dict[str, Any]:
    """
    Deterministically scan the agent's lines for red-flag phrases and add any
    that the LLM may have missed.
    """
    agent_lines = [text for speaker, text in transcript if speaker == "AGENT"]
    full_agent_text = " ".join(agent_lines).lower()

    existing_quotes = {
        issue.get("agent_quote", "").lower()
        for issue in result.get("issues", [])
    }

    for flag in scenario.get("red_flags", []):
        if flag.lower() in full_agent_text:
            # Find which line contains it
            matching_quote = next(
                (line for line in agent_lines if flag.lower() in line.lower()),
                flag,
            )
            if matching_quote.lower() not in existing_quotes:
                result.setdefault("issues", []).append(
                    {
                        "severity": "high",
                        "category": "red_flag_phrase",
                        "description": (
                            f"Agent used a known red-flag phrase: '{flag}'"
                        ),
                        "agent_quote": matching_quote,
                    }
                )
                existing_quotes.add(matching_quote.lower())

    # Downgrade quality if there are critical/high issues
    high_or_critical = [
        i
        for i in result.get("issues", [])
        if i.get("severity") in ("critical", "high")
    ]
    if high_or_critical and result.get("overall_quality") == "good":
        result["overall_quality"] = "acceptable"

    return result
