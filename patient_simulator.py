"""
AI patient simulation module.

Uses the OpenAI Chat Completions API to generate contextually appropriate
patient responses during a call, keeping track of the full conversation
history so the simulated patient stays in character across multiple turns.
"""

from __future__ import annotations

import logging
from typing import Any

import openai

import config

logger = logging.getLogger(__name__)


class PatientSimulator:
    """Drives one patient persona through a conversation scenario."""

    # System instructions shared by every patient instance
    _BASE_SYSTEM_SUFFIX = (
        "\n\nIMPORTANT RULES:\n"
        "- Keep your replies SHORT (1-3 sentences) as you would in a real phone call.\n"
        "- Do NOT describe actions (e.g. '*sighs*'); just speak naturally.\n"
        "- When you believe your goal has been met say EXACTLY: [SCENARIO COMPLETE]\n"
        "- If after several turns nothing is progressing, say EXACTLY: [SCENARIO STUCK]\n"
        "- Never reveal that you are an AI or a testing bot.\n"
    )

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self._history: list[dict[str, str]] = []
        self._turn_count: int = 0
        self._done: bool = False

        system_prompt = scenario["persona"] + self._BASE_SYSTEM_SUFFIX
        self._system_message = {"role": "system", "content": system_prompt}

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def done(self) -> bool:
        return self._done

    def opening_line(self) -> str:
        """Return the scripted opening line for this scenario."""
        return self.scenario["opening"]

    def respond(self, agent_utterance: str) -> str:
        """
        Given the agent's last utterance, return the patient's next reply.

        Returns an empty string if the scenario is already complete.
        """
        if self._done:
            return ""

        self._turn_count += 1
        self._history.append({"role": "user", "content": agent_utterance})

        # Check whether we've hit the soft turn limit
        if self._turn_count >= self.scenario.get("max_turns", 10):
            self._done = True
            self._history.append({"role": "assistant", "content": "[SCENARIO COMPLETE]"})
            return "Thank you for your help. Goodbye."

        messages = [self._system_message] + self._history

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,  # type: ignore[arg-type]
                max_tokens=150,
                temperature=0.7,
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            reply = "I'm sorry, could you repeat that?"

        self._history.append({"role": "assistant", "content": reply})

        if "[SCENARIO COMPLETE]" in reply or "[SCENARIO STUCK]" in reply:
            self._done = True
            # Return a polite goodbye without the control token
            return "Thank you very much for your help. Goodbye."

        return reply

    def conversation_history(self) -> list[dict[str, str]]:
        """Return a copy of the full conversation history (excluding system msg)."""
        return list(self._history)
