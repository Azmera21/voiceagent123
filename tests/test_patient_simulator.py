"""
Tests for the PatientSimulator module.

OpenAI API calls are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from patient_simulator import PatientSimulator


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SCENARIO = {
    "name": "appointment_scheduling",
    "description": "Patient wants to schedule a check-up.",
    "persona": "You are Sarah, a 42-year-old patient wanting a check-up.",
    "opening": "Hi, I'd like to schedule a routine check-up please.",
    "goal": "Schedule an appointment.",
    "max_turns": 3,
    "expected_outcomes": ["appointment confirmed"],
    "red_flags": ["I cannot help"],
}


def _mock_openai(reply: str) -> MagicMock:
    msg = MagicMock()
    msg.content = reply
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp
    return mock_client


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPatientSimulator:

    @patch("patient_simulator.openai.OpenAI")
    def test_opening_line_matches_scenario(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("Some reply")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        assert sim.opening_line() == SAMPLE_SCENARIO["opening"]

    @patch("patient_simulator.openai.OpenAI")
    def test_respond_increments_turn_count(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("I understand, thank you.")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        sim.respond("Hello, how can I help you?")
        assert sim.turn_count == 1

    @patch("patient_simulator.openai.OpenAI")
    def test_respond_returns_non_empty_string(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("Monday afternoon, please.")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        reply = sim.respond("What time works for you?")
        assert isinstance(reply, str)
        assert len(reply) > 0

    @patch("patient_simulator.openai.OpenAI")
    def test_done_after_max_turns(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("Okay thank you.")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        for _ in range(SAMPLE_SCENARIO["max_turns"]):
            sim.respond("Some agent reply.")
        assert sim.done

    @patch("patient_simulator.openai.OpenAI")
    def test_respond_returns_empty_after_done(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("Okay.")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        # exhaust turns
        for _ in range(SAMPLE_SCENARIO["max_turns"]):
            sim.respond("Agent reply.")
        reply = sim.respond("Are you still there?")
        assert reply == ""

    @patch("patient_simulator.openai.OpenAI")
    def test_scenario_complete_token_marks_done(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai(
            "Thank you so much! [SCENARIO COMPLETE]"
        )
        sim = PatientSimulator(SAMPLE_SCENARIO)
        reply = sim.respond("Your appointment is confirmed for Monday at 3 PM.")
        assert sim.done
        # Control token should not appear in returned text
        assert "[SCENARIO COMPLETE]" not in reply

    @patch("patient_simulator.openai.OpenAI")
    def test_scenario_stuck_token_marks_done(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("[SCENARIO STUCK]")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        reply = sim.respond("I'm not sure I can help with that.")
        assert sim.done
        assert "[SCENARIO STUCK]" not in reply

    @patch("patient_simulator.openai.OpenAI")
    def test_conversation_history_grows(self, mock_openai_cls):
        mock_openai_cls.return_value = _mock_openai("Monday please.")
        sim = PatientSimulator(SAMPLE_SCENARIO)
        sim.respond("What day works?")
        history = sim.conversation_history()
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @patch("patient_simulator.openai.OpenAI")
    def test_api_error_returns_fallback(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Network error")
        mock_openai_cls.return_value = mock_client

        sim = PatientSimulator(SAMPLE_SCENARIO)
        reply = sim.respond("Hello!")
        assert isinstance(reply, str)
        assert len(reply) > 0
