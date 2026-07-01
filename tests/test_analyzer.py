"""
Tests for the Analyzer module.

OpenAI API calls are mocked so these tests run without credentials.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from analyzer import Analyzer, _apply_red_flag_rules, _error_result


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SCENARIO = {
    "name": "test_scenario",
    "description": "Test scenario",
    "goal": "Reach a resolution",
    "expected_outcomes": ["appointment confirmed", "callback"],
    "red_flags": ["I cannot help", "call back tomorrow"],
}

GOOD_TRANSCRIPT = [
    ("PATIENT", "Hi, I'd like to schedule an appointment."),
    ("AGENT", "Of course! I can help with that. What day works for you?"),
    ("PATIENT", "Monday afternoon if possible."),
    ("AGENT", "Great, I have Monday at 3 PM available. Your appointment is confirmed."),
    ("PATIENT", "Perfect, thank you!"),
]

BAD_TRANSCRIPT = [
    ("PATIENT", "Hi, I'd like to schedule an appointment."),
    ("AGENT", "I cannot help with scheduling. Please call back tomorrow."),
    ("PATIENT", "Oh, okay..."),
]


def _mock_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnalyzer:

    @patch("analyzer.openai.OpenAI")
    def test_analyse_returns_dict_with_required_keys(self, mock_openai_cls):
        good_result = {
            "overall_quality": "good",
            "issues": [],
            "summary": "Agent performed well.",
            "recommendations": [],
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(good_result)
        )
        mock_openai_cls.return_value = mock_client

        analyzer = Analyzer()
        result = analyzer.analyse(SAMPLE_SCENARIO, GOOD_TRANSCRIPT)

        assert "overall_quality" in result
        assert "issues" in result
        assert "summary" in result
        assert "recommendations" in result

    @patch("analyzer.openai.OpenAI")
    def test_analyse_detects_red_flags(self, mock_openai_cls):
        # LLM returns no issues, but rule-based check should catch red flags
        clean_result = {
            "overall_quality": "good",
            "issues": [],
            "summary": "Agent performed well.",
            "recommendations": [],
        }
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(clean_result)
        )
        mock_openai_cls.return_value = mock_client

        analyzer = Analyzer()
        result = analyzer.analyse(SAMPLE_SCENARIO, BAD_TRANSCRIPT)

        issue_categories = [i["category"] for i in result["issues"]]
        assert "red_flag_phrase" in issue_categories

    @patch("analyzer.openai.OpenAI")
    def test_analyse_handles_json_decode_error(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "This is not valid JSON"
        )
        mock_openai_cls.return_value = mock_client

        analyzer = Analyzer()
        result = analyzer.analyse(SAMPLE_SCENARIO, GOOD_TRANSCRIPT)

        assert result["overall_quality"] == "unknown"
        assert "Analysis could not be completed" in result["summary"]

    @patch("analyzer.openai.OpenAI")
    def test_analyse_handles_api_exception(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_openai_cls.return_value = mock_client

        analyzer = Analyzer()
        result = analyzer.analyse(SAMPLE_SCENARIO, GOOD_TRANSCRIPT)

        assert "Analysis could not be completed" in result["summary"]

    @patch("analyzer.openai.OpenAI")
    def test_format_report_contains_scenario_name(self, mock_openai_cls):
        analysis = {
            "overall_quality": "good",
            "issues": [],
            "summary": "Great performance.",
            "recommendations": ["Keep it up"],
        }
        mock_openai_cls.return_value = MagicMock()

        analyzer = Analyzer()
        report = analyzer.format_report(SAMPLE_SCENARIO, analysis, call_sid="CA123")

        assert "test_scenario" in report
        assert "CA123" in report
        assert "GOOD" in report
        assert "Great performance." in report
        assert "Keep it up" in report

    @patch("analyzer.openai.OpenAI")
    def test_format_report_lists_issues(self, mock_openai_cls):
        analysis = {
            "overall_quality": "poor",
            "issues": [
                {
                    "severity": "high",
                    "category": "unhelpful_response",
                    "description": "Agent failed to help.",
                    "agent_quote": "I cannot assist.",
                }
            ],
            "summary": "Poor performance.",
            "recommendations": [],
        }
        mock_openai_cls.return_value = MagicMock()

        analyzer = Analyzer()
        report = analyzer.format_report(SAMPLE_SCENARIO, analysis)

        assert "unhelpful_response" in report
        assert "I cannot assist." in report


# ── Unit tests for helper functions ───────────────────────────────────────────

class TestApplyRedFlagRules:

    def test_adds_red_flag_issue_when_found(self):
        result = {"overall_quality": "good", "issues": [], "summary": "", "recommendations": []}
        transcript = [("AGENT", "I cannot help you with that right now.")]
        updated = _apply_red_flag_rules(result, SAMPLE_SCENARIO, transcript)
        assert any(i["category"] == "red_flag_phrase" for i in updated["issues"])

    def test_does_not_duplicate_existing_issue(self):
        existing_issue = {
            "severity": "high",
            "category": "red_flag_phrase",
            "description": "Red flag found.",
            "agent_quote": "I cannot help you with that right now.",
        }
        result = {
            "overall_quality": "good",
            "issues": [existing_issue],
            "summary": "",
            "recommendations": [],
        }
        transcript = [("AGENT", "I cannot help you with that right now.")]
        updated = _apply_red_flag_rules(result, SAMPLE_SCENARIO, transcript)
        red_flag_issues = [i for i in updated["issues"] if i["category"] == "red_flag_phrase"]
        assert len(red_flag_issues) == 1

    def test_downgrades_quality_when_high_issue_present(self):
        result = {
            "overall_quality": "good",
            "issues": [
                {
                    "severity": "high",
                    "category": "red_flag_phrase",
                    "description": "...",
                    "agent_quote": "call back tomorrow",
                }
            ],
            "summary": "",
            "recommendations": [],
        }
        transcript = [("AGENT", "Please call back tomorrow.")]
        updated = _apply_red_flag_rules(result, SAMPLE_SCENARIO, transcript)
        assert updated["overall_quality"] == "acceptable"

    def test_no_red_flags_leaves_result_unchanged(self):
        result = {"overall_quality": "good", "issues": [], "summary": "", "recommendations": []}
        transcript = [("AGENT", "Your appointment is confirmed for Monday at 3 PM.")]
        updated = _apply_red_flag_rules(result, SAMPLE_SCENARIO, transcript)
        assert updated["issues"] == []
        assert updated["overall_quality"] == "good"


class TestErrorResult:

    def test_error_result_structure(self):
        r = _error_result("something went wrong")
        assert r["overall_quality"] == "unknown"
        assert r["issues"] == []
        assert "something went wrong" in r["summary"]
        assert r["recommendations"] == []
