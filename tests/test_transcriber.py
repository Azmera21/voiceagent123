"""
Tests for the transcriber module (no network calls).
"""

import pytest

from transcriber import RecordingManager, _parse_transcription_text


class TestParseTranscriptionText:

    def test_empty_string_returns_empty_list(self):
        assert _parse_transcription_text("") == []

    def test_single_sentence_attributed_to_agent(self):
        result = _parse_transcription_text("Hello, how can I help you today.")
        assert result[0][0] == "AGENT"

    def test_alternates_speakers(self):
        raw = "Hello, how can I help. I need an appointment. Let me check. Great."
        result = _parse_transcription_text(raw)
        speakers = [r[0] for r in result]
        assert speakers[0] == "AGENT"
        assert speakers[1] == "PATIENT"
        assert speakers[2] == "AGENT"

    def test_sentences_end_with_period(self):
        raw = "Hello, how can I help. I need to refill my prescription."
        result = _parse_transcription_text(raw)
        for _, text in result:
            assert text.endswith(".")

    def test_whitespace_only_sentences_excluded(self):
        raw = "Hello.  .  How can I help."
        result = _parse_transcription_text(raw)
        for _, text in result:
            assert text.strip()


class TestBuildTranscriptFromHistory:

    def test_user_role_maps_to_agent(self):
        history = [{"role": "user", "content": "Your appointment is confirmed."}]
        rm = RecordingManager.__new__(RecordingManager)
        transcript = rm.build_transcript_from_history(history)
        assert transcript == [("AGENT", "Your appointment is confirmed.")]

    def test_assistant_role_maps_to_patient(self):
        history = [{"role": "assistant", "content": "Thank you very much."}]
        rm = RecordingManager.__new__(RecordingManager)
        transcript = rm.build_transcript_from_history(history)
        assert transcript == [("PATIENT", "Thank you very much.")]

    def test_control_tokens_excluded(self):
        history = [
            {"role": "assistant", "content": "[SCENARIO COMPLETE]"},
            {"role": "assistant", "content": "[SCENARIO STUCK]"},
        ]
        rm = RecordingManager.__new__(RecordingManager)
        transcript = rm.build_transcript_from_history(history)
        assert transcript == []

    def test_empty_content_excluded(self):
        history = [{"role": "user", "content": "  "}]
        rm = RecordingManager.__new__(RecordingManager)
        transcript = rm.build_transcript_from_history(history)
        assert transcript == []

    def test_full_conversation_history(self):
        history = [
            {"role": "user", "content": "Hello, how can I help?"},
            {"role": "assistant", "content": "I need an appointment."},
            {"role": "user", "content": "What day works for you?"},
            {"role": "assistant", "content": "Monday afternoon."},
        ]
        rm = RecordingManager.__new__(RecordingManager)
        transcript = rm.build_transcript_from_history(history)
        assert transcript == [
            ("AGENT", "Hello, how can I help?"),
            ("PATIENT", "I need an appointment."),
            ("AGENT", "What day works for you?"),
            ("PATIENT", "Monday afternoon."),
        ]
