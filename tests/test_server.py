"""
Tests for the Flask webhook server (server.py).

Uses Flask's test client — no real Twilio calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

import server
from server import app, register_call, get_call_state, pop_call_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SCENARIO = {
    "name": "appointment_scheduling",
    "description": "Scheduling test",
    "persona": "You are a patient named Sarah.",
    "opening": "Hi, I would like to schedule an appointment.",
    "goal": "Schedule an appointment.",
    "max_turns": 5,
    "expected_outcomes": ["appointment confirmed"],
    "red_flags": ["I cannot help"],
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure call registry is empty before each test."""
    server._call_registry.clear()
    yield
    server._call_registry.clear()


def _make_simulator(opening="Hi there.", done=False, respond_reply="Monday please."):
    sim = MagicMock()
    sim.opening_line.return_value = opening
    sim.done = done
    sim.respond.return_value = respond_reply
    return sim


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVoiceStart:

    def test_unknown_call_sid_returns_hangup_twiml(self, client):
        resp = client.post("/voice/start", data={"CallSid": "CA_unknown"})
        assert resp.status_code == 200
        assert b"<Hangup" in resp.data

    def test_known_call_returns_opening_in_twiml(self, client):
        sim = _make_simulator(opening="Hi, I need an appointment.")
        register_call("CA001", sim)

        resp = client.post("/voice/start", data={"CallSid": "CA001"})
        assert resp.status_code == 200
        assert b"Hi, I need an appointment." in resp.data

    def test_opening_line_added_to_transcript(self, client):
        sim = _make_simulator(opening="Hi there.")
        register_call("CA002", sim)

        client.post("/voice/start", data={"CallSid": "CA002"})
        state = get_call_state("CA002")
        assert ("PATIENT", "Hi there.") in state["transcript"]


class TestVoiceRespond:

    def test_agent_speech_added_to_transcript(self, client):
        sim = _make_simulator()
        register_call("CA010", sim)

        client.post(
            "/voice/respond",
            data={"CallSid": "CA010", "SpeechResult": "What day works for you?", "Confidence": "0.9"},
        )
        state = get_call_state("CA010")
        assert ("AGENT", "What day works for you?") in state["transcript"]

    def test_no_speech_prompts_repeat(self, client):
        sim = _make_simulator()
        register_call("CA011", sim)

        resp = client.post("/voice/respond", data={"CallSid": "CA011", "SpeechResult": ""})
        assert b"repeat" in resp.data.lower()

    def test_done_simulator_triggers_hangup(self, client):
        sim = _make_simulator(done=True)
        register_call("CA012", sim)

        resp = client.post(
            "/voice/respond",
            data={"CallSid": "CA012", "SpeechResult": "Is there anything else?"},
        )
        assert b"<Hangup" in resp.data

    def test_unknown_call_sid_returns_hangup(self, client):
        resp = client.post(
            "/voice/respond",
            data={"CallSid": "CA_unknown", "SpeechResult": "Hello"},
        )
        assert b"<Hangup" in resp.data

    def test_patient_reply_added_to_transcript(self, client):
        sim = _make_simulator(respond_reply="Monday afternoon please.")
        register_call("CA013", sim)

        client.post(
            "/voice/respond",
            data={"CallSid": "CA013", "SpeechResult": "What time works?"},
        )
        state = get_call_state("CA013")
        assert ("PATIENT", "Monday afternoon please.") in state["transcript"]


class TestRecordingStatus:

    def test_recording_sid_stored_on_completed(self, client):
        sim = _make_simulator()
        register_call("CA020", sim)

        client.post(
            "/voice/recording_status",
            data={
                "CallSid": "CA020",
                "RecordingSid": "RE123",
                "RecordingStatus": "completed",
            },
        )
        state = get_call_state("CA020")
        assert state["recording_sid"] == "RE123"

    def test_recording_sid_not_stored_when_not_completed(self, client):
        sim = _make_simulator()
        register_call("CA021", sim)

        client.post(
            "/voice/recording_status",
            data={
                "CallSid": "CA021",
                "RecordingSid": "RE456",
                "RecordingStatus": "in-progress",
            },
        )
        state = get_call_state("CA021")
        assert state["recording_sid"] is None

    def test_returns_204_no_content(self, client):
        resp = client.post(
            "/voice/recording_status",
            data={"CallSid": "CA_none", "RecordingSid": "RE789", "RecordingStatus": "completed"},
        )
        assert resp.status_code == 204


class TestVoiceStatus:

    def test_call_status_updated(self, client):
        sim = _make_simulator()
        register_call("CA030", sim)

        client.post("/voice/status", data={"CallSid": "CA030", "CallStatus": "completed"})
        state = get_call_state("CA030")
        assert state["status"] == "completed"

    def test_returns_204_for_unknown_call(self, client):
        resp = client.post("/voice/status", data={"CallSid": "CA_none", "CallStatus": "failed"})
        assert resp.status_code == 204
