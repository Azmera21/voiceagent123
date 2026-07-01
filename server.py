"""
Flask webhook server for Twilio call flow.

Twilio makes HTTP requests to these endpoints during the call to drive the
conversation.  The server maintains per-call state (scenario + simulator) in a
simple in-memory registry (sufficient for sequential test calls; for
concurrent calls a persistent store such as Redis would be required).

Endpoints
---------
POST /voice/start
    Called when the outbound call is answered.  Plays the opening patient
    line and sets up speech gathering for the agent's first response.

POST /voice/respond
    Called after Twilio has gathered the agent's speech (or after a timeout).
    Feeds the agent's words to the PatientSimulator, generates the next patient
    reply, and continues the conversation loop.

POST /voice/recording_status
    Called by Twilio when a recording is complete (status callback).

POST /voice/status
    Called by Twilio when the call status changes (e.g., completed, failed).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from flask import Flask, Response, request
from twilio.twiml.voice_response import Gather, VoiceResponse

import config
from patient_simulator import PatientSimulator

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── In-memory call state registry ─────────────────────────────────────────────
# call_sid → {"simulator": PatientSimulator, "transcript": [...], ...}
_call_registry: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()


def register_call(call_sid: str, simulator: PatientSimulator) -> None:
    """Register a new call and its simulator before the call starts."""
    with _registry_lock:
        _call_registry[call_sid] = {
            "simulator": simulator,
            "transcript": [],
            "recording_sid": None,
            "status": "in_progress",
        }


def get_call_state(call_sid: str) -> dict[str, Any] | None:
    with _registry_lock:
        return _call_registry.get(call_sid)


def pop_call_state(call_sid: str) -> dict[str, Any] | None:
    with _registry_lock:
        return _call_registry.pop(call_sid, None)


# ── TwiML helpers ─────────────────────────────────────────────────────────────

def _twiml_say_and_listen(text: str) -> Response:
    """Speak *text* then open a Gather (speech input) block."""
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action=f"{config.WEBHOOK_BASE_URL}/voice/respond",
        method="POST",
        speech_timeout="auto",
        timeout=10,
        language="en-US",
    )
    gather.say(text, voice="Polly.Joanna")
    vr.append(gather)
    # If no speech is detected, redirect back to the respond endpoint anyway
    vr.redirect(f"{config.WEBHOOK_BASE_URL}/voice/respond", method="POST")
    return Response(str(vr), mimetype="text/xml")


def _twiml_hangup(final_message: str = "") -> Response:
    """Optionally say a final message and hang up."""
    vr = VoiceResponse()
    if final_message:
        vr.say(final_message, voice="Polly.Joanna")
    vr.hangup()
    return Response(str(vr), mimetype="text/xml")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/voice/start")
def voice_start() -> Response:
    """
    Entry point when the call is answered.
    Twilio sends: CallSid, CallStatus, etc.
    """
    call_sid = request.form.get("CallSid", "")
    logger.info("[%s] Call answered.", call_sid)

    state = get_call_state(call_sid)
    if state is None:
        logger.error("[%s] Unknown CallSid — hanging up.", call_sid)
        return _twiml_hangup()

    simulator: PatientSimulator = state["simulator"]
    opening = simulator.opening_line()
    state["transcript"].append(("PATIENT", opening))
    logger.info("[%s] Patient says: %s", call_sid, opening)

    return _twiml_say_and_listen(opening)


@app.post("/voice/respond")
def voice_respond() -> Response:
    """
    Handle the agent's speech input and generate the next patient response.
    """
    call_sid = request.form.get("CallSid", "")
    agent_speech = request.form.get("SpeechResult", "").strip()
    confidence = request.form.get("Confidence", "?")

    logger.info(
        "[%s] Agent said (confidence=%s): %s", call_sid, confidence, agent_speech
    )

    state = get_call_state(call_sid)
    if state is None:
        logger.warning("[%s] State not found — hanging up.", call_sid)
        return _twiml_hangup()

    simulator: PatientSimulator = state["simulator"]

    # Record what the agent said
    if agent_speech:
        state["transcript"].append(("AGENT", agent_speech))

    # If the simulator is done, hang up gracefully
    if simulator.done:
        logger.info("[%s] Scenario complete — hanging up.", call_sid)
        return _twiml_hangup("Thank you. Goodbye.")

    # Generate next patient reply
    if not agent_speech:
        # No speech detected; prompt the agent to continue
        patient_reply = "Sorry, I didn't catch that. Could you repeat, please?"
    else:
        patient_reply = simulator.respond(agent_speech)

    state["transcript"].append(("PATIENT", patient_reply))
    logger.info("[%s] Patient says: %s", call_sid, patient_reply)

    # Check again after respond() in case max_turns was hit
    if simulator.done:
        return _twiml_hangup(patient_reply)

    return _twiml_say_and_listen(patient_reply)


@app.post("/voice/recording_status")
def voice_recording_status() -> Response:
    """
    Twilio calls this when a recording status changes.
    We capture the recording SID for later download.
    """
    call_sid = request.form.get("CallSid", "")
    recording_sid = request.form.get("RecordingSid", "")
    recording_status = request.form.get("RecordingStatus", "")

    logger.info(
        "[%s] Recording %s status: %s", call_sid, recording_sid, recording_status
    )

    state = get_call_state(call_sid)
    if state and recording_status == "completed":
        state["recording_sid"] = recording_sid

    return Response("", status=204)


@app.post("/voice/status")
def voice_status() -> Response:
    """
    Twilio call status callback.  Updates the call state on completion.
    """
    call_sid = request.form.get("CallSid", "")
    call_status = request.form.get("CallStatus", "")
    logger.info("[%s] Call status: %s", call_sid, call_status)

    state = get_call_state(call_sid)
    if state:
        state["status"] = call_status

    return Response("", status=204)
