"""
Recording and transcription utilities.

Handles:
  - Downloading Twilio call recordings via the REST API.
  - Transcribing audio using Twilio's built-in transcription resource.
  - Parsing Twilio transcription segments into (speaker, text) tuples.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from twilio.rest import Client

import config

logger = logging.getLogger(__name__)


class RecordingManager:
    """Downloads and transcribes Twilio call recordings."""

    def __init__(self) -> None:
        self._client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        Path(config.RECORDINGS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Recording download ────────────────────────────────────────────────────

    def wait_for_recording(
        self, call_sid: str, timeout: int = 120, poll_interval: int = 5
    ) -> str | None:
        """
        Wait until a recording is available for the given call SID, then
        return its Recording SID.  Returns None if the timeout is exceeded.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            recordings = self._client.recordings.list(call_sid=call_sid, limit=1)
            if recordings:
                return recordings[0].sid
            time.sleep(poll_interval)
        logger.warning("No recording found for call %s after %ds", call_sid, timeout)
        return None

    def download_recording(self, recording_sid: str, call_sid: str) -> str | None:
        """
        Download a Twilio recording as an MP3 file.

        Returns the local file path on success, or None on failure.
        """
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{config.TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.mp3"
        )
        dest = os.path.join(config.RECORDINGS_DIR, f"{call_sid}.mp3")

        try:
            response = requests.get(
                url,
                auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
                timeout=60,
            )
            response.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(response.content)
            logger.info("Recording saved to %s", dest)
            return dest
        except requests.RequestException as exc:
            logger.error("Failed to download recording %s: %s", recording_sid, exc)
            return None

    # ── Transcription ─────────────────────────────────────────────────────────

    def get_transcription(
        self, recording_sid: str, timeout: int = 180, poll_interval: int = 10
    ) -> list[tuple[str, str]] | None:
        """
        Request and wait for Twilio's transcription of a recording.

        Returns a list of (speaker, text) tuples where speaker is "AGENT"
        or "PATIENT", or None if transcription failed / timed out.

        Twilio's basic transcription does not do speaker diarisation;
        we heuristically label turns using the call direction and timing.
        """
        try:
            transcription = self._client.transcriptions.create(
                recording_sid=recording_sid
            )
            transcription_sid = transcription.sid
        except Exception as exc:
            logger.error("Failed to request transcription: %s", exc)
            return None

        # Poll for completion
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                t = self._client.transcriptions(transcription_sid).fetch()
                if t.status == "completed":
                    return _parse_transcription_text(t.transcription_text or "")
                if t.status == "failed":
                    logger.error("Twilio transcription failed for %s", recording_sid)
                    return None
            except Exception as exc:
                logger.error("Error polling transcription: %s", exc)
                return None
            time.sleep(poll_interval)

        logger.warning("Transcription timed out for recording %s", recording_sid)
        return None

    def build_transcript_from_history(
        self,
        conversation_history: list[dict[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Convert the in-memory conversation history (from PatientSimulator) into
        the (speaker, text) format expected by the Analyzer.

        The history uses OpenAI roles:
          - "user"      → the patient heard the AGENT's response and is replying,
                          but in our setup "user" messages = agent utterances fed
                          to the simulator.
          - "assistant" → the simulator's (patient's) reply.

        So we invert: "user" → AGENT, "assistant" → PATIENT.
        """
        transcript: list[tuple[str, str]] = []
        for msg in conversation_history:
            role = msg.get("role", "")
            text = msg.get("content", "").strip()
            if not text or text in ("[SCENARIO COMPLETE]", "[SCENARIO STUCK]"):
                continue
            if role == "user":
                transcript.append(("AGENT", text))
            elif role == "assistant":
                transcript.append(("PATIENT", text))
        return transcript


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_transcription_text(raw: str) -> list[tuple[str, str]]:
    """
    Parse raw Twilio transcription text into speaker-labelled turns.

    Twilio's basic transcription returns a plain string without speaker labels.
    We split it into sentences and alternate AGENT / PATIENT labels, starting
    with AGENT (since the agent typically greets first).

    For production use, consider Twilio Intelligence or Google/AWS STT with
    speaker diarisation for accurate labelling.
    """
    if not raw:
        return []

    sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
    transcript: list[tuple[str, str]] = []
    speakers = ["AGENT", "PATIENT"]
    for i, sentence in enumerate(sentences):
        transcript.append((speakers[i % 2], sentence + "."))
    return transcript
