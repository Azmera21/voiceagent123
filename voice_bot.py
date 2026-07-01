"""
Voice bot orchestrator.

Responsible for:
  1. Making an outbound Twilio call to the target number.
  2. Waiting for the call to complete.
  3. Triggering recording download and transcription.
  4. Running quality analysis on the transcript.
  5. Saving the report to disk.

Usage (called from main.py):
    bot = VoiceBot()
    result = bot.run_scenario("appointment_scheduling")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from twilio.rest import Client

import config
from analyzer import Analyzer
from patient_simulator import PatientSimulator
from scenarios import get_scenario
from server import get_call_state, pop_call_state, register_call
from transcriber import RecordingManager

logger = logging.getLogger(__name__)


class VoiceBot:
    """Orchestrates a single scenario test call."""

    # Safety guard — only ever call this number
    _ALLOWED_TARGET = "+18054398008"

    def __init__(self) -> None:
        self._twilio = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self._recording_manager = RecordingManager()
        self._analyzer = Analyzer()
        Path(config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_scenario(self, scenario_name: str) -> dict[str, Any]:
        """
        Run a complete test scenario end-to-end.

        Returns a dict with keys:
          call_sid, scenario_name, transcript, analysis, report_path, recording_path
        """
        # Enforce the target number safety guard
        target = config.TARGET_PHONE_NUMBER
        if target != self._ALLOWED_TARGET:
            raise ValueError(
                f"TARGET_PHONE_NUMBER must be {self._ALLOWED_TARGET!r}. "
                f"Got {target!r}. Update your .env file."
            )

        scenario = get_scenario(scenario_name)
        simulator = PatientSimulator(scenario)

        logger.info("Starting scenario '%s' → calling %s", scenario_name, target)

        call_sid = self._place_call(simulator)
        if not call_sid:
            return {"error": "Failed to place call", "scenario_name": scenario_name}

        # Wait for the call to finish
        self._wait_for_call(call_sid)

        # Retrieve transcript from the in-memory registry
        state = pop_call_state(call_sid)
        transcript = state["transcript"] if state else []

        # If Twilio recorded the call, try to download and supplement transcript
        recording_path: str | None = None
        if state and state.get("recording_sid"):
            recording_path = self._recording_manager.download_recording(
                state["recording_sid"], call_sid
            )

        # Fall back to the in-memory conversation history if transcript is empty
        if not transcript:
            transcript = self._recording_manager.build_transcript_from_history(
                simulator.conversation_history()
            )

        # Analyse
        analysis = self._analyzer.analyse(scenario, transcript)
        report_text = self._analyzer.format_report(scenario, analysis, call_sid)
        report_path = self._save_report(scenario_name, call_sid, report_text, analysis)

        logger.info("Scenario '%s' complete. Report: %s", scenario_name, report_path)
        logger.info("\n%s", report_text)

        return {
            "call_sid": call_sid,
            "scenario_name": scenario_name,
            "transcript": transcript,
            "analysis": analysis,
            "report_path": report_path,
            "recording_path": recording_path,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _place_call(self, simulator: PatientSimulator) -> str | None:
        """
        Create an outbound Twilio call and return the CallSid.
        Registers the call state so the webhook server can find it.
        """
        try:
            call = self._twilio.calls.create(
                to=config.TARGET_PHONE_NUMBER,
                from_=config.TWILIO_PHONE_NUMBER,
                url=f"{config.WEBHOOK_BASE_URL}/voice/start",
                status_callback=f"{config.WEBHOOK_BASE_URL}/voice/status",
                status_callback_method="POST",
                record=True,
                recording_status_callback=f"{config.WEBHOOK_BASE_URL}/voice/recording_status",
                recording_status_callback_method="POST",
                time_limit=config.MAX_CALL_DURATION,
                method="POST",
            )
            call_sid = call.sid
            logger.info("Call placed. SID: %s", call_sid)
            register_call(call_sid, simulator)
            return call_sid
        except Exception as exc:
            logger.error("Failed to place call: %s", exc)
            return None

    def _wait_for_call(
        self, call_sid: str, poll_interval: int = 5, max_wait: int = 600
    ) -> None:
        """Poll Twilio until the call reaches a terminal status."""
        terminal_statuses = {"completed", "failed", "busy", "no-answer", "canceled"}
        deadline = time.time() + max_wait

        while time.time() < deadline:
            try:
                call = self._twilio.calls(call_sid).fetch()
                status = call.status
                logger.debug("[%s] Status: %s", call_sid, status)
                if status in terminal_statuses:
                    logger.info("[%s] Call ended with status: %s", call_sid, status)
                    return
            except Exception as exc:
                logger.warning("[%s] Error fetching call status: %s", call_sid, exc)
            time.sleep(poll_interval)

        logger.warning("[%s] Timed out waiting for call to complete.", call_sid)

    def _save_report(
        self,
        scenario_name: str,
        call_sid: str,
        report_text: str,
        analysis: dict[str, Any],
    ) -> str:
        """Save the text report and raw JSON analysis to the reports directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(config.REPORTS_DIR, f"{scenario_name}_{timestamp}")

        txt_path = f"{base}.txt"
        json_path = f"{base}.json"

        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"call_sid": call_sid, "scenario": scenario_name, "analysis": analysis},
                fh,
                indent=2,
            )

        return txt_path
