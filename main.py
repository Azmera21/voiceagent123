"""
Main entry point for the Voice Agent Testing Bot.

Usage examples
--------------
# Run all scenarios:
    python main.py

# Run a specific scenario:
    python main.py --scenario appointment_scheduling

# List available scenarios:
    python main.py --list

# Start the webhook server only (for manual testing):
    python main.py --server-only

Prerequisites
-------------
1. Copy .env.example to .env and fill in your credentials.
2. Expose the Flask server publicly (e.g., via ngrok) and set WEBHOOK_BASE_URL.
3. Ensure your Twilio number has voice capabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading

from flask import Flask

import config
import scenarios as scenario_module
from server import app as flask_app
from voice_bot import VoiceBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _start_server_thread() -> None:
    """Start the Flask webhook server in a daemon thread."""
    flask_app.run(
        host="0.0.0.0",
        port=config.WEBHOOK_PORT,
        debug=False,
        use_reloader=False,
    )


def _validate_config() -> None:
    """Abort early with a clear message if required config is missing."""
    missing = []
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "OPENAI_API_KEY"):
        if not getattr(config, var):
            missing.append(var)
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Copy .env.example to .env and fill in the values.")
        sys.exit(1)

    if config.TARGET_PHONE_NUMBER != "+18054398008":
        logger.error(
            "TARGET_PHONE_NUMBER is not set to the authorised test number. "
            "Update your .env file."
        )
        sys.exit(1)


def _print_summary(results: list[dict]) -> None:
    """Print a concise summary of all scenario results."""
    print("\n" + "=" * 70)
    print("OVERALL TEST RUN SUMMARY")
    print("=" * 70)
    for r in results:
        name = r.get("scenario_name", "?")
        quality = r.get("analysis", {}).get("overall_quality", "unknown").upper()
        n_issues = len(r.get("analysis", {}).get("issues", []))
        report = r.get("report_path", "N/A")
        error = r.get("error", "")
        if error:
            print(f"  ✗  {name:<35} ERROR: {error}")
        else:
            icon = "✓" if quality == "GOOD" else ("!" if quality == "ACCEPTABLE" else "✗")
            print(f"  {icon}  {name:<35} {quality} ({n_issues} issue(s))  → {report}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated voice agent testing bot for healthcare call centres.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        help="Run a single named scenario (default: run all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scenario names and exit.",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Start the webhook server without placing any calls.",
    )
    args = parser.parse_args()

    # ── List scenarios ─────────────────────────────────────────────────────────
    if args.list:
        print("Available scenarios:")
        for s in scenario_module.SCENARIOS:
            print(f"  {s['name']:<35} {s['description']}")
        sys.exit(0)

    _validate_config()

    # ── Start webhook server in background ─────────────────────────────────────
    server_thread = threading.Thread(target=_start_server_thread, daemon=True)
    server_thread.start()
    logger.info("Webhook server running at %s (port %d)", config.WEBHOOK_BASE_URL, config.WEBHOOK_PORT)

    # ── Server-only mode ───────────────────────────────────────────────────────
    if args.server_only:
        logger.info("Server-only mode. Press Ctrl+C to stop.")
        try:
            server_thread.join()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        sys.exit(0)

    # ── Determine which scenarios to run ──────────────────────────────────────
    if args.scenario:
        try:
            scenarios_to_run = [scenario_module.get_scenario(args.scenario)]
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        scenarios_to_run = scenario_module.SCENARIOS

    # ── Run scenarios sequentially ─────────────────────────────────────────────
    bot = VoiceBot()
    results = []

    for scenario in scenarios_to_run:
        logger.info("─" * 60)
        logger.info("Running scenario: %s", scenario["name"])
        result = bot.run_scenario(scenario["name"])
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
