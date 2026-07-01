"""
Central configuration for the Voice Agent Testing Bot.

All settings are loaded from environment variables (see .env.example).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Twilio ──────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID: str = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER: str = os.environ.get("TWILIO_PHONE_NUMBER", "")

# ── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Call target ───────────────────────────────────────────────────────────────
# Only this number is ever dialled by the bot.
TARGET_PHONE_NUMBER: str = os.environ.get("TARGET_PHONE_NUMBER", "+18054398008")

# ── Webhook server ────────────────────────────────────────────────────────────
WEBHOOK_BASE_URL: str = os.environ.get("WEBHOOK_BASE_URL", "http://localhost:5000")
WEBHOOK_PORT: int = int(os.environ.get("WEBHOOK_PORT", "5000"))

# ── Call behaviour ────────────────────────────────────────────────────────────
MAX_CALL_DURATION: int = int(os.environ.get("MAX_CALL_DURATION", "300"))

# ── Output directories ────────────────────────────────────────────────────────
RECORDINGS_DIR: str = os.environ.get("RECORDINGS_DIR", "recordings")
REPORTS_DIR: str = os.environ.get("REPORTS_DIR", "reports")
