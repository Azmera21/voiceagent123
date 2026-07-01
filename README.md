# voiceagent123 — Automated Healthcare Voice Agent Testing Bot

An automated testing bot that calls your healthcare voice agent, simulates realistic patient scenarios, records and transcribes the conversations, and identifies bugs or quality issues in the agent's responses.

---

## How It Works

```
┌──────────────┐   outbound call    ┌──────────────────┐
│  Voice Bot   │ ─────────────────▶ │  Agent under test │
│  (this repo) │                    │  +1-805-439-8008  │
└──────┬───────┘                    └──────────────────┘
       │  TwiML webhooks
       ▼
┌──────────────┐
│  Flask server│  ◀── Twilio sends agent speech via /voice/respond
│  (server.py) │  ──▶ PatientSimulator generates next patient reply
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Analyzer    │  Scores the transcript, flags bugs / quality issues
│  (analyzer.py│  Saves .txt + .json report to reports/
└──────────────┘
```

### Flow per scenario

1. `VoiceBot` places an outbound Twilio call to `+18054398008`.
2. When the agent answers, Twilio POSTs to `/voice/start` → the bot plays the patient's opening line.
3. Twilio's speech recognition captures the agent's reply and POSTs it to `/voice/respond`.
4. `PatientSimulator` (GPT-4o-mini) generates the next contextually appropriate patient response.
5. Steps 3–4 repeat until the scenario goal is met or `max_turns` is reached.
6. The call is recorded by Twilio; the bot downloads the MP3 to `recordings/`.
7. `Analyzer` (GPT-4o) reviews the full transcript and produces a structured quality report.

---

## Scenarios

| Name | Description |
|---|---|
| `appointment_scheduling` | Patient wants to book a routine annual physical |
| `prescription_refill` | Patient requests a lisinopril refill |
| `urgent_symptom_question` | Patient reports chest tightness; needs triage guidance |
| `billing_inquiry` | Patient disputes an unexpected lab charge |
| `after_hours_non_urgent` | Patient asks about antibiotic side effect after hours |
| `new_patient_registration` | New patient wants to register and schedule first visit |

---

## Setup

### Prerequisites

- Python 3.11+
- A [Twilio](https://www.twilio.com/) account with a voice-capable phone number
- An [OpenAI](https://platform.openai.com/) API key
- A publicly reachable HTTPS URL for Twilio webhooks  
  (e.g., [ngrok](https://ngrok.com/): `ngrok http 5000`)

### Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

| Variable | Description |
|---|---|
| `TWILIO_ACCOUNT_SID` | Your Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Your Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your outbound Twilio number (e.g. `+12025551234`) |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `TARGET_PHONE_NUMBER` | **Must be `+18054398008`** (the test number) |
| `WEBHOOK_BASE_URL` | Public HTTPS URL pointing to this server (e.g. ngrok URL) |
| `WEBHOOK_PORT` | Local port (default `5000`) |
| `MAX_CALL_DURATION` | Max call length in seconds (default `300`) |

---

## Usage

```bash
# List all available scenarios
python main.py --list

# Run all scenarios sequentially
python main.py

# Run a single scenario
python main.py --scenario appointment_scheduling
python main.py --scenario prescription_refill
python main.py --scenario urgent_symptom_question

# Start the webhook server only (for manual testing)
python main.py --server-only
```

### Output

- **`recordings/<call_sid>.mp3`** — full call audio
- **`reports/<scenario>_<timestamp>.txt`** — human-readable quality report
- **`reports/<scenario>_<timestamp>.json`** — structured JSON analysis

#### Example report

```
======================================================================
SCENARIO : appointment_scheduling
CALL SID : CA1234567890abcdef
QUALITY  : ACCEPTABLE
----------------------------------------------------------------------
SUMMARY  : The agent successfully scheduled an appointment but failed
           to confirm the date clearly and did not verify the patient's
           date of birth before booking.

ISSUES (2):
  [HIGH] missing_verification — Agent did not verify patient identity before booking.
    Agent said: "Sure, I'll book you for Monday at 3 PM."
  [MEDIUM] poor_empathy — Response felt rushed and mechanical.
    Agent said: "OK what's your number."

RECOMMENDATIONS:
  • Always verify patient name and date of birth before making changes.
  • Use warmer language when confirming appointments.
======================================================================
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests run without live API credentials (Twilio and OpenAI calls are mocked).

---

## Project Structure

```
voiceagent123/
├── main.py               # Entry point / CLI
├── config.py             # Configuration from environment variables
├── scenarios.py          # Patient scenario definitions
├── patient_simulator.py  # AI patient (GPT-4o-mini)
├── server.py             # Flask webhook server for Twilio
├── voice_bot.py          # Call orchestrator (places calls, waits, saves reports)
├── transcriber.py        # Recording download & transcription utilities
├── analyzer.py           # Quality analysis (GPT-4o + rule-based red-flag checks)
├── requirements.txt
├── .env.example
└── tests/
    ├── test_scenarios.py
    ├── test_patient_simulator.py
    ├── test_server.py
    ├── test_transcriber.py
    └── test_analyzer.py
```

---

## Safety Note

The bot enforces a hard-coded guard in `VoiceBot`: **it will refuse to call any number other than `+18054398008`**, even if `TARGET_PHONE_NUMBER` is changed in the environment. This prevents accidental calls to real patients.
