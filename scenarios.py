"""
Patient scenario definitions for voice agent testing.

Each scenario describes a realistic patient interaction that the bot will
simulate when calling the healthcare voice agent under test.

A scenario is a plain dict with the following keys:

  name        – short human-readable label used in reports
  description – one-line summary
  persona     – the system-prompt that shapes the AI patient's personality /
                situation; injected as the OpenAI "system" message
  opening     – the very first thing the patient says when the agent picks up
  goal        – what the patient is trying to accomplish (used by the
                AI to decide when the scenario is "done")
  max_turns   – soft limit on the number of back-and-forth turns before the
                bot hangs up naturally

Quality-check criteria are used by the analyzer to decide whether the agent
handled the scenario correctly:

  expected_outcomes – list of strings that *should* appear (or be implied) in
                      the agent's final responses
  red_flags         – list of strings / phrases that indicate a quality issue if
                      the agent says them
"""

from typing import Any

SCENARIOS: list[dict[str, Any]] = [
    # ── 1. Appointment scheduling ─────────────────────────────────────────────
    {
        "name": "appointment_scheduling",
        "description": "Patient wants to schedule a routine check-up appointment.",
        "persona": (
            "You are a patient named Sarah Johnson, 42 years old. "
            "You are calling to schedule a routine annual physical exam. "
            "You are available Monday or Wednesday afternoons and prefer Dr. Williams if possible. "
            "You have insurance through BlueCross BlueShield. "
            "Be polite but slightly impatient if the agent is slow or unclear. "
            "Do not provide information the agent does not ask for."
        ),
        "opening": (
            "Hi, I'd like to schedule an appointment for a routine check-up please."
        ),
        "goal": "Successfully schedule an appointment or be placed on a callback list.",
        "max_turns": 10,
        "expected_outcomes": [
            "appointment confirmed",
            "date and time provided",
            "callback",
            "availability",
        ],
        "red_flags": [
            "I don't know",
            "I cannot help",
            "please call back",
            "system is down",
        ],
    },

    # ── 2. Prescription refill ────────────────────────────────────────────────
    {
        "name": "prescription_refill",
        "description": "Patient requests a refill for a recurring prescription.",
        "persona": (
            "You are a patient named Michael Torres, 58 years old. "
            "You are calling to refill your lisinopril 10 mg prescription for blood pressure. "
            "Your date of birth is March 12, 1966. "
            "Your preferred pharmacy is CVS on Main Street. "
            "You last refilled about 30 days ago and are running low. "
            "If the agent asks for your provider, it's Dr. Patel."
        ),
        "opening": "Hello, I need to get a refill on my blood pressure medication.",
        "goal": "Refill request successfully submitted or routed to the prescribing provider.",
        "max_turns": 8,
        "expected_outcomes": [
            "refill request submitted",
            "sent to provider",
            "pharmacy",
            "48 hours",
            "business days",
        ],
        "red_flags": [
            "we don't handle refills",
            "call your pharmacy directly",
            "I cannot process",
        ],
    },

    # ── 3. Urgent symptom question ────────────────────────────────────────────
    {
        "name": "urgent_symptom_question",
        "description": "Patient reports chest tightness and asks for guidance.",
        "persona": (
            "You are a patient named Linda Park, 65 years old. "
            "You are experiencing mild chest tightness and shortness of breath that started an hour ago. "
            "You are not sure if it is serious. You want to know if you should come in or go to the ER. "
            "You do not have a fever. You have a history of GERD. "
            "Convey mild anxiety. Accept the agent's recommendation but ask a follow-up question."
        ),
        "opening": (
            "Hi, I'm a patient and I'm having some chest tightness. "
            "I wanted to ask whether I need to come in or if it's okay to wait."
        ),
        "goal": "Receive clear triage guidance (ER, urgent care, or clinic visit).",
        "max_turns": 8,
        "expected_outcomes": [
            "emergency",
            "911",
            "urgent care",
            "come in",
            "doctor",
            "nurse",
        ],
        "red_flags": [
            "I cannot give medical advice",
            "just wait and see",
            "take some antacids",
            "hang up",
        ],
    },

    # ── 4. Insurance and billing inquiry ─────────────────────────────────────
    {
        "name": "billing_inquiry",
        "description": "Patient disputes an unexpected charge on their bill.",
        "persona": (
            "You are a patient named Robert Greene, 47 years old. "
            "You received a bill for $320 for a lab test you thought was covered by insurance. "
            "Your insurance is Aetna PPO. You saw Dr. Kim on April 15. "
            "You are mildly frustrated and want an explanation or to be transferred to billing. "
            "Do not immediately accept the first answer — ask at least one clarifying question."
        ),
        "opening": "I received a bill for a lab test and I was told it would be covered. I'd like to understand the charge.",
        "goal": "Transferred to billing department or given a clear explanation / callback promise.",
        "max_turns": 8,
        "expected_outcomes": [
            "billing department",
            "transfer",
            "review",
            "insurance",
            "callback",
            "explanation",
        ],
        "red_flags": [
            "you owe the money",
            "we cannot dispute",
            "talk to your insurance",
            "nothing we can do",
        ],
    },

    # ── 5. After-hours message for non-urgent matter ──────────────────────────
    {
        "name": "after_hours_non_urgent",
        "description": "Patient calls after hours for a non-urgent matter (medication side effect question).",
        "persona": (
            "You are a patient named Emma Clark, 34 years old. "
            "You started a new antibiotic (amoxicillin) today and are experiencing mild nausea. "
            "You want to know if this is normal and whether you should stop taking it. "
            "You are calm and cooperative. It is currently 9 PM."
        ),
        "opening": "Hi, I started a new antibiotic today and I'm feeling a bit nauseous. Is that normal?",
        "goal": "Receive reassurance or nurse callback instructions for a non-urgent side effect.",
        "max_turns": 6,
        "expected_outcomes": [
            "common side effect",
            "take with food",
            "nurse",
            "callback",
            "on-call",
            "continue taking",
        ],
        "red_flags": [
            "stop taking",
            "go to the ER immediately",
            "I don't know",
            "call back tomorrow",
        ],
    },

    # ── 6. New patient registration ───────────────────────────────────────────
    {
        "name": "new_patient_registration",
        "description": "New patient tries to register and schedule a first appointment.",
        "persona": (
            "You are a new patient named James Wilson, 29 years old. "
            "You recently moved to the area and are looking for a new primary care physician. "
            "You have UnitedHealthcare insurance. You have no major health issues. "
            "You are flexible with scheduling and available any weekday. "
            "Be friendly and provide information only when asked."
        ),
        "opening": "Hi, I'm a new patient and I'd like to register and set up an appointment with a primary care doctor.",
        "goal": "Successfully register as a new patient and schedule or be sent registration forms.",
        "max_turns": 12,
        "expected_outcomes": [
            "new patient forms",
            "registration",
            "appointment scheduled",
            "welcome",
            "information",
        ],
        "red_flags": [
            "we are not accepting",
            "cannot register",
            "try another clinic",
        ],
    },
]


def get_scenario(name: str) -> dict[str, Any]:
    """Return a scenario by name, raising ValueError if not found."""
    for scenario in SCENARIOS:
        if scenario["name"] == name:
            return scenario
    raise ValueError(f"Unknown scenario: {name!r}. Available: {[s['name'] for s in SCENARIOS]}")
