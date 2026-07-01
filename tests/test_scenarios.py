"""
Tests for the scenarios module.
"""

import pytest
from scenarios import SCENARIOS, get_scenario


def test_all_scenarios_have_required_keys():
    required = {"name", "description", "persona", "opening", "goal", "max_turns",
                "expected_outcomes", "red_flags"}
    for scenario in SCENARIOS:
        missing = required - set(scenario.keys())
        assert not missing, f"Scenario '{scenario['name']}' missing keys: {missing}"


def test_scenario_names_are_unique():
    names = [s["name"] for s in SCENARIOS]
    assert len(names) == len(set(names)), "Duplicate scenario names found."


def test_get_scenario_returns_correct_scenario():
    scenario = get_scenario("appointment_scheduling")
    assert scenario["name"] == "appointment_scheduling"
    assert "Dr." in scenario["persona"]


def test_get_scenario_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("nonexistent_scenario")


def test_all_scenarios_have_non_empty_opening():
    for scenario in SCENARIOS:
        assert scenario["opening"].strip(), (
            f"Scenario '{scenario['name']}' has an empty opening line."
        )


def test_all_scenarios_have_expected_outcomes():
    for scenario in SCENARIOS:
        assert scenario["expected_outcomes"], (
            f"Scenario '{scenario['name']}' has no expected outcomes."
        )


def test_all_scenarios_max_turns_positive():
    for scenario in SCENARIOS:
        assert scenario["max_turns"] > 0, (
            f"Scenario '{scenario['name']}' has non-positive max_turns."
        )


def test_known_scenario_names_exist():
    expected_names = {
        "appointment_scheduling",
        "prescription_refill",
        "urgent_symptom_question",
        "billing_inquiry",
        "after_hours_non_urgent",
        "new_patient_registration",
    }
    actual_names = {s["name"] for s in SCENARIOS}
    assert expected_names == actual_names
