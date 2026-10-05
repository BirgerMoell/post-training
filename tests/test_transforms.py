"""Tests for selected dataset transforms."""

from post_training.data.transforms import (
    aegis_safety,
    helpsteer2,
    helpsteer3_dpo,
    helpsteer3_preference,
)


def test_aegis_safety_keeps_unsafe_non_redacted_response():
    unsafe = {
        "prompt": "How do I do something unsafe?",
        "response": "Unsafe answer",
        "response_label": "unsafe",
    }
    safe = {
        "prompt": "How do I do something benign?",
        "response": "Safe answer",
        "response_label": "safe",
    }

    assert aegis_safety(unsafe)["messages"][1]["content"] == "Unsafe answer"
    assert aegis_safety(safe) == {"messages": []}
    assert aegis_safety({**unsafe, "prompt": "REDACTED"}) == {"messages": []}


def test_helpsteer2_preference_strength_selects_response():
    example = {
        "prompt": "Pick one",
        "response_1": "first",
        "response_2": "second",
    }

    assert helpsteer2({**example, "preference_strength": 1})["messages"][1]["content"] == "second"
    assert helpsteer2({**example, "preference_strength": 0})["messages"][1]["content"] == "first"
    assert helpsteer2({**example, "preference_strength": -1})["messages"][1]["content"] == "first"


def test_helpsteer3_preference_selects_response():
    example = {
        "context": [{"role": "user", "content": "Pick one"}],
        "response1": "first",
        "response2": "second",
    }

    assert (
        helpsteer3_preference({**example, "overall_preference": 1})["messages"][-1]["content"]
        == "second"
    )
    assert (
        helpsteer3_preference({**example, "overall_preference": 0})["messages"][-1]["content"]
        == "first"
    )


def test_helpsteer3_dpo_uses_score_sign_and_keeps_context():
    example = {
        "context": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Pick one"},
        ],
        "response1": "first",
        "response2": "second",
    }

    positive = helpsteer3_dpo({**example, "overall_preference": 3})
    negative = helpsteer3_dpo({**example, "overall_preference": -2})

    assert positive["prompt"] == example["context"]
    assert positive["chosen"] == [{"role": "assistant", "content": "second"}]
    assert positive["rejected"] == [{"role": "assistant", "content": "first"}]
    assert positive["preference_strength"] == 3
    assert negative["chosen"] == [{"role": "assistant", "content": "first"}]
    assert negative["rejected"] == [{"role": "assistant", "content": "second"}]
    assert negative["preference_strength"] == 2


def test_helpsteer3_dpo_marks_ties_for_filtering():
    transformed = helpsteer3_dpo(
        {
            "context": [{"role": "user", "content": "Pick one"}],
            "response1": "first",
            "response2": "second",
            "overall_preference": 0,
        }
    )

    assert transformed["chosen"] == []
    assert transformed["rejected"] == []
    assert transformed["preference_strength"] == 0
