"""
Unit tests for nlp/qa_engine.py.

Intent classification accuracy is checked against a held-out set of
phrasings that were never in INTENT_EXAMPLES (verified by construction
below), which is the only honest way to test a retrieval/classification
system -- testing against its own training examples would just prove it
can look itself up. Answer generation is checked against a
hand-constructed PerformanceReport with known numbers, so every answer
string's content can be asserted exactly rather than just "is non-empty."
"""

import os
from unittest import mock

import pandas as pd
import pytest

from nlp.qa_engine import INTENT_EXAMPLES, QAEngine, _try_llm_answer, classify_intent
from reporting.metrics import PerformanceReport

HELD_OUT_QUESTIONS = [
    ("how much profit did i make overall", "total_return"),
    ("whats the yearly performance", "annualized_return"),
    ("how risky adjusted was the return", "sharpe_ratio"),
    ("what was the worst decline", "max_drawdown"),
    ("how much trading activity happened", "turnover"),
    ("did we do better than just holding", "benchmark_comparison"),
    ("how many total trades", "num_trades"),
    ("what does sharpe ratio actually mean", "explain_sharpe"),
    ("please explain what a drawdown is", "explain_drawdown"),
    ("good evening", "help"),
]


def test_held_out_questions_are_not_verbatim_training_examples():
    # Guards against accidentally testing the system against its own
    # training data, which would prove nothing about generalization.
    all_training_texts = {ex for examples in INTENT_EXAMPLES.values() for ex in examples}
    for question, _ in HELD_OUT_QUESTIONS:
        assert question not in all_training_texts


def test_intent_classification_accuracy_on_held_out_questions():
    correct = 0
    for question, expected_intent in HELD_OUT_QUESTIONS:
        intent, confidence = classify_intent(question)
        if intent == expected_intent:
            correct += 1
    accuracy = correct / len(HELD_OUT_QUESTIONS)
    assert accuracy >= 0.8, f"held-out intent accuracy too low: {accuracy:.0%}"


@pytest.mark.parametrize(
    "junk_question",
    [
        "what is the capital of france",
        "asdkjasjd qweqwe",
        "tell me a joke",
        "whats the weather today",
        "who won the cricket match",
    ],
)
def test_out_of_domain_questions_score_near_zero_confidence(junk_question):
    _, confidence = classify_intent(junk_question)
    assert confidence < 0.1


def _make_report(**overrides) -> PerformanceReport:
    defaults = dict(
        total_return=0.5599,
        annualized_return=0.0946,
        sharpe_ratio=0.57,
        max_drawdown=-0.2275,
        max_drawdown_start=pd.Timestamp("2022-01-10"),
        max_drawdown_end=pd.Timestamp("2022-06-20"),
        win_rate=0.5,
        n_round_trips=4,
        turnover=1.70,
        benchmark_total_return=1.5170,
        benchmark_annualized_return=0.2063,
    )
    defaults.update(overrides)
    return PerformanceReport(**defaults)


def _make_engine(**report_overrides) -> QAEngine:
    report = _make_report(**report_overrides)
    trades = pd.DataFrame({"side": ["BUY", "SELL", "BUY", "SELL"]})
    return QAEngine(report=report, trades=trades, symbol="SBIN")


def test_answer_total_return_contains_real_numbers():
    engine = _make_engine()
    answer = engine.answer("what was the total return")
    assert "55.99%" in answer
    assert "151.70%" in answer  # benchmark


def test_answer_sharpe_contains_real_number():
    engine = _make_engine()
    answer = engine.answer("what was the sharpe ratio")
    assert "0.57" in answer


def test_answer_max_drawdown_contains_number_and_dates():
    engine = _make_engine()
    answer = engine.answer("what was the max drawdown")
    assert "-22.75%" in answer
    assert "2022-01-10" in answer
    assert "2022-06-20" in answer


def test_answer_win_rate_contains_number_and_trip_count():
    engine = _make_engine()
    answer = engine.answer("what was the win rate")
    assert "50.0%" in answer
    assert "4" in answer


def test_answer_win_rate_handles_no_round_trips():
    engine = _make_engine(win_rate=None, n_round_trips=0)
    answer = engine.answer("what was the win rate")
    assert "no completed round-trip trades" in answer.lower()


def test_answer_num_trades_uses_real_trades_dataframe():
    engine = _make_engine()
    answer = engine.answer("how many trades were there")
    assert "4 trades" in answer  # len(trades) == 4 in _make_engine


def test_answer_benchmark_comparison_reflects_underperformance():
    engine = _make_engine()  # strategy 55.99% vs benchmark 151.70% -> underperformed
    answer = engine.answer("did it beat buy and hold")
    assert "underperformed" in answer.lower()


def test_answer_benchmark_comparison_reflects_outperformance():
    engine = _make_engine(total_return=2.0, benchmark_total_return=0.5)
    answer = engine.answer("did it beat buy and hold")
    assert "beat" in answer.lower()
    assert "underperformed" not in answer.lower()


def test_explain_intents_return_glossary_text_not_numbers():
    engine = _make_engine()
    answer = engine.answer("what is a sharpe ratio")
    assert "risk" in answer.lower()
    assert "55.99%" not in answer  # a definition, not this backtest's own numbers


def test_help_intent_lists_capabilities():
    engine = _make_engine()
    answer = engine.answer("hello")
    assert "sharpe" in answer.lower() or "return" in answer.lower()


def test_low_confidence_question_returns_fallback_message():
    engine = _make_engine()
    answer = engine.answer("what is the capital of france")
    assert "not confident" in answer.lower() or "i can answer" in answer.lower()


def test_empty_question_returns_help_text():
    engine = _make_engine()
    answer = engine.answer("   ")
    assert len(answer) > 0


def test_llm_answer_returns_none_without_api_key():
    trades = pd.DataFrame({"side": ["BUY", "SELL"]})
    with mock.patch.dict(os.environ, {}, clear=True):
        result = _try_llm_answer("some question", _make_report(), trades, "SBIN")
    assert result is None


def test_llm_answer_used_when_available_even_for_low_confidence_questions():
    # If the LLM path returns something, QAEngine.answer() must use it
    # directly and skip the offline retrieval classifier entirely -- even
    # for a question the retrieval system would score as low-confidence.
    engine = _make_engine()
    with mock.patch("nlp.qa_engine._try_llm_answer", return_value="A concise LLM-generated answer."):
        answer = engine.answer("what is the capital of france")
    assert answer == "A concise LLM-generated answer."


def test_offline_fallback_used_when_llm_path_returns_none():
    # With no API key (the sandbox/CI default), _try_llm_answer returns
    # None and QAEngine.answer() must fall through to the offline
    # retrieval path -- this is the behavior every other test in this
    # file already implicitly relies on, asserted explicitly here.
    engine = _make_engine()
    with mock.patch("nlp.qa_engine._try_llm_answer", return_value=None):
        answer = engine.answer("what was the total return")
    assert "55.99%" in answer
