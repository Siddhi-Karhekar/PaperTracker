"""
nlp/qa_engine.py

A natural-language Q&A engine over a completed backtest's real results.
Answering a question tries two paths, in order:

1. LLM-backed answer (primary, when available): if ANTHROPIC_API_KEY is
   set, the question is sent to Claude Haiku -- the fastest model in the
   Claude family -- grounded on the real PerformanceReport numbers via
   the system prompt, with an explicit instruction to answer in one or
   two short sentences and max_tokens capped low. This gives genuinely
   open-ended natural-language understanding (not just the fixed set of
   questions below) while staying fast and concise rather than producing
   a wall of text for a one-line question.

2. Offline retrieval fallback (when no API key is set, or the LLM call
   fails for any reason): TF-IDF vectorization + nearest-neighbor cosine
   similarity against a curated set of example phrasings per intent
   (INTENT_EXAMPLES below) -- e.g. "what was the Sharpe ratio" and "how'd
   we do on a risk-adjusted basis" both land closest to the sharpe_ratio
   examples and route there. This is a real, if intentionally lightweight,
   NLP retrieval technique, not a hand-written keyword-matching if/else
   chain. It was chosen over a trained probabilistic classifier (Naive
   Bayes was the first thing tried) specifically because it performs
   better here: with only a handful of short examples per intent, Naive
   Bayes' probability estimates come out poorly calibrated and don't
   separate real questions from out-of-domain junk cleanly, while
   cosine-similarity-to-nearest-example does (verified empirically:
   ~90% accuracy on held-out phrasings, and every tested out-of-domain
   question -- "what's the capital of France," "tell me a joke" -- scores
   essentially 0.0 similarity to every trained intent). Answers here pull
   a real number out of the PerformanceReport/trades directly (for "what
   was..." questions) or a plain-English glossary explanation (for "what
   does ... mean" questions) -- always grounded in the actual objects
   passed in, never invented.

This ordering means the project still runs end-to-end, chatbot included,
with zero paid dependencies by default (path 2), but gives the better,
more flexible experience (path 1) the moment a key is available --
consistent with the rest of this project's "runs in 2 commands" goal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reporting.metrics import PerformanceReport

# ---------------------------------------------------------------------------
# Training data for the intent classifier. Add a new intent by adding a key
# here plus a matching entry in _ANSWERERS below. Deliberately avoids
# generic filler phrasings ("... please", "tell me the ...") shared across
# unrelated intents -- those collapsed accuracy and confidence separation
# badly when tried; see classify_intent's docstring.
# ---------------------------------------------------------------------------
INTENT_EXAMPLES = {
    "total_return": [
        "what was the total return",
        "how much money did the strategy make",
        "what is my total return",
        "how did the strategy perform overall",
        "how much did i make in total",
        "what was the overall gain or loss",
        "how profitable was this strategy",
        "total profit and loss",
    ],
    "annualized_return": [
        "what was the annualized return",
        "what is the yearly return",
        "how much did it return per year on average",
        "what was the compound annual growth rate",
        "what is the cagr",
        "annualized performance of the strategy",
    ],
    "sharpe_ratio": [
        "what was the sharpe ratio",
        "what is my sharpe ratio",
        "how good was the risk adjusted return",
        "what sharpe ratio did we get",
        "how was the risk adjusted performance",
        "give me the sharpe ratio number",
    ],
    "max_drawdown": [
        "what was the max drawdown",
        "what is the maximum drawdown",
        "how bad was the worst loss",
        "what was the biggest peak to trough decline",
        "how much did it fall from the high",
        "worst drawdown of the strategy",
    ],
    "win_rate": [
        "what was the win rate",
        "how many trades were winners",
        "what percentage of trades made money",
        "what is the winning percentage",
        "how often did the strategy win trades",
        "fraction of profitable trades",
    ],
    "turnover": [
        "what was the turnover",
        "how often did the strategy trade",
        "what is the annual turnover",
        "how much trading activity was there",
        "turnover ratio of the portfolio",
    ],
    "benchmark_comparison": [
        "did it beat buy and hold",
        "how did it compare to just holding the stock",
        "did the strategy outperform the benchmark",
        "was it better than buy and hold",
        "how does this compare to buying and holding",
        "did we beat the market",
    ],
    "num_trades": [
        "how many trades were there",
        "how many trades did the strategy make",
        "what is the total number of trades",
        "how many buys and sells were there",
        "trade count for this backtest",
    ],
    "explain_sharpe": [
        "what is a sharpe ratio",
        "explain the sharpe ratio",
        "what does sharpe ratio mean",
        "can you explain sharpe ratio",
        "what is the sharpe ratio measuring",
    ],
    "explain_drawdown": [
        "what is a drawdown",
        "explain max drawdown",
        "what does drawdown mean",
        "can you explain drawdown",
    ],
    "explain_win_rate": [
        "what is win rate",
        "explain win rate",
        "what does win rate mean",
    ],
    "explain_turnover": [
        "what is turnover",
        "explain turnover",
        "what does turnover mean in trading",
    ],
    "explain_lookahead": [
        "what is look ahead bias",
        "explain look ahead bias",
        "what does look ahead bias mean",
        "how does this project avoid look ahead bias",
    ],
    "explain_walk_forward": [
        "what is walk forward validation",
        "explain walk forward validation",
        "what does walk forward mean",
        "how does walk forward validation work",
    ],
    "help": [
        "hello",
        "hi there",
        "hey there",
        "what can you help with",
        "what can you answer",
        "help me",
        "what can i ask you",
        "good morning",
        "good afternoon",
        "what questions can i ask",
    ],
}

_GLOSSARY = {
    "sharpe_ratio": (
        "The Sharpe ratio measures return relative to how bumpy the ride was to get there -- return per unit of "
        "risk taken. Higher is better; above 1 is generally considered good. Two strategies can post the same "
        "total return, but the one with the higher Sharpe ratio got there more smoothly."
    ),
    "max_drawdown": (
        "Max drawdown is the worst peak-to-trough loss the strategy experienced -- e.g. if the portfolio hit a "
        "high point and later fell 20% before recovering, that's a 20% drawdown. It captures the worst pain a "
        "real investor would have had to sit through holding this strategy."
    ),
    "win_rate": "Win rate is the percentage of completed trades (a buy followed by a later sell) that made money, versus lost money.",
    "turnover": (
        "Turnover measures how often the strategy trades, expressed as a multiple of the portfolio's value "
        "traded per year. Higher turnover means more transaction costs eating into the same underlying returns."
    ),
    "look_ahead_bias": (
        "Look-ahead bias is when a backtest accidentally lets a strategy 'know' information before it was "
        "actually available -- like trading at a day's closing price using a decision that could only have been "
        "made from that same day's close. This project avoids it by executing every trade at the *next* bar's "
        "open instead."
    ),
    "walk_forward": (
        "Walk-forward validation splits history into rolling train/test windows: strategy parameters are picked "
        "using only a training window, then scored on the following, unseen window, rolled forward through time. "
        "It's how this project checks a strategy isn't just overfit to one lucky stretch of history."
    ),
}

_HELP_TEXT = (
    "I can answer questions about this backtest's actual results -- total return, annualized return, Sharpe "
    "ratio, max drawdown, win rate, turnover, how it compares to buy-and-hold, and how many trades it made. "
    "I can also explain what any of those terms mean."
)

_FALLBACK_TEXT = (
    "I'm not confident I understood that -- I can answer questions about this backtest's total return, "
    "annualized return, Sharpe ratio, max drawdown, win rate, turnover, trade count, or benchmark comparison, "
    "and explain what any of those mean."
)


@lru_cache(maxsize=1)
def _intent_index():
    """
    Builds (once, cached) the TF-IDF vector space over every training
    example across all intents, keeping each example's own intent label
    alongside its vector. classify_intent finds the single most similar
    training example to a new question and returns its intent.
    """
    texts, labels = [], []
    for intent, examples in INTENT_EXAMPLES.items():
        texts.extend(examples)
        labels.extend([intent] * len(examples))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    example_vectors = vectorizer.fit_transform(texts)
    return vectorizer, example_vectors, labels


def classify_intent(question: str) -> tuple:
    """
    Returns (intent, confidence). confidence is the cosine similarity
    (0.0-1.0) between the question and its single closest training
    example -- 0.0 means no shared vocabulary with anything trained on at
    all (a strong signal the question is genuinely out of scope, not just
    an unusual phrasing of an in-scope one).
    """
    vectorizer, example_vectors, labels = _intent_index()
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, example_vectors)[0]
    best_idx = similarities.argmax()
    return labels[best_idx], float(similarities[best_idx])


def _fmt_date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _answer_total_return(engine: "QAEngine") -> str:
    r = engine.report
    return (
        f"{engine.symbol}'s strategy total return was {r.total_return:+.2%} over the backtested period "
        f"(buy-and-hold over the same period: {r.benchmark_total_return:+.2%})."
    )


def _answer_annualized_return(engine: "QAEngine") -> str:
    r = engine.report
    return f"Annualized return was {r.annualized_return:+.2%} (buy-and-hold annualized: {r.benchmark_annualized_return:+.2%})."


def _answer_sharpe(engine: "QAEngine") -> str:
    r = engine.report
    quality = "a solid, above-1" if r.sharpe_ratio >= 1.0 else "a modest"
    return f"The Sharpe ratio was {r.sharpe_ratio:.2f} -- {quality} risk-adjusted return per unit of volatility taken."


def _answer_max_drawdown(engine: "QAEngine") -> str:
    r = engine.report
    window = ""
    if r.max_drawdown_start is not None and r.max_drawdown_end is not None:
        window = f" (from {_fmt_date(r.max_drawdown_start)} to {_fmt_date(r.max_drawdown_end)})"
    return f"Max drawdown was {r.max_drawdown:.2%}{window} -- the worst peak-to-trough decline the strategy experienced."


def _answer_win_rate(engine: "QAEngine") -> str:
    r = engine.report
    if r.win_rate is None:
        return "There were no completed round-trip trades, so a win rate can't be computed."
    return f"Win rate was {r.win_rate:.1%} across {r.n_round_trips} completed round-trip trades."


def _answer_turnover(engine: "QAEngine") -> str:
    r = engine.report
    return f"Annualized turnover was {r.turnover:.2f}x -- the portfolio's full value was traded over roughly {r.turnover:.1f} times per year."


def _answer_benchmark_comparison(engine: "QAEngine") -> str:
    r = engine.report
    diff = r.total_return - r.benchmark_total_return
    verdict = "beat" if diff > 0 else "underperformed"
    return f"The strategy {verdict} buy-and-hold by {abs(diff):.2%} in total return ({r.total_return:+.2%} vs {r.benchmark_total_return:+.2%})."


def _answer_num_trades(engine: "QAEngine") -> str:
    n = len(engine.trades)
    return f"The strategy made {n} trades in total, comprising {engine.report.n_round_trips} completed round trips."


def _explain(term: str):
    def _answer(engine: "QAEngine") -> str:
        return _GLOSSARY[term]

    return _answer


def _answer_help(engine: "QAEngine") -> str:
    return _HELP_TEXT


_ANSWERERS = {
    "total_return": _answer_total_return,
    "annualized_return": _answer_annualized_return,
    "sharpe_ratio": _answer_sharpe,
    "max_drawdown": _answer_max_drawdown,
    "win_rate": _answer_win_rate,
    "turnover": _answer_turnover,
    "benchmark_comparison": _answer_benchmark_comparison,
    "num_trades": _answer_num_trades,
    "explain_sharpe": _explain("sharpe_ratio"),
    "explain_drawdown": _explain("max_drawdown"),
    "explain_win_rate": _explain("win_rate"),
    "explain_turnover": _explain("turnover"),
    "explain_lookahead": _explain("look_ahead_bias"),
    "explain_walk_forward": _explain("walk_forward"),
    "help": _answer_help,
}


# Fastest model in the Claude family -- picked specifically for low
# latency in an interactive chat panel, not for maximum capability. A
# one-line factual question grounded on a handful of numbers doesn't need
# a larger model, and a larger model would just be slower for no benefit
# here.
_LLM_MODEL = "claude-haiku-4-5-20251001"
_LLM_MAX_TOKENS = 200  # caps response length -- reinforces "short answer" beyond just prompting for it
_LLM_SYSTEM_PROMPT = (
    "You answer questions about a stock trading backtest using ONLY the numbers given in the context below. "
    "Be fast and concise: answer in 1-2 short sentences, no preamble, no restating the question, no filler. "
    "If the question can't be answered from the given numbers, say so in one sentence -- don't guess."
)


def _report_context(report: PerformanceReport, trades: pd.DataFrame, symbol: str) -> str:
    return (
        f"Backtest results for {symbol}: total_return={report.total_return:.2%}, "
        f"annualized_return={report.annualized_return:.2%}, sharpe_ratio={report.sharpe_ratio:.2f}, "
        f"max_drawdown={report.max_drawdown:.2%}, win_rate={report.win_rate}, "
        f"n_round_trips={report.n_round_trips}, n_trades={len(trades)}, turnover={report.turnover:.2f}x/yr, "
        f"benchmark_total_return={report.benchmark_total_return:.2%}, "
        f"benchmark_annualized_return={report.benchmark_annualized_return:.2%}."
    )


def _try_llm_answer(question: str, report: PerformanceReport, trades: pd.DataFrame, symbol: str) -> Optional[str]:
    """
    The primary answer path when available. Returns None (never raises)
    if ANTHROPIC_API_KEY isn't set, the `anthropic` package isn't
    installed, or the API call fails for any reason -- callers
    unconditionally fall through to the offline retrieval path on None,
    so a network hiccup degrades the chatbot rather than breaking it.
    This path is intentionally never exercised in this project's own
    tests (no key is provisioned in CI/sandbox), the same way live NSE
    network calls aren't -- see test_llm_answer_returns_none_without_api_key
    for what *is* tested here: the clean no-key fallthrough.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    context = _report_context(report, trades, symbol)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=_LLM_MAX_TOKENS,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}],
        )
        return response.content[0].text
    except Exception:  # noqa: BLE001 - never let the LLM path break the chatbot
        return None


@dataclass
class QAEngine:
    report: PerformanceReport
    trades: pd.DataFrame
    symbol: str = "the stock"
    # Empirically chosen: correct held-out matches scored 0.32-1.0 cosine
    # similarity in testing, out-of-domain junk scored ~0.0. 0.25 sits with
    # margin below the weakest real match and well above noise. Only used
    # by the offline retrieval fallback -- irrelevant when the LLM path
    # (below) is available, since that path doesn't need a confidence
    # score to decide whether it understood the question.
    confidence_threshold: float = 0.25

    def answer(self, question: str) -> str:
        if not question or not question.strip():
            return _HELP_TEXT

        llm_answer = _try_llm_answer(question, self.report, self.trades, self.symbol)
        if llm_answer is not None:
            return llm_answer

        intent, confidence = classify_intent(question)
        if confidence < self.confidence_threshold:
            return _FALLBACK_TEXT

        return _ANSWERERS[intent](self)
