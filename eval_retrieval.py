"""
Automated retrieval-quality eval for RulesBot.

Instead of manually spot-checking answers, this script runs a fixed set of
question/expected-game pairs through retrieve() and measures whether the
correct game's rules surface in the top results. It reports an accuracy score
and logs every failure to a JSON file for human review.

This evaluates RETRIEVAL only (not generation), so it's fast and needs no
Groq API key — it's the layer that everything downstream depends on. A bad
answer almost always starts as a bad retrieval.

Run:  python eval_retrieval.py
Exit code is 0 if top-k accuracy meets PASS_THRESHOLD, 1 otherwise (so this
can gate a CI check later).

Two metrics are reported:
  - top-1 accuracy : the single closest chunk is from the right game (strict)
  - top-k accuracy : the right game appears anywhere in the top N_RESULTS
                     chunks (this is what generate_response() actually sees)
"""

import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

from config import N_RESULTS
from retriever import retrieve

# Fraction of cases whose correct game must appear in the top-k results for
# the suite to "pass". Tune as the eval set grows.
PASS_THRESHOLD = 0.80

FAILURE_LOG = "eval_failures.json"

# --- Eval set -------------------------------------------------------------
# Each case: a natural-language question, the game its answer lives in
# (must match the game name ingest.py produces from the filename), and an
# optional keyword we expect to see in at least one retrieved chunk's text
# (a lightweight content check beyond just matching the game).
#
# Some questions deliberately OMIT the game name to test real semantic
# retrieval rather than keyword matching on the title.
EVAL_SET = [
    {
        "question": "How many victory points do you need to win Catan?",
        "expected_game": "Catan",
        "keyword": "10 Victory Points",
    },
    {
        "question": "What resources can players collect in Catan?",
        "expected_game": "Catan",
        "keyword": "Brick",
    },
    {
        "question": "How do you win the game of Clue?",
        "expected_game": "Clue",
        "keyword": "accusation",
    },
    {
        "question": "Which deduction game has players solve a murder in a mansion?",
        "expected_game": "Clue",
        "keyword": "murder",
    },
    {
        "question": "What is the role of the Spymaster in Codenames?",
        "expected_game": "Codenames",
        "keyword": "Spymaster",
    },
    {
        "question": "How many codename cards are laid out on the table?",
        "expected_game": "Codenames",
        "keyword": "25",
    },
    {
        "question": "How do you win at Monopoly?",
        "expected_game": "Monopoly",
        "keyword": "bankrupt",
    },
    {
        "question": "What can players build on properties to charge more rent?",
        "expected_game": "Monopoly",
        "keyword": "house",
    },
    {
        "question": "Is Pandemic a cooperative game?",
        "expected_game": "Pandemic",
        "keyword": "cooperative",
    },
    {
        "question": "How many diseases must the team cure to win Pandemic?",
        "expected_game": "Pandemic",
        "keyword": "four",
    },
    {
        "question": "How does a player win in Risk?",
        "expected_game": "Risk",
        "keyword": "mission",
    },
    {
        "question": "How many dice does the attacker roll when battling for a territory?",
        "expected_game": "Risk",
        "keyword": "dice",
    },
    {
        "question": "How do you score points in Ticket to Ride?",
        "expected_game": "Ticket To Ride",
        "keyword": "route",
    },
    {
        "question": "What do players collect to claim railway routes between cities?",
        "expected_game": "Ticket To Ride",
        "keyword": "train",
    },
    {
        "question": "How many points do you need to win a game of Uno?",
        "expected_game": "Uno",
        "keyword": "500",
    },
    {
        "question": "In which card game do you match cards by color or number to empty your hand?",
        "expected_game": "Uno",
        "keyword": "color",
    },
    {
        "question": "How many $500 bills do you start with in Monopoly?",
        "expected_game": "Monopoly",
        "keyword": "two",
    },
    {
        "question": "What happens when, while moving across the board, you roll doubles thrice?",
        "expected_game": "Monopoly",
        "keyword": "Jail",
    },
]


def run_retrieval(question, n_results=N_RESULTS):
    """Call retrieve(), suppressing its debug prints to keep eval output clean."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        return retrieve(question, n_results=n_results)


def evaluate_case(case):
    """Run one case and return a result dict with the metrics for that case."""
    results = run_retrieval(case["question"])
    games = [r["game"] for r in results]

    top1_hit = bool(results) and results[0]["game"] == case["expected_game"]
    topk_hit = case["expected_game"] in games

    keyword = case.get("keyword")
    if keyword:
        keyword_hit = any(keyword.lower() in r["text"].lower() for r in results)
    else:
        keyword_hit = None  # not checked for this case

    return {
        "question": case["question"],
        "expected_game": case["expected_game"],
        "keyword": keyword,
        "retrieved_games": games,
        "top_distance": round(results[0]["distance"], 4) if results else None,
        "top1_hit": top1_hit,
        "topk_hit": topk_hit,
        "keyword_hit": keyword_hit,
        # A short snippet of the top chunk, handy when reviewing failures.
        "top_chunk_preview": results[0]["text"][:120] if results else "",
    }


def main():
    # The rule texts contain em dashes etc.; force UTF-8 so printing snippets
    # doesn't crash on a cp1252 Windows console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print("=" * 60)
    print(f"  RulesBot retrieval eval — {len(EVAL_SET)} cases, top-{N_RESULTS}")
    print("=" * 60)

    # Each evaluate_case does one case, and inside each evaluate_case is one call to retrieve.
    # Each evaluate_case outputs a dict, and then we're looping, so we get a list of dicts.
    case_results = [evaluate_case(c) for c in EVAL_SET]

    # Loop through the dicts, doing stuff based on the values of certain keys.
    top1_hits = sum(r["top1_hit"] for r in case_results)
    topk_hits = sum(r["topk_hit"] for r in case_results)
    keyword_checked = [r for r in case_results if r["keyword_hit"] is not None]
    keyword_hits = sum(r["keyword_hit"] for r in keyword_checked)

    # Per-case lines.
    for r in case_results:
        mark = "PASS" if r["topk_hit"] else "FAIL" # Fail if none of top k work.
        place = "top-1" if r["top1_hit"] else ("top-k" if r["topk_hit"] else "miss ")
        print(
            f"[{mark}] ({place}) {r['question']}\n"
            f"        expected={r['expected_game']!r}  got={r['retrieved_games']}"
            f"  dist={r['top_distance']}"
        )

    total = len(case_results)
    topk_acc = topk_hits / total
    top1_acc = top1_hits / total

    print("-" * 60)
    print(f"top-1 accuracy : {top1_hits}/{total} = {top1_acc:.0%}")
    print(f"top-k accuracy : {topk_hits}/{total} = {topk_acc:.0%}   (k={N_RESULTS})")
    if keyword_checked:
        print(
            f"keyword recall : {keyword_hits}/{len(keyword_checked)} = "
            f"{keyword_hits / len(keyword_checked):.0%}   "
            "(expected snippet found in a retrieved chunk)"
        )

    # Log failures for human review.
    failures = [r for r in case_results if not r["topk_hit"]] # Fail if none of top k work.
    if failures:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_results": N_RESULTS,
            "total_cases": total,
            "top1_accuracy": top1_acc,
            "topk_accuracy": topk_acc,
            "failures": failures,
        }
        with open(FAILURE_LOG, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n{len(failures)} failure(s) logged to {FAILURE_LOG} for review.")
    else:
        # Clean run — remove a stale log from a previous failing run if present.
        if os.path.exists(FAILURE_LOG):
            os.remove(FAILURE_LOG)
        print("\nNo failures. 🎉")

    # Pass if greater than or equal to certain % right, where in top k is enough.
    passed = topk_acc >= PASS_THRESHOLD
    print(
        f"\nResult: {'PASS' if passed else 'FAIL'} "
        f"(top-k accuracy {topk_acc:.0%} vs threshold {PASS_THRESHOLD:.0%})"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
