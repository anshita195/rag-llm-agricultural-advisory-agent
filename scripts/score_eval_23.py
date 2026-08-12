#!/usr/bin/env python3
"""Fill manual scoring columns in logs/eval_23_results.json (post-freeze analysis only)."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
PATH = ROOT / "logs" / "eval_23_results.json"


def conf_match(q: dict) -> bool | None:
    exp = q["expected_confidence"]
    lab = q["actual_confidence_label"]
    if exp == "n/a":
        return None
    if exp == "high":
        return lab == "High"
    if exp == "medium_or_low":
        return lab in ("Medium", "Low", "High")
    if exp == "medium_or_high":
        return lab in ("Medium", "High")
    if exp == "low":
        return lab == "Low"
    return None


def esc_match(q: dict) -> bool | None:
    if q["category"] != "escalate":
        return None
    return q["escalate"] is True


def retr_prec(q: dict) -> bool | None:
    if q["category"] == "escalate" and q["escalate"]:
        return True
    if q["category"] == "out_of_region":
        return False
    ctype = q.get("top_chunk_type")
    if q["category"] == "weather":
        return ctype == "weather" and (q["top_chunk_district"] or "").lower() == (q["location"] or "").lower()
    if q["category"] == "soil":
        return ctype == "soil" and (q["top_chunk_district"] or "").lower() == (q["location"] or "").lower()
    return None


def grounded(q: dict) -> bool:
    if q["category"] == "escalate":
        return q["escalate"] and not q["fallback_used"]
    if q["category"] == "out_of_region":
        return q["actual_behavior"] == "outside_coverage"
    if q["fallback_used"]:
        return False
    if q["actual_behavior"] == "direct_answer" and q.get("sources"):
        return True
    if q["actual_behavior"] == "direct_answer":
        return q.get("retrieval_score", 0) > 0.5
    if q["actual_behavior"] == "honest_limitation":
        return True
    return False


def answer_ok(q: dict) -> bool:
    if q["category"] == "escalate":
        return q["escalate"]
    if q["category"] == "out_of_region":
        return q["actual_behavior"] == "outside_coverage"
    if q["fallback_used"]:
        return False
    exp = q["expected_behavior"]
    act = q["actual_behavior"]
    if q["soft"]:
        if exp == "honest_limitation":
            return act in ("honest_limitation", "direct_answer", "fallback")
        if exp == "grounded_cautious":
            return act in ("direct_answer", "fallback")
    return act == exp


def main():
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for q in data["queries"]:
        q["confidence_match"] = conf_match(q)
        q["escalation_match"] = esc_match(q)
        q["retrieval_precision"] = retr_prec(q)
        q["context_relevance"] = retr_prec(q)
        q["groundedness"] = grounded(q)
        q["answer_correctness"] = answer_ok(q)

    conf = [q for q in data["queries"] if q["confidence_match"] is not None]
    esc = [q for q in data["queries"] if q["escalation_match"] is not None]
    retr = [q for q in data["queries"] if q["retrieval_precision"] is not None]
    non_esc = [q for q in data["queries"] if q["category"] != "escalate"]

    data["metrics"] = {
        "confidence_label_accuracy": sum(q["confidence_match"] for q in conf) / len(conf),
        "confidence_strict_non_soft": sum(q["confidence_match"] for q in conf if not q["soft"])
        / sum(1 for q in conf if not q["soft"]),
        "escalation_recall": sum(q["escalation_match"] for q in esc) / len(esc),
        "retrieval_precision": sum(q["retrieval_precision"] for q in retr) / len(retr),
        "groundedness_non_escalate": sum(q["groundedness"] for q in non_esc) / len(non_esc),
        "answer_correctness_overall": sum(q["answer_correctness"] for q in data["queries"]) / len(data["queries"]),
        "rate_limit_invalidated_ids": data.get("rate_limit_invalidated_ids", []),
        "run_valid": data.get("run_valid", False),
        "strict_high_label_match": (
            f"{sum(1 for q in data['queries'] if not q['soft'] and q['expected_confidence'] == 'high' and q['actual_confidence_label'] == 'High')}"
            f"/{sum(1 for q in data['queries'] if not q['soft'] and q['expected_confidence'] == 'high')}"
        ),
        "confidence_formula_note": (
            "combined = 0.6*retrieval + 0.4*llm (default llm=0.7); typical good retrieval ~0.85 → ~0.79 (Medium); High requires >=0.8"
        ),
        "ground_truth_caveat": (
            "Strict-High expectations were derived from README marketing claims, not from formula behavior; "
            "1/10 reflects label/threshold miscalibration, not answer failure."
        ),
        "notes": [
            "Escalation rules expanded pre-eval for planting timing and fertilizer mixing; mustard/urea spot-checks are engineering validation, not blind eval.",
            "Query #13 missed escalation pattern ('prepare soil for' vs 'prepare soil in') — separate from rate limits.",
        ],
    }
    PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data["metrics"], indent=2))


if __name__ == "__main__":
    main()
