#!/usr/bin/env python3
"""Recompute selected published results from the committed CSV/JSON evidence.

Standard library only. This command does not load models or start an evaluation.
It validates aggregate metrics, 880 cross-suite trial records and paired coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_SUITES = ("libero_spatial", "libero_goal", "libero_10")
CROSS_METHODS = ("H50", "H20", "mlp_lvm", "mlp_ogg")
OBJECT_METHODS = ("H50", "H20", "lvm", "ogg")


class EvidenceError(ValueError):
    """Published evidence disagrees with its declared evaluation protocol."""


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise EvidenceError(detail)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(bool(reader.fieldnames), f"Missing CSV header: {path}")
        return list(reader)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_bool(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise EvidenceError(f"Invalid boolean in recorded trial: {value!r}")


def object10(root: Path) -> dict:
    base = root / "object10"
    data = rows(base / "closed_loop.csv")
    selected = {
        r["method"]: r for r in data if r["cohort"] == "night/mlp_s42"
    }
    require(set(selected) == set(OBJECT_METHODS), "Object10 MLP method set changed")
    result = {}
    for method in OBJECT_METHODS:
        row = selected[method]
        n, k = int(row["completed"]), int(row["successes"])
        require(n == 60 and int(row["planned"]) == n, f"Object10 {method} trial count")
        require(0 <= k <= n, f"Object10 {method} count invalid")
        require(abs(float(row["success_rate"]) - k / n) < 1e-9, f"Object10 {method} rate")
        result[method] = {
            "successes": k, "trials": n, "rate": round(k / n, 6),
            "mean_chunk_calls": round(float(row["mean_chunk_calls"]), 3),
            "mean_wall_s": round(float(row["mean_episode_wall_s"]), 3),
        }
    require(
        [result[m]["successes"] for m in OBJECT_METHODS] == [44, 51, 55, 55],
        "Object10 published reference totals disagree",
    )
    pairs = rows(base / "paired_comparisons.csv")
    p = [r for r in pairs if r["cohort"] == "night/mlp_s42"
         and r["baseline"] == "lvm" and r["candidate"] == "ogg"]
    require(len(p) == 1, "Expected one Object10 LVM/OGG paired comparison")
    pair = p[0]
    require(int(pair["matched"]) == 60 and int(pair["candidate_only_success"]) == 0
            and int(pair["baseline_only_success"]) == 0,
            "Object10 OGG incremental comparison changed")
    coverage = rows(base / "monitor_coverage.csv")
    mlp = [r for r in coverage if r["cohort"] == "night/mlp_s42"]
    require({r["method"] for r in mlp} == {"lvm", "ogg"}, "Object10 coverage methods")
    for r in coverage:
        eligible, scored = int(r["eligible_steps"]), int(r["scored_steps"])
        require(0 <= scored <= eligible and eligible > 0, "Invalid coverage counts")
        require(abs(float(r["coverage"]) - scored / eligible) < 1e-9, "Coverage ratio mismatch")
    return {"methods": result, "ogg_extra_success_over_lvm": 0,
            "mlp_coverage": {r["method"]: round(float(r["coverage"]), 6) for r in mlp}}


def cross_suite(root: Path) -> dict:
    base = root / "cross_suite"
    meta = read_json(base / "summary.json")
    trials = rows(base / "trials.csv")
    require(meta["status"] == "completed", "Cross-suite execution not finalized")
    require(len(trials) == meta["formal_trials"] == 880, "Cross-suite trial count mismatch")
    require(len({r["id"] for r in trials}) == len(trials), "Duplicate trial IDs")
    groups = defaultdict(set)
    primary = defaultdict(list)
    for r in trials:
        key = (r["suite"], r["task_id"], r["init"], r["seed"])
        require(r["method"] in CROSS_METHODS, f"Unknown cross-suite method {r['method']}")
        require(r["method"] not in groups[key], f"Repeated method for {key}")
        groups[key].add(r["method"])
        success = as_bool(r["success"])
        if r["phase"] == "primary" and r["suite"] in MAIN_SUITES:
            primary[(r["suite"], r["method"])].append(success)
    require(len(groups) == meta["paired_quartets"] == 220, "Paired quartet count")
    require(all(methods == set(CROSS_METHODS) for methods in groups.values()),
            "One or more cross-suite quartets missing a method")
    method_rows = rows(base / "methods.csv")
    published = {(r["suite"], r["method"]): r for r in method_rows
                 if r["cohort"] == "primary" and r["suite"] in MAIN_SUITES}
    outcome = {}
    for method in CROSS_METHODS:
        k = 0
        for suite in MAIN_SUITES:
            observed = primary[(suite, method)]
            require(len(observed) == 40, f"Cross-suite primary sample size {suite}/{method}")
            row = published[(suite, method)]
            require(int(row["completed"]) == 40 and int(row["successes"]) == sum(observed),
                    f"Cross-suite method aggregates differ: {suite}/{method}")
            k += sum(observed)
        declared = meta["overall_main_descriptive"][method]
        require(declared["primary_n"] == 120 and declared["primary_successes"] == k,
                f"Cross-suite summary disagrees for {method}")
        outcome[method] = {"successes": k, "trials": 120, "rate": round(k / 120, 6)}
    require([outcome[m]["successes"] for m in CROSS_METHODS] == [81, 93, 80, 77],
            "Published cross-suite totals changed")
    return {"formal_trials": len(trials), "paired_quartets": len(groups),
            "primary_suites": list(MAIN_SUITES), "methods": outcome,
            "notes": "Primary fixed protocol only; separate from extra initial states and LIBERO-90 subset."}


def plus_horizon(root: Path) -> dict:
    base = root / "plus_horizon"
    data = rows(base / "plus_horizon_results.csv")
    meta = read_json(base / "summary.json")
    grouped = defaultdict(list)
    for r in data:
        grouped[r["method"]].append(as_bool(r["success"]))
    result = {}
    for name, expected in [("H50", 7), ("H20", 20), ("H10", 19), ("H5", 15), ("H1", 6)]:
        runs = grouped[name]
        require(len(runs) == 32, f"Plus horizon trial count for {name}")
        require(sum(runs) == expected == meta["methods"][name]["successes"],
                f"Plus horizon {name} success mismatch")
        result[name] = {"successes": sum(runs), "trials": len(runs)}
    require(set(grouped) == set(result), "Unexpected Plus horizon method")
    return {"methods": result, "scope": "32 selected conditions, one evaluation seed"}


def summarize(root: Path = ROOT / "results") -> dict:
    return {
        "object10": object10(root),
        "cross_suite": cross_suite(root),
        "plus_horizon": plus_horizon(root),
    }


def markdown(result: dict) -> str:
    o, c, h = result["object10"], result["cross_suite"], result["plus_horizon"]
    text = [
        "# Recomputed evaluation results", "",
        "## Object10 · 60 paired conditions per method", "",
        "| Method | Successes | Rate | Mean VLA chunk calls | Mean wall time (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m in OBJECT_METHODS:
        z = o["methods"][m]
        text.append(f"| {m} | {z['successes']}/{z['trials']} | {100*z['rate']:.1f}% | {z['mean_chunk_calls']:.2f} | {z['mean_wall_s']:.2f} |")
    text.extend(["", f"OGG-only wins over LVM: {o['ogg_extra_success_over_lvm']}.",
                 "VLA calls and total wall time are different costs; do not conflate them.", "",
                 "## Cross-suite · primary comparison only", "",
                 "| Method | Spatial + Goal + Long (120 trials) |",
                 "| --- | ---: |"])
    for m in CROSS_METHODS:
        z = c["methods"][m]
        text.append(f"| {m} | {z['successes']}/{z['trials']} ({100*z['rate']:.1f}%) |")
    text.extend(["", f"Additional recorded trials: {c['formal_trials']} across {c['paired_quartets']} paired quartets, including extra initial states and a fixed LIBERO-90 subset.", "",
                 "## LIBERO-Plus execution horizon · selected 32 conditions", "",
                 "| Horizon | Successes |", "| --- | ---: |"])
    for m in ("H50", "H20", "H10", "H5", "H1"):
        z = h["methods"][m]
        text.append(f"| {m} | {z['successes']}/{z['trials']} |")
    return "\n".join(text) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "results")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    result = summarize(args.data_root)
    print(markdown(result) if args.format == "markdown"
          else json.dumps(result, indent=2, sort_keys=True), end="" if args.format == "markdown" else "\n")


if __name__ == "__main__":
    main()
