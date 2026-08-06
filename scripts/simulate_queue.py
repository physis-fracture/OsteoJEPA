"""E3 — does queue ordering save hours, and what does a new hospital need?

No GPU and no model. A discrete-event simulation feeds case arrivals into a
radiologist queue twice, once in arrival order and once in triage-score order,
and reports the difference in median time to adjudication.

**Every hour reported here comes from the assumptions in the table this script
prints, not from field measurement.** The table belongs beside the result in the
paper. Two assumptions matter more than the rest:

* Prevalence. 68.2% of test-fold studies carry a fracture, because
  GRAZPEDWRI-DX is a curated wrist-trauma cohort. Ordering a queue that is
  mostly positive can barely help - there is nothing to move ahead of. Results
  are reported across a prevalence sweep for that reason, and the dataset rate
  is the least representative point on it.
* Flag budget. Only the top fraction by score is promoted; the rest keep their
  arrival order. Promoting everything is not a triage policy, it is a re-sort.

The radiologist reads during a **day shift**, not continuously. That is the
whole premise: the on-call physician decides overnight and the radiologist
re-reads the same films the next morning, so cases arriving in the evening peak
wait until the shift opens. A continuously-staffed queue never backs up at these
arrival rates - median wait collapses to the read time itself - and with nothing
waiting there is nothing for an ordering to do.

    python scripts/simulate_queue.py --scores artifacts/clf/clf_main/scores_test.csv
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physis.data.geometry import band_list
from physis.utils.config import load_config
from physis.utils.run import setup_run

# Assumptions. Changing any of these changes every hour figure below.
DEFAULTS = {
    "peak_hours": (16, 21),          # Mattijssen-Horstink: 49% of missed fractures 16:00-21:00
    # Chosen so the base scenario runs at about 65% of reading capacity. A queue
    # at 97% utilisation does not back up, it collapses: the backlog grows without
    # bound and the "hours saved" it reports are an artefact of instability rather
    # than a property of the ordering.
    "arrivals_peak_per_hour": 4.0,
    "arrivals_offpeak_per_hour": 1.0,
    "radiologists_on_duty": 1,
    "reading_shift": (8, 16),        # the radiologist reads 08:00-16:00 only
    "read_minutes_per_case": 8.0,
    "flag_budget": 0.20,             # fraction of cases allowed into the priority lane
    "simulation_days": 30,
    "prevalences": (0.682, 0.40, 0.20, 0.10),
    "arrival_sensitivity": (0.75, 1.0, 1.25),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3 queue simulation and recalibration budget")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--val-scores", default=None, help="for the recalibration budget")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--run-subdir", default=None)
    return parser.parse_args()


def study_pool(scores_path: str) -> pd.DataFrame:
    """Per-study score and label: max over the study's images, as the product does."""
    frame = pd.read_csv(scores_path)
    return frame.groupby("study_id").agg(
        score=("score", "max"), label=("label", "max"), band=("band", "first")
    ).reset_index()


def sample_at_prevalence(pool: pd.DataFrame, prevalence: float, n: int, rng) -> pd.DataFrame:
    """Draw `n` studies whose positive rate matches `prevalence`.

    Sampling with replacement from each class separately. The score distribution
    within a class is preserved, which is what the ordering depends on.
    """
    positives = pool[pool["label"] == 1]
    negatives = pool[pool["label"] == 0]
    n_pos = int(round(n * prevalence))
    picked = pd.concat(
        [
            positives.sample(n_pos, replace=True, random_state=int(rng.integers(2**31))),
            negatives.sample(n - n_pos, replace=True, random_state=int(rng.integers(2**31))),
        ]
    )
    return picked.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def generate_arrivals(days: int, peak: tuple, peak_rate: float, offpeak_rate: float, rng):
    """Poisson arrivals with a diurnal rate, in hours since the simulation start."""
    times = []
    for day in range(days):
        for hour in range(24):
            rate = peak_rate if peak[0] <= hour < peak[1] else offpeak_rate
            count = rng.poisson(rate)
            if count:
                times.extend(day * 24 + hour + rng.random(count))
    return np.sort(np.asarray(times))


def next_shift_time(t: float, shift: tuple) -> float:
    """First moment at or after `t` when a radiologist is on duty."""
    start, end = shift
    day, hour = divmod(t, 24.0)
    if hour < start:
        return day * 24.0 + start
    if hour >= end:
        return (day + 1) * 24.0 + start
    return t


def simulate(arrivals: np.ndarray, scores: np.ndarray, *, servers: int,
             service_hours: float, priority: bool, flag_threshold: float,
             shift: tuple) -> np.ndarray:
    """Return time from arrival to completed read, in hours, per case.

    Under `priority`, cases scoring at or above `flag_threshold` are served
    before everything else; within each lane the order is by arrival. Cases
    below the threshold are never overtaken by other below-threshold cases, so
    the policy is a promotion, not a full re-sort.
    """
    n = len(arrivals)
    free = [0.0] * servers
    heapq.heapify(free)
    waiting: list = []
    waits = np.zeros(n)
    next_arrival = 0

    while next_arrival < n or waiting:
        now = next_shift_time(heapq.heappop(free), shift)
        while next_arrival < n and arrivals[next_arrival] <= now:
            _admit(waiting, next_arrival, arrivals, scores, priority, flag_threshold)
            next_arrival += 1
        if not waiting:
            if next_arrival >= n:
                heapq.heappush(free, now)
                break
            now = next_shift_time(max(now, arrivals[next_arrival]), shift)
            while next_arrival < n and arrivals[next_arrival] <= now:
                _admit(waiting, next_arrival, arrivals, scores, priority, flag_threshold)
                next_arrival += 1

        _, _, index = heapq.heappop(waiting)
        start = next_shift_time(max(now, arrivals[index]), shift)
        finish = start + service_hours
        waits[index] = finish - arrivals[index]
        heapq.heappush(free, finish)

    return waits


def _admit(waiting, index, arrivals, scores, priority, flag_threshold) -> None:
    lane = 0 if (priority and scores[index] >= flag_threshold) else 1
    heapq.heappush(waiting, (lane, arrivals[index], index))


def run_scenario(pool, assumptions, prevalence, arrival_multiplier, rng) -> dict:
    days = assumptions["simulation_days"]
    arrivals = generate_arrivals(
        days,
        assumptions["peak_hours"],
        assumptions["arrivals_peak_per_hour"] * arrival_multiplier,
        assumptions["arrivals_offpeak_per_hour"] * arrival_multiplier,
        rng,
    )
    cases = sample_at_prevalence(pool, prevalence, len(arrivals), rng)
    scores = cases["score"].to_numpy()
    labels = cases["label"].to_numpy().astype(bool)

    threshold = float(np.quantile(scores, 1.0 - assumptions["flag_budget"]))
    service = assumptions["read_minutes_per_case"] / 60.0

    shift_hours = assumptions["reading_shift"][1] - assumptions["reading_shift"][0]
    capacity = shift_hours / service * assumptions["radiologists_on_duty"]
    utilisation = (len(arrivals) / days) / capacity
    shared = dict(
        servers=int(assumptions["radiologists_on_duty"]),
        service_hours=service,
        flag_threshold=threshold,
        shift=assumptions["reading_shift"],
    )
    fifo = simulate(arrivals, scores, priority=False, **shared)
    physis = simulate(arrivals, scores, priority=True, **shared)

    return {
        "prevalence": prevalence,
        "arrival_multiplier": arrival_multiplier,
        "n_cases": int(len(arrivals)),
        "cases_per_day": float(len(arrivals) / days),
        "reading_capacity_per_day": float(capacity),
        "utilisation": float(utilisation),
        # Above ~0.95 the queue is not busy, it is unstable, and every number in
        # this row measures that instead of the policy.
        "saturated": bool(utilisation >= 0.95),
        "flag_threshold": threshold,
        "median_wait_all_fifo_h": float(np.median(fifo)),
        "median_wait_all_physis_h": float(np.median(physis)),
        "median_wait_positive_fifo_h": float(np.median(fifo[labels])),
        "median_wait_positive_physis_h": float(np.median(physis[labels])),
        "hours_saved_positive": float(np.median(fifo[labels]) - np.median(physis[labels])),
        "median_wait_negative_fifo_h": float(np.median(fifo[~labels])),
        "median_wait_negative_physis_h": float(np.median(physis[~labels])),
    }


def recalibration_budget(val_scores_path: str, cfg, rng, log) -> dict:
    """How many normal images per band a new site needs before the threshold settles.

    The product reports a percentile against normal images of the same age band,
    so moving to a new population means re-estimating that reference. This
    measures the spread of the estimated 95th percentile against sample size,
    which is the number a hospital has to be told before it can install anything.
    """
    frame = pd.read_csv(val_scores_path)
    manifest = pd.read_csv(cfg.manifest).set_index("stem")
    frame = frame.join(manifest[["clean_strict"]], on="stem")
    clean = frame[frame["clean_strict"]]
    bands = band_list(cfg)
    # A sigmoid trained to convergence saturates: clean images sit at 1e-4 and
    # fractures at 1 - 1e-7. Quantiles of that are unstable for reasons that have
    # nothing to do with sample size, so the reference distribution is estimated
    # on the logit, which is monotone and therefore leaves every ranking intact.
    logit = np.log(np.clip(clean["score"].to_numpy(), 1e-6, 1 - 1e-6))
    logit = logit - np.log1p(-np.clip(clean["score"].to_numpy(), 1e-6, 1 - 1e-6))
    clean = clean.assign(score=logit)

    out = []
    for n in (10, 25, 50, 100, 200):
        spreads = []
        for index in range(len(bands)):
            pool = clean[clean["band"] == index]["score"].to_numpy()
            if pool.size < 10:
                continue
            estimates = [
                np.quantile(rng.choice(pool, size=min(n, pool.size), replace=True), 0.95)
                for _ in range(200)
            ]
            spreads.append(float(np.std(estimates)))
        if spreads:
            out.append({"n_per_band": n, "mean_sd_of_q95": float(np.mean(spreads))})
            log.info("  n=%3d per band | sd of the estimated q95: %.4f", n, np.mean(spreads))
    return {"bootstrap": out}


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if not any(o.startswith("run.name=") for o in overrides):
        overrides.append("run.name=e3_queue")
    cfg = load_config(args.config, overrides)
    run = setup_run(cfg, subdir=args.run_subdir)
    log = run.log
    rng = np.random.default_rng(args.seed)

    pool = study_pool(args.scores)
    log.info(
        "%d studies in the pool, %.1f%% with a fracture",
        len(pool), 100 * pool["label"].mean(),
    )

    log.info("--- Table 1: simulation assumptions ---")
    for key, value in DEFAULTS.items():
        log.info("  %-28s %s", key, value)

    log.info("--- queue simulation ---")
    results = []
    for prevalence in DEFAULTS["prevalences"]:
        for multiplier in DEFAULTS["arrival_sensitivity"]:
            row = run_scenario(pool, DEFAULTS, prevalence, multiplier, rng)
            results.append(row)
            log.info(
                "  prevalence %.3f | arrivals x%.2f | util %.2f | fracture cases wait "
                "%.2f h FIFO vs %.2f h Physis | saved %+.2f h%s",
                prevalence, multiplier, row["utilisation"],
                row["median_wait_positive_fifo_h"],
                row["median_wait_positive_physis_h"],
                row["hours_saved_positive"],
                "  [SATURATED - not reportable]" if row["saturated"] else "",
            )

    report = {"assumptions": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in DEFAULTS.items()},
              "pool_prevalence": float(pool["label"].mean()),
              "scenarios": results}

    if args.val_scores:
        log.info("--- recalibration budget ---")
        report["recalibration"] = recalibration_budget(args.val_scores, cfg, rng, log)

    run.write_json("e3_queue.json", report)
    log.info("wrote %s", run.dir / "e3_queue.json")
    log.info(
        "Every hour above follows from Table 1 and from a simulated arrival "
        "process, not from measurement at a hospital."
    )


if __name__ == "__main__":
    main()
