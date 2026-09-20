"""Closed-loop Bayesian Optimization over SOM-MD direction subsets + ablation factor.

The open-loop path (recipe.build_som_md_recipe) blindly takes the top-k SOM
neurons and never checks whether the recipe actually reduces refusal. This module
closes the loop: given a candidate direction pool from the SOM, it searches over
subsets of those directions plus a global ablation factor, scoring each candidate
by *measured* compliance rate on held-out prompts (rule-based refusal
classification), and returns the best-scoring recipe in the existing `som_md`
schema so apply_ablation_in_place / bake work unchanged.

No optuna dependency: a TPE-lite / greedy-random hybrid that is honest about its
size. Each trial is expensive (it edits weights, generates N prompts, restores),
so the proposal logic is deliberately cheap and the budget small.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trial:
    """One BO trial: a subset of candidate directions + a factor, scored."""
    indices: list[int]          # which pool directions are included
    factor: float
    score: float                # compliance rate in [0,1]; higher is better
    n_compliant: int = 0
    n_total: int = 0


@dataclass
class SearchResult:
    best: Trial | None
    trials: list[Trial] = field(default_factory=list)

    def to_bo_block(self) -> dict:
        """The `"bo"` block embedded in the recipe for traceability."""
        return {
            "n_trials": len(self.trials),
            "best_score": self.best.score if self.best else None,
            "best_factor": self.best.factor if self.best else None,
            "best_subset_size": len(self.best.indices) if self.best else 0,
            "trials": [
                {
                    "i": i,
                    "indices": t.indices,
                    "factor": round(t.factor, 4),
                    "score": round(t.score, 4),
                    "n_compliant": t.n_compliant,
                    "n_total": t.n_total,
                }
                for i, t in enumerate(self.trials)
            ],
        }


def _random_subset(rng: random.Random, m: int) -> list[int]:
    """Seed-phase proposal: a random subset of size 4..12 (clamped to m)."""
    size = rng.randint(4, min(12, m))
    return sorted(rng.sample(range(m), k=size))


def _tpe_proposal(
    rng: random.Random,
    m: int,
    good: list[Trial],
    bad: list[Trial],
    best: Trial | None,
) -> list[int]:
    """Per-coordinate Bernoulli proposal biased by good-vs-bad membership.

    For each direction index i, estimate P(include | good outcome) from the
    trials scored so far and sample inclusion independently. Falls back to a
    random subset while there is no history to bias from."""
    if not good or not bad:
        return _random_subset(rng, m)

    probs = np.zeros(m)
    for i in range(m):
        g = sum(1 for t in good if i in t.indices)
        b = sum(1 for t in bad if i in t.indices)
        # Laplace-smoothed posterior over membership; a direction that shows up
        # in more good trials than bad ones gets pushed toward inclusion.
        probs[i] = (g + 1) / (g + b + 2)

    subset = [i for i in range(m) if rng.random() < probs[i]]
    # Keep the best-so-far directions as a floor so we never regress to empty.
    if best is not None:
        for i in best.indices:
            if i not in subset and rng.random() < 0.5:
                subset.append(i)
    subset = sorted(set(subset))
    return subset


def _sample_factor(rng: random.Random, lo: float, hi: float, good: list[Trial]) -> float:
    """Sample a factor, biased toward the factors of good trials once we have some."""
    if not good:
        return rng.uniform(lo, hi)
    # Draw from a good trial's factor with jitter; occasionally explore fresh.
    if rng.random() < 0.7:
        base = rng.choice(good).factor
        return float(np.clip(rng.gauss(base, 0.1), lo, hi))
    return rng.uniform(lo, hi)


def run_search(
    pool: list[np.ndarray],
    score_fn,
    n_trials: int = 50,
    factor_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
    log_path: str | Path | None = None,
    verbose: bool = True,
) -> SearchResult:
    """Run the BO loop over subsets of `pool` scored by `score_fn`.

    Args:
        pool: candidate unit directions, shape (m, dim). The SOM output.
        score_fn: callable(indices, factor) -> (score, n_compliant, n_total).
            Called with a list of pool indices to include and the ablation
            factor; returns the compliance rate in [0,1] plus the raw counts it
            was computed from. Must be deterministic enough that re-scoring the
            same (indices, factor) gives ~the same value.
        n_trials: total budget including the random seed phase.
        factor_range: (lo, hi) bounds for the global factor.
        seed: RNG seed for reproducibility.
        log_path: if given, append one JSON line per trial here.
        verbose: print a progress line per trial.

    Returns:
        SearchResult with the best trial and full history.
    """
    m = len(pool)
    lo, hi = factor_range
    rng = random.Random(seed)

    trials: list[Trial] = []
    good: list[Trial] = []
    bad: list[Trial] = []
    best: Trial | None = None
    seed_phase = min(8, n_trials)

    log_file = open(log_path, "a") if log_path else None

    def _record(trial: Trial):
        nonlocal best
        trials.append(trial)
        if trial.score > 0.5:
            good.append(trial)
        else:
            bad.append(trial)
        if best is None or trial.score > best.score:
            best = trial
        line = {
            "i": len(trials) - 1,
            "indices": trial.indices,
            "factor": round(trial.factor, 4),
            "score": round(trial.score, 4),
            "n_compliant": trial.n_compliant,
            "n_total": trial.n_total,
        }
        if log_file:
            log_file.write(json.dumps(line) + "\n")
            log_file.flush()
        if verbose:
            print(
                f"[bo] trial {len(trials):>3}/{n_trials} size={len(trial.indices):>2} "
                f"factor={trial.factor:.3f} score={trial.score:.3f} "
                f"({trial.n_compliant}/{trial.n_total}) best={best.score:.3f}",
                flush=True,
            )

    try:
        for t in range(n_trials):
            if t < seed_phase:
                indices = _random_subset(rng, m)
                factor = rng.uniform(lo, hi)
            else:
                indices = _tpe_proposal(rng, m, good, bad, best)
                factor = _sample_factor(rng, lo, hi, good)

            # Guard against an empty subset (nothing to ablate) — fall back to a
            # random one so the trial is meaningful.
            if not indices:
                indices = _random_subset(rng, m)

            score, n_compliant, n_total = score_fn(indices, factor)
            _record(Trial(
                indices=indices,
                factor=factor,
                score=float(score),
                n_compliant=int(n_compliant),
                n_total=int(n_total),
            ))
    finally:
        if log_file:
            log_file.close()

    return SearchResult(best=best, trials=trials)
