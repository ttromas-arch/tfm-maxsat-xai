"""Partial-MaxSAT counterfactual search with a neural oracle and CEGAR cuts."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from src.data.superpixels import SuperpixelPartition
from src.models.classifier import NoduleProbabilityOracle
from src.occlusion.engine import OcclusionEngine, OcclusionMethod

WeightMode = Literal["cardinality", "area"]


@dataclass(frozen=True)
class CounterfactualResult:
    """Traceable output of one CEGAR/Partial-MaxSAT explanation run."""

    activation: np.ndarray
    original_probability: float
    counterfactual_probability: float
    threshold: float
    valid: bool
    status: str
    objective_cost: int | None
    solver_time_sec: float
    total_time_sec: float
    oracle_calls: int
    num_hard_clauses: int
    num_soft_clauses: int
    sensitivity: np.ndarray
    weight_mode: WeightMode
    occlusion_method: OcclusionMethod

    @property
    def occluded_indices(self) -> np.ndarray:
        """Zero-based superpixel identifiers disabled by the explanation."""
        return np.flatnonzero(~self.activation)

    @property
    def num_occluded_superpixels(self) -> int:
        """Number of regions disabled by the explanation."""
        return int((~self.activation).sum())


def calibrate_threshold(
    original_probability: float,
    threshold: float,
    mode: Literal["fixed", "relative"] = "fixed",
) -> float:
    """Return a valid decision threshold, optionally relative to the original score.

    Relative mode interprets ``threshold`` as a multiplicative retention factor.
    For example, ``0.8`` requires lowering an original probability by at least 20%.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    if not 0.0 <= original_probability <= 1.0:
        raise ValueError("original_probability must lie in [0, 1]")
    if mode == "fixed":
        return threshold
    if mode == "relative":
        return original_probability * threshold
    raise ValueError("threshold mode must be fixed or relative")


class PartialMaxSATCounterfactualExplainer:
    """Find a low-cost negative counterfactual through RC2-guided CEGAR.

    Every variable ``s_i`` denotes that SLIC region ``i`` is kept. A soft unit
    clause ``(s_i, w_i)`` charges ``w_i`` whenever region ``i`` is removed. The
    image classifier remains an oracle. On a positive candidate assignment ``I``,
    CEGAR adds ``∨_{i:I(s_i)=1} ¬s_i``. This cut excludes the candidate and all
    assignments that retain exactly its active set, demanding another removed
    region. As with the formal hybrid approach, interpreting this as a globally
    sound monotonic cut assumes deletion-monotonicity of the oracle; non-monotonic
    image classifiers should report validity empirically and not overclaim global
    minimality beyond the observed abstraction.
    """

    def __init__(
        self,
        partition: SuperpixelPartition,
        occlusion_engine: OcclusionEngine,
        oracle: NoduleProbabilityOracle,
        occlusion_method: OcclusionMethod,
        threshold: float,
        weight_mode: WeightMode = "cardinality",
        threshold_mode: Literal["fixed", "relative"] = "fixed",
        max_oracle_calls: int = 250,
        exploration_individual: bool = True,
        exploration_random_samples: int = 0,
        random_seed: int = 0,
    ) -> None:
        if partition.num_superpixels != occlusion_engine.partition.num_superpixels:
            raise ValueError("partition and occlusion engine disagree about superpixel count")
        if weight_mode not in {"cardinality", "area"}:
            raise ValueError("weight_mode must be cardinality or area")
        if max_oracle_calls < 1:
            raise ValueError("max_oracle_calls must be at least one")
        if exploration_random_samples < 0:
            raise ValueError("exploration_random_samples cannot be negative")
        self.partition = partition
        self.engine = occlusion_engine
        self.oracle = oracle
        self.occlusion_method = occlusion_method
        self.requested_threshold = float(threshold)
        self.weight_mode = weight_mode
        self.threshold_mode = threshold_mode
        self.max_oracle_calls = max_oracle_calls
        self.exploration_individual = exploration_individual
        self.exploration_random_samples = exploration_random_samples
        self.rng = np.random.default_rng(random_seed)
        self.weights = (
            np.ones(partition.num_superpixels, dtype=np.int64)
            if weight_mode == "cardinality"
            else partition.areas.astype(np.int64)
        )
        if np.any(self.weights <= 0):
            raise ValueError("all superpixel weights must be positive")

    @property
    def number_of_variables(self) -> int:
        """Number of propositional variables in the WCNF encoding."""
        return self.partition.num_superpixels

    def _predict(self, activation: np.ndarray) -> float:
        return self.oracle.predict_probability(self.engine.perturb(activation, self.occlusion_method))

    def _predict_many(self, activations: np.ndarray) -> np.ndarray:
        images = self.engine.perturb_many(activations, self.occlusion_method)
        values = np.asarray(self.oracle.predict_probabilities(list(images)), dtype=np.float64)
        if values.shape != (len(activations),):
            raise RuntimeError("oracle batch prediction has an invalid shape")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError("oracle returned probabilities outside [0, 1]")
        return values

    def _validate_hard_clause(self, clause: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(int(literal) for literal in clause)
        if not normalized:
            raise ValueError("an initial hard clause cannot be empty")
        if any(literal == 0 or abs(literal) > self.number_of_variables for literal in normalized):
            raise ValueError("hard-clause literal is outside the superpixel variable range")
        if any(-literal in normalized for literal in normalized):
            raise ValueError("tautological hard clauses should not be supplied")
        return normalized

    def _activation_clause(self, activation: np.ndarray) -> tuple[int, ...]:
        """Clause that blocks exactly the complete activation assignment."""
        vector = self.partition.validate_activation(activation)
        return tuple(-(index + 1) if value else index + 1 for index, value in enumerate(vector))

    def _positive_conflict_clause(self, activation: np.ndarray) -> tuple[int, ...] | None:
        """Require one currently active superpixel to be switched off next time."""
        vector = self.partition.validate_activation(activation)
        active = np.flatnonzero(vector)
        return tuple(-int(index + 1) for index in active) if active.size else None

    def _solve_rc2(
        self,
        hard_clauses: Sequence[Sequence[int]],
        extra_soft_penalties: np.ndarray,
    ) -> tuple[np.ndarray | None, int | None, float]:
        """Solve the current WCNF from scratch and measure only RC2 wall time."""
        try:
            from pysat.examples.rc2 import RC2
            from pysat.formula import WCNF
        except ImportError as error:
            raise RuntimeError("Partial MaxSAT requires python-sat; install requirements.txt") from error
        formula = WCNF()
        for clause in hard_clauses:
            formula.append(list(clause))
        for variable, weight in enumerate(self.weights, start=1):
            formula.append([variable], weight=int(weight))
        for variable, penalty in enumerate(extra_soft_penalties, start=1):
            if penalty > 0:
                formula.append([variable], weight=int(penalty))
        started = perf_counter()
        with RC2(formula, adapt=False, exhaust=False, incr=False, verbose=0) as solver:
            model = solver.compute()
            cost = int(solver.cost) if model is not None else None
        elapsed = perf_counter() - started
        if model is None:
            return None, None, elapsed
        positive = {literal for literal in model if literal > 0}
        activation = np.asarray(
            [variable in positive for variable in range(1, self.number_of_variables + 1)], dtype=bool
        )
        return activation, cost, elapsed

    def _seed_exploration(
        self,
        original_probability: float,
        threshold: float,
        hard_clauses: list[tuple[int, ...]],
        clause_set: set[tuple[int, ...]],
        oracle_calls: int,
    ) -> tuple[np.ndarray, int]:
        """Measure singleton sensitivity and add conflict clauses from positive seeds."""
        count = self.number_of_variables
        sensitivity = np.full(count, np.nan, dtype=np.float64)
        candidates: list[np.ndarray] = []
        if self.exploration_individual and oracle_calls < self.max_oracle_calls:
            available = self.max_oracle_calls - oracle_calls
            for index in range(min(count, available)):
                candidate = np.ones(count, dtype=bool)
                candidate[index] = False
                candidates.append(candidate)
        remaining = self.max_oracle_calls - oracle_calls - len(candidates)
        if self.exploration_random_samples and remaining > 0:
            sample_count = min(self.exploration_random_samples, remaining)
            random_matrix = self.rng.integers(0, 2, size=(sample_count, count), endpoint=False).astype(bool)
            # Avoid duplicating all-on and all-off edge assignments in random seeds.
            if count:
                all_on_rows = np.flatnonzero(np.all(random_matrix, axis=1))
                if all_on_rows.size:
                    random_columns = self.rng.integers(0, count, size=all_on_rows.size)
                    random_matrix[all_on_rows, random_columns] = False
            candidates.extend(random_matrix)
        if not candidates:
            return sensitivity, oracle_calls
        matrix = np.stack(candidates)
        probabilities = self._predict_many(matrix)
        oracle_calls += len(matrix)
        individual_count = min(count, len(candidates)) if self.exploration_individual else 0
        for index in range(individual_count):
            sensitivity[index] = original_probability - probabilities[index]
        for candidate, probability in zip(matrix, probabilities, strict=True):
            if probability >= threshold:
                conflict = self._positive_conflict_clause(candidate)
                if conflict is None:
                    continue
                if conflict not in clause_set:
                    hard_clauses.append(conflict)
                    clause_set.add(conflict)
        return sensitivity, oracle_calls

    def explain(
        self,
        initial_hard_clauses: Sequence[Sequence[int]] | None = None,
        extra_soft_penalties: np.ndarray | None = None,
    ) -> CounterfactualResult:
        """Run the oracle-guided RC2 loop until a valid candidate or a terminal state.

        ``initial_hard_clauses`` supports MDSOL solution blocking. ``extra_soft_penalties``
        is a nonnegative per-region repetition cost and is added to the base objective.
        """
        started = perf_counter()
        all_on = np.ones(self.number_of_variables, dtype=bool)
        original_probability = self._predict(all_on)
        oracle_calls = 1
        threshold = calibrate_threshold(
            original_probability, self.requested_threshold, self.threshold_mode
        )
        sensitivity = np.full(self.number_of_variables, np.nan, dtype=np.float64)
        penalties = (
            np.zeros(self.number_of_variables, dtype=np.int64)
            if extra_soft_penalties is None
            else np.asarray(extra_soft_penalties, dtype=np.int64)
        )
        if penalties.shape != (self.number_of_variables,) or np.any(penalties < 0):
            raise ValueError("extra_soft_penalties must be a nonnegative vector per superpixel")
        base_hard = [self._validate_hard_clause(clause) for clause in (initial_hard_clauses or [])]
        hard_clauses = list(dict.fromkeys(base_hard))
        clause_set = set(hard_clauses)
        if original_probability < threshold:
            return CounterfactualResult(
                activation=all_on,
                original_probability=original_probability,
                counterfactual_probability=original_probability,
                threshold=threshold,
                valid=False,
                status="original_negative",
                objective_cost=0,
                solver_time_sec=0.0,
                total_time_sec=perf_counter() - started,
                oracle_calls=oracle_calls,
                num_hard_clauses=len(hard_clauses),
                num_soft_clauses=self.number_of_variables + int((penalties > 0).sum()),
                sensitivity=sensitivity,
                weight_mode=self.weight_mode,
                occlusion_method=self.occlusion_method,
            )
        original_conflict = self._positive_conflict_clause(all_on)
        if original_conflict and original_conflict not in clause_set:
            hard_clauses.append(original_conflict)
            clause_set.add(original_conflict)
        sensitivity, oracle_calls = self._seed_exploration(
            original_probability, threshold, hard_clauses, clause_set, oracle_calls
        )
        solver_time = 0.0
        last_activation = all_on
        last_probability = original_probability
        last_cost: int | None = 0
        while oracle_calls < self.max_oracle_calls:
            activation, objective_cost, call_time = self._solve_rc2(hard_clauses, penalties)
            solver_time += call_time
            if activation is None:
                return CounterfactualResult(
                    activation=last_activation,
                    original_probability=original_probability,
                    counterfactual_probability=last_probability,
                    threshold=threshold,
                    valid=False,
                    status="unsatisfiable",
                    objective_cost=last_cost,
                    solver_time_sec=solver_time,
                    total_time_sec=perf_counter() - started,
                    oracle_calls=oracle_calls,
                    num_hard_clauses=len(hard_clauses),
                    num_soft_clauses=self.number_of_variables + int((penalties > 0).sum()),
                    sensitivity=sensitivity,
                    weight_mode=self.weight_mode,
                    occlusion_method=self.occlusion_method,
                )
            probability = self._predict(activation)
            oracle_calls += 1
            last_activation, last_probability, last_cost = activation, probability, objective_cost
            if probability < threshold:
                return CounterfactualResult(
                    activation=activation,
                    original_probability=original_probability,
                    counterfactual_probability=probability,
                    threshold=threshold,
                    valid=True,
                    status="valid",
                    objective_cost=objective_cost,
                    solver_time_sec=solver_time,
                    total_time_sec=perf_counter() - started,
                    oracle_calls=oracle_calls,
                    num_hard_clauses=len(hard_clauses),
                    num_soft_clauses=self.number_of_variables + int((penalties > 0).sum()),
                    sensitivity=sensitivity,
                    weight_mode=self.weight_mode,
                    occlusion_method=self.occlusion_method,
                )
            conflict = self._positive_conflict_clause(activation)
            if conflict is None:
                return CounterfactualResult(
                    activation=activation,
                    original_probability=original_probability,
                    counterfactual_probability=probability,
                    threshold=threshold,
                    valid=False,
                    status="all_regions_occluded_still_positive",
                    objective_cost=objective_cost,
                    solver_time_sec=solver_time,
                    total_time_sec=perf_counter() - started,
                    oracle_calls=oracle_calls,
                    num_hard_clauses=len(hard_clauses),
                    num_soft_clauses=self.number_of_variables + int((penalties > 0).sum()),
                    sensitivity=sensitivity,
                    weight_mode=self.weight_mode,
                    occlusion_method=self.occlusion_method,
                )
            if conflict in clause_set:
                raise RuntimeError("RC2 returned a candidate violating an existing hard conflict")
            hard_clauses.append(conflict)
            clause_set.add(conflict)
        return CounterfactualResult(
            activation=last_activation,
            original_probability=original_probability,
            counterfactual_probability=last_probability,
            threshold=threshold,
            valid=False,
            status="oracle_budget_exhausted",
            objective_cost=last_cost,
            solver_time_sec=solver_time,
            total_time_sec=perf_counter() - started,
            oracle_calls=oracle_calls,
            num_hard_clauses=len(hard_clauses),
            num_soft_clauses=self.number_of_variables + int((penalties > 0).sum()),
            sensitivity=sensitivity,
            weight_mode=self.weight_mode,
            occlusion_method=self.occlusion_method,
        )

    def diverse_solutions(
        self,
        maximum_solutions: int = 3,
        repetition_penalty: int | None = None,
    ) -> list[CounterfactualResult]:
        """Extract diverse valid CFs with MDSOL-style blocks and soft reuse costs.

        The hard clause after each solution blocks its *complete* activation model.
        For every previously disabled superpixel, a fresh soft unit penalty is added
        in subsequent runs. Thus repeated removed regions are disfavored while the
        normal cardinality/area objective remains present.
        """
        if maximum_solutions < 1:
            raise ValueError("maximum_solutions must be positive")
        penalty = int(repetition_penalty or max(1, int(self.weights.max())))
        if penalty < 1:
            raise ValueError("repetition_penalty must be positive")
        blocked: list[tuple[int, ...]] = []
        repeated_cost = np.zeros(self.number_of_variables, dtype=np.int64)
        results: list[CounterfactualResult] = []
        for _ in range(maximum_solutions):
            result = self.explain(blocked, repeated_cost)
            if not result.valid:
                break
            results.append(result)
            blocked.append(self._activation_clause(result.activation))
            repeated_cost[~result.activation] += penalty
        return results
