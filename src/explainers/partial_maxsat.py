"""Non-monotonic Partial-MaxSAT counterfactual search for neural image oracles.

The image classifier is an arbitrary black-box oracle, not a monotone Boolean
formula. Consequently, the default CEGAR cut blocks only the exact failed assignment.
The stronger expansion cut from monotone hitting-set formulations remains available
for ablation, but is never presented as a sound non-monotonic inference rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Sequence

import numpy as np

from src.data.superpixels import SuperpixelPartition
from src.models.classifier import NoduleProbabilityOracle
from src.occlusion.engine import OcclusionEngine, OcclusionMethod

WeightMode = Literal["cardinality", "area"]
CegarCutMode = Literal["exact", "expansion", "hybrid"]
SensitivityNormalization = Literal["rank", "minmax"]


@dataclass
class SearchState:
    """Mutable state shared by seed exploration, CEGAR lanes and MDSOL.

    Oracle cache keys are complete global activation vectors. This prevents MDSOL
    from repeating seed or failed-candidate inferences and makes the call budget a
    true per-image/per-occlusion-method budget.
    """

    original_probability: float | None = None
    threshold: float | None = None
    sensitivity: np.ndarray | None = None
    seed_probabilities: np.ndarray | None = None
    oracle_cache: dict[tuple[bool, ...], float] = field(default_factory=dict)
    oracle_calls: int = 0
    oracle_cache_hits: int = 0
    oracle_cache_misses: int = 0
    solver_time_sec: float = 0.0
    exact_block_count: int = 0
    expansion_cut_count: int = 0
    exact_block_clauses: set[tuple[int, ...]] = field(default_factory=set)
    expansion_cut_clauses: set[tuple[int, ...]] = field(default_factory=set)
    explored_assignments: int = 0
    seed_completed: bool = False
    guidance_penalties: np.ndarray | None = None


@dataclass(frozen=True)
class CounterfactualResult:
    """Traceable output of one non-monotonic Partial-MaxSAT explanation."""

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
    seed_probabilities: np.ndarray
    weight_mode: WeightMode
    occlusion_method: OcclusionMethod
    cegar_cut_mode: CegarCutMode
    search_lane: str
    solver_superpixel_indices: np.ndarray
    frozen_outside_lung_indices: np.ndarray
    base_weights: np.ndarray
    guidance_penalties: np.ndarray
    seed_guidance_weight: float
    anti_sensitivity_penalty: float
    low_sensitivity_penalty: float
    sensitivity_normalization: SensitivityNormalization
    minimum_lung_overlap: float
    hybrid_exact_budget_fraction: float
    exact_block_count: int
    expansion_cut_count: int
    oracle_cache_hits: int
    oracle_cache_misses: int

    @property
    def occluded_indices(self) -> np.ndarray:
        """Zero-based global superpixel identifiers disabled by the explanation."""
        return np.flatnonzero(~self.activation)

    @property
    def num_occluded_superpixels(self) -> int:
        """Number of global SLIC regions disabled by the explanation."""
        return int((~self.activation).sum())

    @property
    def selected_mean_seed_drop(self) -> float:
        """Mean singleton probability drop for regions selected in this solution."""
        selected = self.sensitivity[~self.activation]
        finite = selected[np.isfinite(selected)]
        return float(finite.mean()) if finite.size else float("nan")

    @property
    def selected_min_seed_drop(self) -> float:
        """Minimum singleton probability drop among regions selected in this solution."""
        selected = self.sensitivity[~self.activation]
        finite = selected[np.isfinite(selected)]
        return float(finite.min()) if finite.size else float("nan")

    @property
    def selected_num_artifact_increasing_regions(self) -> int:
        """Number of selected regions whose singleton occlusion increased probability."""
        return int(np.sum(self.sensitivity[~self.activation] < 0.0))


def _json_float_list(values: np.ndarray) -> list[float | None]:
    """Serialize an array without leaking NaN values into strict JSON traces."""
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=float)]


def counterfactual_trace(
    result: CounterfactualResult,
    image_id: str,
    solution_index: int,
) -> dict[str, object]:
    """Convert one result into an auditable, JSON-serializable explanation trace."""
    return {
        "image_id": image_id,
        "occlusion_method": result.occlusion_method,
        "solution_index": int(solution_index),
        "activation": result.activation.astype(int).tolist(),
        "occluded_superpixel_indices": result.occluded_indices.astype(int).tolist(),
        "sensitivity": _json_float_list(result.sensitivity),
        "seed_probabilities": _json_float_list(result.seed_probabilities),
        "original_probability": float(result.original_probability),
        "counterfactual_probability": float(result.counterfactual_probability),
        "threshold": float(result.threshold),
        "valid": bool(result.valid),
        "status": result.status,
        "weight_mode": result.weight_mode,
        "cegar_cut_mode": result.cegar_cut_mode,
        "search_lane": result.search_lane,
        "solver_superpixel_indices": result.solver_superpixel_indices.astype(int).tolist(),
        "frozen_outside_lung_indices": result.frozen_outside_lung_indices.astype(int).tolist(),
        "base_weights": result.base_weights.astype(int).tolist(),
        "guidance_penalties": result.guidance_penalties.astype(int).tolist(),
        "seed_guidance_weight": float(result.seed_guidance_weight),
        "anti_sensitivity_penalty": float(result.anti_sensitivity_penalty),
        "low_sensitivity_penalty": float(result.low_sensitivity_penalty),
        "sensitivity_normalization": result.sensitivity_normalization,
        "minimum_lung_overlap": float(result.minimum_lung_overlap),
        "hybrid_exact_budget_fraction": float(result.hybrid_exact_budget_fraction),
        "objective_cost": result.objective_cost,
        "oracle_calls": int(result.oracle_calls),
        "oracle_cache_hits": int(result.oracle_cache_hits),
        "oracle_cache_misses": int(result.oracle_cache_misses),
        "exact_block_count": int(result.exact_block_count),
        "expansion_cut_count": int(result.expansion_cut_count),
    }


def calibrate_threshold(
    original_probability: float,
    threshold: float,
    mode: Literal["fixed", "relative"] = "fixed",
) -> float:
    """Return a valid decision threshold, optionally relative to original score."""
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
    """Search minimal counterfactuals under an explicitly non-monotonic oracle.

    A global variable ``s_i`` is true when SLIC region ``i`` is retained. Only
    regions with nonzero lung overlap enter RC2. The default ``exact`` CEGAR mode
    blocks a failed complete assignment ``I`` with

    ``∨_{I(s_i)=1} ¬s_i ∨ ∨_{I(s_i)=0} s_i``.

    Unlike an expansion-only cut, this is valid for arbitrary non-monotonic neural
    decision functions because it excludes no unobserved alternative assignment.
    ``expansion`` and ``hybrid`` are retained strictly as documented ablations.
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
        cegar_cut_mode: CegarCutMode = "exact",
        seed_guidance_weight: float = 1.0,
        anti_sensitivity_penalty: float = 4.0,
        low_sensitivity_penalty: float = 1.0,
        sensitivity_normalization: SensitivityNormalization = "rank",
        minimum_lung_overlap: float = 0.0,
        hybrid_exact_budget_fraction: float = 0.5,
    ) -> None:
        if partition.num_superpixels != occlusion_engine.partition.num_superpixels:
            raise ValueError("partition and occlusion engine disagree about superpixel count")
        if weight_mode not in {"cardinality", "area"}:
            raise ValueError("weight_mode must be cardinality or area")
        if cegar_cut_mode not in {"exact", "expansion", "hybrid"}:
            raise ValueError("cegar_cut_mode must be exact, expansion or hybrid")
        if sensitivity_normalization not in {"rank", "minmax"}:
            raise ValueError("sensitivity_normalization must be rank or minmax")
        if max_oracle_calls < 1:
            raise ValueError("max_oracle_calls must be at least one")
        if exploration_random_samples < 0:
            raise ValueError("exploration_random_samples cannot be negative")
        if not exploration_individual:
            raise ValueError("individual singleton seed exploration is mandatory")
        if min(seed_guidance_weight, anti_sensitivity_penalty, low_sensitivity_penalty) < 0:
            raise ValueError("sensitivity guidance parameters must be nonnegative")
        if not 0.0 <= minimum_lung_overlap <= 1.0:
            raise ValueError("minimum_lung_overlap must lie in [0, 1]")
        if not 0.0 < hybrid_exact_budget_fraction < 1.0:
            raise ValueError("hybrid_exact_budget_fraction must lie strictly between zero and one")
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
        self.cegar_cut_mode = cegar_cut_mode
        self.seed_guidance_weight = float(seed_guidance_weight)
        self.anti_sensitivity_penalty = float(anti_sensitivity_penalty)
        self.low_sensitivity_penalty = float(low_sensitivity_penalty)
        self.sensitivity_normalization = sensitivity_normalization
        self.minimum_lung_overlap = float(minimum_lung_overlap)
        self.hybrid_exact_budget_fraction = float(hybrid_exact_budget_fraction)
        self.rng = np.random.default_rng(random_seed)
        self.base_weights = (
            np.ones(partition.num_superpixels, dtype=np.int64)
            if weight_mode == "cardinality"
            else partition.areas.astype(np.int64)
        )
        if np.any(self.base_weights <= 0):
            raise ValueError("all superpixel base weights must be positive")
        self.lung_overlap = self._compute_lung_overlap()
        self.solver_global_indices = np.flatnonzero(self.lung_overlap > self.minimum_lung_overlap).astype(np.int64)
        self.frozen_outside_lung_indices = np.flatnonzero(self.lung_overlap <= self.minimum_lung_overlap).astype(np.int64)
        if not self.solver_global_indices.size:
            raise ValueError("no superpixel has sufficient lung overlap for MaxSAT")
        self.global_to_solver_index = np.full(partition.num_superpixels, -1, dtype=np.int64)
        self.global_to_solver_index[self.solver_global_indices] = np.arange(
            self.solver_global_indices.size, dtype=np.int64
        )
        if self.exploration_individual and self.max_oracle_calls < self.number_of_variables + 1:
            raise ValueError(
                "max_oracle_calls must cover original inference plus every eligible singleton "
                f"({self.number_of_variables + 1} required)"
            )
        self.last_result: CounterfactualResult | None = None

    def _compute_lung_overlap(self) -> np.ndarray:
        """Compute exact fraction of every global superpixel inside the lung mask."""
        labels, lung = self.partition.labels, self.partition.anatomy_mask
        valid = labels >= 0
        inside = np.bincount(labels[valid & lung], minlength=self.partition.num_superpixels)
        return np.divide(
            inside,
            self.partition.areas,
            out=np.zeros(self.partition.num_superpixels, dtype=np.float64),
            where=self.partition.areas > 0,
        )

    @property
    def number_of_variables(self) -> int:
        """Number of RC2 variables after anatomical filtering."""
        return int(self.solver_global_indices.size)

    def _all_on_global_activation(self) -> np.ndarray:
        return np.ones(self.partition.num_superpixels, dtype=bool)

    def _to_global_activation(self, solver_activation: np.ndarray) -> np.ndarray:
        local = np.asarray(solver_activation, dtype=bool)
        if local.shape != (self.number_of_variables,):
            raise ValueError("solver activation has an invalid length")
        global_activation = self._all_on_global_activation()
        global_activation[self.solver_global_indices] = local
        return global_activation

    def _to_solver_activation(self, global_activation: np.ndarray) -> np.ndarray:
        global_vector = self.partition.validate_activation(global_activation)
        if not np.all(global_vector[self.frozen_outside_lung_indices]):
            raise ValueError("an anatomically frozen superpixel cannot be disabled")
        return global_vector[self.solver_global_indices]

    @staticmethod
    def _activation_key(activation: np.ndarray) -> tuple[bool, ...]:
        return tuple(bool(value) for value in np.asarray(activation, dtype=bool))

    def _query_oracle(self, activation: np.ndarray, state: SearchState) -> float:
        """Return a cached global-activation probability while enforcing total budget."""
        global_vector = self.partition.validate_activation(activation)
        key = self._activation_key(global_vector)
        if key in state.oracle_cache:
            state.oracle_cache_hits += 1
            return state.oracle_cache[key]
        if state.oracle_calls >= self.max_oracle_calls:
            raise RuntimeError("oracle budget exhausted")
        probability = float(self.oracle.predict_probability(self.engine.perturb(global_vector, self.occlusion_method)))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("oracle returned a probability outside [0, 1]")
        state.oracle_cache[key] = probability
        state.oracle_calls += 1
        state.oracle_cache_misses += 1
        return probability

    def _query_oracle_many(self, activations: np.ndarray, state: SearchState) -> np.ndarray:
        """Query a global activation batch while preserving cache and total budget."""
        matrix = np.asarray(activations, dtype=bool)
        if matrix.ndim != 2 or matrix.shape[1] != self.partition.num_superpixels:
            raise ValueError("activations must be a global (batch, num_superpixels) binary matrix")
        result = np.empty(matrix.shape[0], dtype=np.float64)
        uncached_positions: dict[tuple[bool, ...], list[int]] = {}
        uncached_images: list[np.ndarray] = []
        uncached_keys: list[tuple[bool, ...]] = []
        for index, activation in enumerate(matrix):
            key = self._activation_key(activation)
            if key in state.oracle_cache:
                result[index] = state.oracle_cache[key]
                state.oracle_cache_hits += 1
            elif key in uncached_positions:
                uncached_positions[key].append(index)
                state.oracle_cache_hits += 1
            else:
                uncached_positions[key] = [index]
                uncached_keys.append(key)
                uncached_images.append(self.engine.perturb(activation, self.occlusion_method))
        if state.oracle_calls + len(uncached_images) > self.max_oracle_calls:
            raise RuntimeError("oracle budget exhausted")
        if uncached_images:
            probabilities = np.asarray(self.oracle.predict_probabilities(uncached_images), dtype=np.float64)
            if probabilities.shape != (len(uncached_images),):
                raise RuntimeError("oracle batch prediction has an invalid shape")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
                raise RuntimeError("oracle returned probabilities outside [0, 1]")
            for key, probability in zip(uncached_keys, probabilities, strict=True):
                value = float(probability)
                state.oracle_cache[key] = value
                for position in uncached_positions[key]:
                    result[position] = value
            state.oracle_calls += len(uncached_images)
            state.oracle_cache_misses += len(uncached_images)
        return result

    def _exact_block_clause(self, solver_activation: np.ndarray) -> tuple[int, ...]:
        """Block exactly one complete local assignment, safe for non-monotonic oracles."""
        local = np.asarray(solver_activation, dtype=bool)
        return tuple(-(index + 1) if value else index + 1 for index, value in enumerate(local))

    def _expansion_cut_clause(self, solver_activation: np.ndarray) -> tuple[int, ...] | None:
        """Require an extra active local variable to be disabled; monotonic heuristic only."""
        active = np.flatnonzero(np.asarray(solver_activation, dtype=bool))
        return tuple(-int(index + 1) for index in active) if active.size else None

    def _normalize_sensitivity(self, sensitivity: np.ndarray) -> np.ndarray:
        """Map finite singleton drops to [0, 1], where one is most favorable."""
        normalized = np.zeros(self.partition.num_superpixels, dtype=np.float64)
        values = sensitivity[self.solver_global_indices]
        finite = np.isfinite(values)
        if not finite.any():
            return normalized
        finite_values = values[finite]
        if self.sensitivity_normalization == "minmax":
            low, high = float(finite_values.min()), float(finite_values.max())
            mapped = np.full(finite_values.shape, 0.5) if high == low else (finite_values - low) / (high - low)
        else:
            order = np.argsort(np.argsort(finite_values, kind="stable"), kind="stable")
            mapped = np.full(finite_values.shape, 0.5) if finite_values.size == 1 else order / (finite_values.size - 1)
        local = np.zeros(values.shape, dtype=np.float64)
        local[finite] = mapped
        normalized[self.solver_global_indices] = local
        return normalized

    def _guidance_penalties(self, sensitivity: np.ndarray) -> np.ndarray:
        """Build integer tiebreak costs from singleton evidence without changing base optimum."""
        normalized = self._normalize_sensitivity(sensitivity)
        penalties = np.zeros(self.partition.num_superpixels, dtype=np.int64)
        eligible = self.solver_global_indices
        finite = np.isfinite(sensitivity[eligible])
        raw = self.seed_guidance_weight * self.low_sensitivity_penalty * (1.0 - normalized[eligible])
        raw += self.seed_guidance_weight * self.anti_sensitivity_penalty * np.maximum(
            0.0, -sensitivity[eligible]
        )
        raw[~finite] = self.seed_guidance_weight * self.low_sensitivity_penalty
        penalties[eligible] = np.rint(raw * 1000.0).astype(np.int64)
        return penalties

    def _lexicographic_weights(self, guidance_penalties: np.ndarray, repetition_penalties: np.ndarray) -> np.ndarray:
        """Encode primary area/cardinality then heuristic tiebreaks in one WCNF weight."""
        secondary = np.asarray(guidance_penalties + repetition_penalties, dtype=np.int64)
        if secondary.shape != (self.partition.num_superpixels,) or np.any(secondary < 0):
            raise ValueError("secondary penalties must be nonnegative global vectors")
        maximum_secondary_sum = int(secondary[self.solver_global_indices].sum())
        multiplier = maximum_secondary_sum + 1
        return self.base_weights * multiplier + secondary

    def _solve_rc2(
        self,
        hard_clauses: Sequence[Sequence[int]],
        global_weights: np.ndarray,
        state: SearchState,
    ) -> tuple[np.ndarray | None, int | None]:
        """Solve a local WCNF once, accumulating pure RC2 time in shared state."""
        try:
            from pysat.examples.rc2 import RC2
            from pysat.formula import WCNF
        except ImportError as error:
            raise RuntimeError("Partial MaxSAT requires python-sat; install requirements.txt") from error
        weights = np.asarray(global_weights, dtype=np.int64)[self.solver_global_indices]
        formula = WCNF()
        for clause in hard_clauses:
            if not clause:
                raise ValueError("hard clauses cannot be empty")
            formula.append([int(literal) for literal in clause])
        for variable, weight in enumerate(weights, start=1):
            formula.append([variable], weight=int(weight))
        started = perf_counter()
        with RC2(formula, adapt=False, exhaust=False, incr=False, verbose=0) as solver:
            model = solver.compute()
            objective = int(solver.cost) if model is not None else None
        state.solver_time_sec += perf_counter() - started
        if model is None:
            return None, None
        positives = {literal for literal in model if literal > 0}
        local = np.asarray([variable in positives for variable in range(1, self.number_of_variables + 1)], bool)
        return local, objective

    def _run_seed_exploration(self, state: SearchState) -> None:
        """Evaluate every anatomically eligible singleton exactly once before RC2."""
        if state.seed_completed:
            return
        all_on = self._all_on_global_activation()
        if state.original_probability is None:
            state.original_probability = self._query_oracle(all_on, state)
            state.threshold = calibrate_threshold(
                state.original_probability, self.requested_threshold, self.threshold_mode
            )
        sensitivity = np.full(self.partition.num_superpixels, np.nan, dtype=np.float64)
        seed_probabilities = np.full(self.partition.num_superpixels, np.nan, dtype=np.float64)
        candidates = np.tile(all_on, (self.number_of_variables, 1))
        candidates[np.arange(self.number_of_variables), self.solver_global_indices] = False
        available = self.max_oracle_calls - state.oracle_calls
        evaluated = min(self.number_of_variables, max(0, available))
        if evaluated:
            probabilities = self._query_oracle_many(candidates[:evaluated], state)
            globals_evaluated = self.solver_global_indices[:evaluated]
            seed_probabilities[globals_evaluated] = probabilities
            sensitivity[globals_evaluated] = state.original_probability - probabilities
        remaining = self.max_oracle_calls - state.oracle_calls
        if self.exploration_random_samples and remaining > 0:
            sample_count = min(self.exploration_random_samples, remaining)
            local_random = self.rng.integers(0, 2, size=(sample_count, self.number_of_variables), endpoint=False).astype(bool)
            all_on_rows = np.flatnonzero(np.all(local_random, axis=1))
            if all_on_rows.size:
                local_random[all_on_rows, self.rng.integers(0, self.number_of_variables, size=all_on_rows.size)] = False
            random_global = np.tile(all_on, (sample_count, 1))
            random_global[:, self.solver_global_indices] = local_random
            self._query_oracle_many(random_global, state)
        state.sensitivity, state.seed_probabilities, state.seed_completed = sensitivity, seed_probabilities, True

    def _result(
        self,
        state: SearchState,
        activation: np.ndarray,
        probability: float,
        valid: bool,
        status: str,
        objective_cost: int | None,
        hard_clauses: Sequence[Sequence[int]],
        global_weights: np.ndarray,
        search_lane: str,
        started: float,
    ) -> CounterfactualResult:
        sensitivity = state.sensitivity if state.sensitivity is not None else np.full(self.partition.num_superpixels, np.nan)
        seed_probabilities = state.seed_probabilities if state.seed_probabilities is not None else np.full(self.partition.num_superpixels, np.nan)
        return CounterfactualResult(
            activation=np.asarray(activation, dtype=bool).copy(),
            original_probability=float(state.original_probability or 0.0),
            counterfactual_probability=float(probability),
            threshold=float(state.threshold or 0.0),
            valid=bool(valid),
            status=status,
            objective_cost=objective_cost,
            solver_time_sec=state.solver_time_sec,
            total_time_sec=perf_counter() - started,
            oracle_calls=state.oracle_calls,
            num_hard_clauses=len(hard_clauses),
            num_soft_clauses=self.number_of_variables,
            sensitivity=np.asarray(sensitivity, dtype=np.float64).copy(),
            seed_probabilities=np.asarray(seed_probabilities, dtype=np.float64).copy(),
            weight_mode=self.weight_mode,
            occlusion_method=self.occlusion_method,
            cegar_cut_mode=self.cegar_cut_mode,
            search_lane=search_lane,
            solver_superpixel_indices=self.solver_global_indices.copy(),
            frozen_outside_lung_indices=self.frozen_outside_lung_indices.copy(),
            base_weights=self.base_weights.copy(),
            guidance_penalties=(
                np.asarray(state.guidance_penalties, dtype=np.int64).copy()
                if state.guidance_penalties is not None
                else np.zeros(self.partition.num_superpixels, dtype=np.int64)
            ),
            seed_guidance_weight=self.seed_guidance_weight,
            anti_sensitivity_penalty=self.anti_sensitivity_penalty,
            low_sensitivity_penalty=self.low_sensitivity_penalty,
            sensitivity_normalization=self.sensitivity_normalization,
            minimum_lung_overlap=self.minimum_lung_overlap,
            hybrid_exact_budget_fraction=self.hybrid_exact_budget_fraction,
            exact_block_count=state.exact_block_count,
            expansion_cut_count=state.expansion_cut_count,
            oracle_cache_hits=state.oracle_cache_hits,
            oracle_cache_misses=state.oracle_cache_misses,
        )

    def _search_lane(
        self,
        state: SearchState,
        lane: Literal["exact", "expansion"],
        initial_hard_clauses: Sequence[Sequence[int]],
        repetition_penalties: np.ndarray,
        minimum_new_from: Sequence[np.ndarray],
        minimum_new_occluded_regions: int,
        oracle_call_limit: int | None = None,
    ) -> CounterfactualResult:
        """Run one CEGAR lane with complete-assignment cache and bounded RC2 calls."""
        started = perf_counter()
        self._run_seed_exploration(state)
        all_on = self._all_on_global_activation()
        if state.original_probability is None or state.threshold is None:
            raise RuntimeError("seed state was not initialized")
        if state.original_probability < state.threshold:
            return self._result(
                state, all_on, state.original_probability, False, "original_negative", 0, [], self.base_weights,
                lane, started,
            )
        sensitivity = state.sensitivity if state.sensitivity is not None else np.full(self.partition.num_superpixels, np.nan)
        guidance = self._guidance_penalties(sensitivity)
        state.guidance_penalties = guidance
        global_weights = self._lexicographic_weights(guidance, repetition_penalties)
        seed_probabilities = (
            state.seed_probabilities
            if state.seed_probabilities is not None
            else np.full(self.partition.num_superpixels, np.nan)
        )
        hard_clauses = [tuple(int(literal) for literal in clause) for clause in initial_hard_clauses]
        hard_clauses.extend(clause for clause in sorted(state.exact_block_clauses) if clause not in hard_clauses)
        if lane == "expansion":
            hard_clauses.extend(
                clause for clause in sorted(state.expansion_cut_clauses) if clause not in hard_clauses
            )
        hard_set = set(hard_clauses)
        top_id = self.number_of_variables
        for previous_global in minimum_new_from:
            previous_local = self._to_solver_activation(previous_global)
            available_new = np.flatnonzero(previous_local)
            if minimum_new_occluded_regions == 0:
                continue
            if available_new.size < minimum_new_occluded_regions:
                return self._result(
                    state,
                    self._all_on_global_activation(),
                    state.original_probability,
                    False,
                    "diversity_novelty_unsatisfiable",
                    None,
                    hard_clauses,
                    self.base_weights,
                    lane,
                    started,
                )
            if minimum_new_occluded_regions == 1:
                novelty_clauses = [tuple(-int(index + 1) for index in available_new)]
            else:
                try:
                    from pysat.card import CardEnc
                except ImportError as error:
                    raise RuntimeError("minimum-new MDSOL constraints require python-sat") from error
                encoding = CardEnc.atleast(
                    lits=[-int(index + 1) for index in available_new],
                    bound=minimum_new_occluded_regions,
                    top_id=top_id,
                    encoding=1,
                )
                top_id = encoding.nv
                novelty_clauses = [tuple(int(literal) for literal in clause) for clause in encoding.clauses]
            for novelty_clause in novelty_clauses:
                if novelty_clause not in hard_set:
                    hard_clauses.append(novelty_clause)
                    hard_set.add(novelty_clause)

        def satisfies_constraints(local_activation: np.ndarray) -> bool:
            return all(
                any(
                    (literal > 0 and bool(local_activation[literal - 1]))
                    or (literal < 0 and not bool(local_activation[-literal - 1]))
                    for literal in clause
                )
                for clause in hard_clauses
            )

        seed_valid_indices = self.solver_global_indices[
            np.isfinite(seed_probabilities[self.solver_global_indices])
            & (seed_probabilities[self.solver_global_indices] < state.threshold)
        ]
        has_auxiliary_variables = any(abs(literal) > self.number_of_variables for clause in hard_clauses for literal in clause)
        if seed_valid_indices.size and not has_auxiliary_variables:
            valid_seed_indices = [
                int(index)
                for index in seed_valid_indices
                if satisfies_constraints(
                    np.asarray(
                        [solver_index != self.global_to_solver_index[index] for solver_index in range(self.number_of_variables)],
                        dtype=bool,
                    )
                )
            ]
            if valid_seed_indices:
                chosen = int(min(valid_seed_indices, key=lambda index: int(global_weights[index])))
                state.oracle_cache_hits += 1
                seed_activation = self._all_on_global_activation()
                seed_activation[chosen] = False
                return self._result(
                    state,
                    seed_activation,
                    float(seed_probabilities[chosen]),
                    True,
                    "valid_seed",
                    int(global_weights[chosen]),
                    hard_clauses,
                    global_weights,
                    lane,
                    started,
                )
        last_activation, last_probability, last_cost = all_on, state.original_probability, 0
        call_limit = self.max_oracle_calls if oracle_call_limit is None else min(oracle_call_limit, self.max_oracle_calls)
        while state.oracle_calls < call_limit:
            local, objective = self._solve_rc2(hard_clauses, global_weights, state)
            if local is None:
                return self._result(
                    state, last_activation, last_probability, False, "unsatisfiable", last_cost,
                    hard_clauses, global_weights, lane, started,
                )
            activation = self._to_global_activation(local)
            try:
                probability = self._query_oracle(activation, state)
            except RuntimeError as error:
                if str(error) != "oracle budget exhausted":
                    raise
                break
            state.explored_assignments += 1
            last_activation, last_probability, last_cost = activation, probability, objective
            if probability < state.threshold:
                return self._result(
                    state, activation, probability, True, "valid", objective,
                    hard_clauses, global_weights, lane, started,
                )
            exact = self._exact_block_clause(local)
            if exact not in hard_set:
                hard_clauses.append(exact)
                hard_set.add(exact)
                state.exact_block_count += 1
                state.exact_block_clauses.add(exact)
            if lane == "expansion":
                expansion = self._expansion_cut_clause(local)
                if expansion is not None and expansion not in hard_set:
                    hard_clauses.append(expansion)
                    hard_set.add(expansion)
                    state.expansion_cut_count += 1
                    state.expansion_cut_clauses.add(expansion)
        return self._result(
            state, last_activation, last_probability, False, "oracle_budget_exhausted", last_cost,
            hard_clauses, global_weights, lane, started,
        )

    def explain(
        self,
        initial_hard_clauses: Sequence[Sequence[int]] | None = None,
        extra_soft_penalties: np.ndarray | None = None,
        state: SearchState | None = None,
        minimum_new_from: Sequence[np.ndarray] | None = None,
        minimum_new_occluded_regions: int = 0,
    ) -> CounterfactualResult:
        """Find one valid CF, defaulting to exact assignment blocking.

        ``hybrid`` runs an exact lane first and, only if it fails within the shared
        budget, continues an expansion lane. No seed failure creates an expansion cut.
        """
        shared_state = state or SearchState()
        penalties = (
            np.zeros(self.partition.num_superpixels, dtype=np.int64)
            if extra_soft_penalties is None
            else np.asarray(extra_soft_penalties, dtype=np.int64)
        )
        if penalties.shape != (self.partition.num_superpixels,) or np.any(penalties < 0):
            raise ValueError("extra_soft_penalties must be a nonnegative global vector")
        if minimum_new_occluded_regions < 0:
            raise ValueError("minimum_new_occluded_regions cannot be negative")
        clauses = initial_hard_clauses or []
        novelty = minimum_new_from or []
        if self.cegar_cut_mode != "hybrid":
            result = self._search_lane(
                shared_state, self.cegar_cut_mode, clauses, penalties, novelty, minimum_new_occluded_regions
            )
            self.last_result = result
            return result
        seed_calls = self.number_of_variables + 1 + min(
            self.exploration_random_samples,
            max(0, self.max_oracle_calls - self.number_of_variables - 1),
        )
        exact_limit = max(
            seed_calls,
            int(np.ceil(self.max_oracle_calls * self.hybrid_exact_budget_fraction)),
        )
        exact = self._search_lane(
            shared_state, "exact", clauses, penalties, novelty, minimum_new_occluded_regions, exact_limit
        )
        if exact.valid or shared_state.oracle_calls >= self.max_oracle_calls:
            self.last_result = exact
            return exact
        expansion = self._search_lane(
            shared_state, "expansion", clauses, penalties, novelty, minimum_new_occluded_regions, self.max_oracle_calls
        )
        if not expansion.valid:
            result = exact if exact.counterfactual_probability <= expansion.counterfactual_probability else expansion
            self.last_result = result
            return result
        if not exact.valid:
            self.last_result = expansion
            return expansion
        result = min(
            (exact, expansion),
            key=lambda item: (item.objective_cost or np.iinfo(np.int64).max, item.counterfactual_probability),
        )
        self.last_result = result
        return result

    def diverse_solutions(
        self,
        maximum_solutions: int = 3,
        repetition_penalty: int | None = None,
        minimum_new_occluded_regions: int = 1,
    ) -> list[CounterfactualResult]:
        """Extract MDSOL-style diverse CFs under one shared seed/cache/budget state."""
        if maximum_solutions < 1:
            raise ValueError("maximum_solutions must be positive")
        if minimum_new_occluded_regions < 0:
            raise ValueError("minimum_new_occluded_regions cannot be negative")
        state = SearchState()
        blocked: list[tuple[int, ...]] = []
        repetition = np.zeros(self.partition.num_superpixels, dtype=np.int64)
        results: list[CounterfactualResult] = []
        penalty = int(repetition_penalty or max(1, int(self.base_weights.max())))
        for _ in range(maximum_solutions):
            novelty_sources = results if minimum_new_occluded_regions else []
            result = self.explain(
                blocked,
                repetition,
                state,
                [item.activation for item in novelty_sources],
                minimum_new_occluded_regions,
            )
            if not result.valid:
                break
            results.append(result)
            local = self._to_solver_activation(result.activation)
            blocked.append(self._exact_block_clause(local))
            repetition[~result.activation] += penalty
        return results