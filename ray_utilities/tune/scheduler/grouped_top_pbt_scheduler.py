"""Group-based Population Based Training scheduler for Ray Tune.

This module provides :class:`GroupedTopPBTTrialScheduler`, which extends
:class:`TopPBTTrialScheduler` to apply PBT quantile logic at the group level
rather than individual trials. Trials with equivalent configurations (differing
only by seed) are grouped together, and their performance is averaged for
quantile calculations.

Key Features:
    - Groups trials by config (excluding seed differences)
    - Averages scores within groups for quantile determination
    - 1:1 group matching between lower and upper quantiles
    - Individual trial pairing within matched groups
    - Skip list updates include all values from upper quantile groups
    - Assigns a stable, human-readable group_key to each trial's config based on
      the group's shared hyperparameters, excluded from config hashing and restored
      after exploitations
    - Tracks perturbation epoch for statistical grouping

The scheduler assumes group membership is stable (configs don't change between
groups during training), which is typical when using grid search mutations.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple, cast

from frozendict import frozendict
from ray.rllib.utils import flatten_dict

from ray_utilities.constants import FORK_FROM
from ray_utilities.misc import deep_freeze
from ray_utilities.nice_logger import ImportantLogger
from ray_utilities.tune.experiments import CONFIG_HASH_EXCLUDE_KEYS
from ray_utilities.tune.scheduler.top_pbt_scheduler import PERTURBATION_EPOCH, TopPBTTrialScheduler, _trial_id

if TYPE_CHECKING:
    from ray.tune.experiment import Trial
    from ray.tune.execution.tune_controller import TuneController


logger = logging.getLogger(__name__)

GROUP_KEY = "pbt_group_key"
"""Special config key for group identity. Must not participate in config hashing."""


def thawdict(x: Any) -> Any:
    if isinstance(x, frozendict):
        return {k: thawdict(v) for k, v in x.items()}
    return x


class GroupedTopPBTTrialScheduler(TopPBTTrialScheduler):
    """Group-based Population Based Training scheduler with averaged group scoring.

    This scheduler extends TopPBTTrialScheduler to group trials with equivalent
    configurations (differing only by seed) and apply PBT quantile logic at the
    group level. Performance metrics are averaged within each group, and exploitation
    decisions are made on a group-to-group basis with 1:1 matching.

    Within matched groups, individual trials are paired for exploitation, with each
    lower trial exploiting a different trial from the assigned upper group. This
    ensures balanced exploitation across all trials while maintaining group coherence.

    Additionally, a stable, human-readable ``group_key`` is added to each trial's
    config, derived from the group's shared hyperparameters (excluding seeds and
    other non-group-defining keys). This key is excluded from config hashing and
    is restored after exploitations so that a trial keeps its original group identity.

    The scheduler also tracks a ``pbt_epoch`` for each trial, indicating which
    perturbation round it's in. This enables statistical grouping and analysis:
    ``df.groupby(['pbt_group_key', 'pbt_epoch'])``

    Args:
        time_attr: Attribute to use for time progression tracking.
        metric: Metric name to optimize (e.g., "episode_reward_mean").
        mode: Optimization mode, either "max" or "min".
        perturbation_interval: Number of time units between perturbations.
        burn_in_period: Time units before perturbations begin.
        hyperparam_mutations: Dictionary mapping hyperparameter names to mutation
            specifications. Supports grid_search definitions for deterministic sampling.
        quantile_fraction: Fraction of population (by group count) to consider as
            "top performers".
        resample_probability: Probability of resampling parameters during perturbation.
        perturbation_factors: Tuple of (lower, upper) factors for parameter perturbation.
        custom_explore_fn: Optional custom function for exploration logic.
        log_config: Whether to log configuration changes.
        require_attrs: Whether to require time_attr and metric in results.
        synch: Whether to use synchronous perturbation.
        reseed: When trials are perturbed, update env_seed to avoid repetition.
        num_samples: Multiplier for identical configs (including same seed).
        group_size: Number of trials with same config differing only in seed.
            Defaults to 3. The actual expected group size is group_size * num_samples.
        prune_late_trials: Whether to prune slow, poorly-performing trials.
        recompute_groups: If True, recompute group membership at each perturbation.
            Defaults to False, assuming stable group membership.

    Example:
        >>> scheduler = GroupedTopPBTTrialScheduler(
        ...     metric="episode_reward_mean",
        ...     mode="max",
        ...     perturbation_interval=50000,
        ...     hyperparam_mutations={
        ...         "lr": {"grid_search": [1e-4, 5e-4, 1e-3]},
        ...         "batch_size": {"grid_search": [64, 128, 256]},
        ...     },
        ...     quantile_fraction=0.2,  # Keep top 20% of groups
        ...     num_samples=3,  # Expect 3 trials per config
        ... )

        # Later, for analysis:
        >>> df.groupby(["pbt_group_key", "pbt_epoch"]).agg({"reward": "mean"})

    Note:
        - Group membership is determined by config similarity (excluding seed and
          fork metadata).
        - Quantile fraction applies to group count, not trial count.
        - Group sizes may vary; warnings are logged for significant deviations from
          num_samples.
        - CyclicMutation skip lists include all values from all trials in upper groups.
        - Each trial carries a stable ``group_key`` in its config that identifies the
          original group. This key is excluded from config hashing and restored after
          exploitations.
        - Trials ended during perturbation can be distinguished from newly exploited ones
          by their ``pbt_epoch`` value.

    See Also:
        :class:`TopPBTTrialScheduler`: Base scheduler with trial-level PBT
        :func:`_group_trials_by_config`: Config-based grouping logic
    """

    def __init__(
        self,
        *,
        time_attr: str = "current_step",
        metric: str | None = None,
        mode: str | None = None,
        perturbation_interval: float = 100_000,
        burn_in_period: float = 0,
        hyperparam_mutations=None,
        quantile_fraction: float = 0.1,
        resample_probability: float = 1.0,
        perturbation_factors: Tuple[float, float] = (0.8, 1.2),
        custom_explore_fn=None,
        log_config: bool = True,
        require_attrs: bool = True,
        synch: bool = False,
        reseed: bool = True,
        num_samples: int = 1,
        group_size: int = 3,
        prune_late_trials: bool = False,
        recompute_groups: bool = False,
    ):
        # Register GROUP_KEY BEFORE calling super().__init__() so parent processes it correctly
        if GROUP_KEY not in TopPBTTrialScheduler.additional_config_keys:
            TopPBTTrialScheduler.additional_config_keys.append(GROUP_KEY)
        if GROUP_KEY not in CONFIG_HASH_EXCLUDE_KEYS:
            CONFIG_HASH_EXCLUDE_KEYS.append(GROUP_KEY)

        super().__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            burn_in_period=burn_in_period,
            hyperparam_mutations=hyperparam_mutations,
            quantile_fraction=quantile_fraction,
            resample_probability=resample_probability,
            perturbation_factors=perturbation_factors,
            custom_explore_fn=custom_explore_fn,
            log_config=log_config,
            require_attrs=require_attrs,
            synch=synch,
            reseed=reseed,
            num_samples=num_samples,
            prune_late_trials=prune_late_trials,
        )

        self._trial_to_group: Dict[Trial, frozendict[str, Any]] = {}
        """Maps each trial to its frozen config group key."""

        self._config_groups: Dict[frozendict, List[Trial]] = {}
        """Cache of config groups (frozen config → list of trials)."""

        self._group_assignments: Dict[frozendict[str, Any], frozendict[str, Any]] = {}
        """Maps lower quantile groups to upper quantile groups (for exploitation)."""

        self._recompute_groups = recompute_groups
        """Whether to recompute group membership at each perturbation."""

        self.__group_size = group_size  # not sure if we need it

        self._expected_group_size = group_size * num_samples
        """Expected number of trials per group (group_size * num_samples)."""

        logger.info(
            "Initialized GroupedTopPBTTrialScheduler with num_samples=%d, group_size=%d, "
            "expected_group_size=%d, recompute_groups=%s",
            num_samples,
            group_size,
            self._expected_group_size,
            recompute_groups,
        )

    def _build_group_key_from_config(self, config: Mapping[str, Any]) -> str:
        """Build a deterministic group key from hyperparameter mutation keys.

        Args:
            config: Normalized group config (seeds and tracking keys removed).

        Returns:
            Stable string identifying the group based on mutation-defined hyperparameters.
        """
        flat_mutations = flatten_dict(self._hyperparam_mutations)
        flat_config = flatten_dict(config)  # pyright: ignore[reportArgumentType]
        chosen_values = {}
        for key in flat_mutations:
            if key not in flat_config:
                # if it is some iterable structure was was extended possibly: fcnet_hiddens=[8] -> fcnet_hiddens/0 : 8
                if key in config:
                    chosen_values[key] = config[key]
                else:
                    ImportantLogger.important_warning(
                        logger,
                        "'%s' is a hyperparam_mutation but the key is not present in the flattened config. "
                        "It likely was not added by the scheduler/searcher "
                        "and could not be filled in by this class beforehand.",
                        key,
                    )
                continue
            chosen_values[key] = flat_config[key]
        parts = (f"{key}={value}" for key, value in sorted(chosen_values.items()))
        return "|".join(parts)

    def _group_trials_by_config(self, trials: List[Trial] | None = None) -> Dict[frozendict[str, Any], List[Trial]]:
        """Group trials by their configuration, excluding seed and fork metadata.

        Trials are grouped based on frozen config dictionaries after excluding:
        - FORK_FROM metadata
        - env_seed (seed differences don't constitute different configs)
        - Keys in CONFIG_HASH_EXCLUDE_KEYS
        - Scheduler tracking keys (self.additional_config_keys), including group_key

        Respects ``self._recompute_groups``:
        - If False (default), previously assigned group membership is preserved even
          if trial configs change due to PBT exploitation; the trial's config[group_key]
          is restored accordingly.
        - If True, group membership (and group_key) is recomputed based on the current config.

        Args:
            trials: List of trials to group. If None, uses all trials from _trial_state.

        Returns:
            Dictionary mapping frozen config keys to lists of trials with that config.
            Also updates self._trial_to_group for efficient reverse lookups, preserving
            stable mapping when ``_recompute_groups`` is False.
        """
        if trials is None:
            trials = list(self._trial_state.keys())

        config_to_trials: Dict[frozendict, List[Trial]] = {}

        for trial in trials:
            use_existing_mapping = (not self._recompute_groups) and (trial in self._trial_to_group)

            if use_existing_mapping:
                frozen_config = self._trial_to_group[trial]
                trial.config[GROUP_KEY] = self._build_group_key_from_config(frozen_config)
            else:
                config = trial.config.copy()

                config.pop(FORK_FROM, None)
                config.pop("env_seed", None)
                if "cli_args" in config:
                    config["cli_args"] = config["cli_args"].copy()
                    config["cli_args"].pop("env_seed", None)
                for key in {*CONFIG_HASH_EXCLUDE_KEYS, "seed", "env_seed"}:
                    config.pop(key, None)
                for key in self.additional_config_keys:
                    config.pop(key, None)

                frozen_config = deep_freeze(config)
                self._trial_to_group[trial] = frozen_config
                trial.config[GROUP_KEY] = self._build_group_key_from_config(config)

            config_to_trials.setdefault(frozen_config, []).append(trial)

        if config_to_trials:
            group_sizes = [len(trials) for trials in config_to_trials.values()]
            avg_size = sum(group_sizes) / len(group_sizes)
            logger.debug(
                "Grouped %d trials into %d groups. Group sizes: min=%d, max=%d, avg=%.1f, expected=%d",
                len(trials),
                len(config_to_trials),
                min(group_sizes),
                max(group_sizes),
                avg_size,
                self._expected_group_size,
            )

            # Warn if group sizes deviate significantly from expected_group_size
            if self._expected_group_size > 1:
                deviations = [abs(size - self._expected_group_size) for size in group_sizes]
                max_deviation = max(deviations)
                if max_deviation > max(1, self._expected_group_size * 0.2):
                    logger.warning(
                        "Group sizes deviate significantly from expected_group_size=%d. "
                        "Max deviation: %d. This may indicate inconsistent seeding or trial failures.",
                        self._expected_group_size,
                        max_deviation,
                    )

        return config_to_trials

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in lower and upper quantiles based on group-averaged scores.

        Groups trials by config, computes average score per group, applies quantile
        fraction to group count (not trial count), then flattens back to trial lists.

        Groups with no valid scores are excluded from quantile calculations.

        Returns:
            Tuple of (lower_quantile_trials, upper_quantile_trials).
            Lower quantile trials will exploit upper quantile trials.
        """
        # Group trials by config
        config_to_trials = self._group_trials_by_config()

        if not config_to_trials:
            logger.warning("No trial groups found for quantile calculation.")
            return [], []

        # Compute average score per group
        group_scores: List[Tuple[float, frozendict, List[Trial]]] = []

        for frozen_config, group_trials in config_to_trials.items():
            # Collect valid scores from trials in this group
            valid_scores = []
            for trial in group_trials:
                state = self._trial_state.get(trial)
                if (
                    state
                    and state.last_score is not None
                    and not math.isnan(state.last_score)
                    and not trial.is_finished()
                ):
                    valid_scores.append(state.last_score)

            # Skip groups with no valid scores
            if not valid_scores:
                logger.debug("Group %s has no valid scores, skipping from quantiles.", frozen_config)
                continue

            # Compute average score for this group
            avg_score = sum(valid_scores) / len(valid_scores)
            group_scores.append((avg_score, frozen_config, group_trials))

            logger.debug(
                "Group %s: %d/%d trials with valid scores, avg_score=%.2f",
                frozen_config,
                len(valid_scores),
                len(group_trials),
                avg_score,
            )

        if not group_scores:
            logger.warning("No groups with valid scores found for quantile calculation.")
            return [], []

        # Sort groups by average score (already mode-adjusted in _trial_state)
        group_scores.sort(key=lambda x: x[0])

        # Apply quantile fraction to group count
        num_groups = len(group_scores)
        num_top_groups = max(1, math.ceil(num_groups * self._quantile_fraction))

        # Ensure we don't exceed half the population
        if num_top_groups > num_groups / 2:
            num_top_groups = math.floor(num_groups / 2)

        if num_top_groups < 1:
            logger.warning("Quantile calculation resulted in 0 top groups. Using 1 group.")
            num_top_groups = 1

        # Split into top and bottom groups
        top_groups = group_scores[-num_top_groups:]
        bottom_groups = group_scores[:-num_top_groups] if num_top_groups < num_groups else []

        logger.info(
            "Quantile calculation: %d total groups, %d top groups (%.1f%%), %d bottom groups",
            num_groups,
            num_top_groups,
            100 * num_top_groups / num_groups,
            len(bottom_groups),
        )

        # Flatten to trial lists for compatibility with base class
        upper_quantile_trials = [trial for _, _, trials in top_groups for trial in trials]
        lower_quantile_trials = [trial for _, _, trials in bottom_groups for trial in trials]

        logger.debug(
            "Quantiles: %d upper trials from %d groups, %d lower trials from %d groups",
            len(upper_quantile_trials),
            len(top_groups),
            len(lower_quantile_trials),
            len(bottom_groups),
        )

        return lower_quantile_trials, upper_quantile_trials

    def _distribute_exploitation(self, lower_quantile: List[Trial], upper_quantile: List[Trial]) -> Dict[Trial, Trial]:
        """Distribute exploitation with 1:1 group matching and trial chaining.

        Groups lower and upper quantile trials separately, matches groups in 1:1
        fashion using cyclic iteration, then chains/zips individual trials within
        matched groups to create trial-level exploitation assignments.

        Args:
            lower_quantile: List of trials that will exploit top trials.
            upper_quantile: List of top-performing trials to be exploited.

        Returns:
            Dictionary mapping each lower trial to the upper trial it should exploit.
        """
        if not upper_quantile or not lower_quantile:
            return {}

        # Group both quantiles by config
        lower_groups = self._group_trials_by_config(lower_quantile)
        upper_groups = self._group_trials_by_config(upper_quantile)

        if not lower_groups or not upper_groups:
            logger.warning("Empty group sets in exploitation distribution. Skipping exploitation.")
            return {}

        # Convert to lists for indexing
        lower_group_list = list(lower_groups.items())
        upper_group_list = list(upper_groups.items())

        logger.debug(
            "Distributing exploitation: %d lower groups, %d upper groups", len(lower_group_list), len(upper_group_list)
        )

        # Match groups 1:1 with rotating offset for balance
        assignments: Dict[Trial, Trial] = {}
        self._group_assignments = {}

        for idx, (lower_config, lower_trials) in enumerate(lower_group_list):
            # Use rotating offset to balance upper group usage
            upper_idx = idx % len(upper_group_list)
            upper_config, upper_trials = upper_group_list[upper_idx]

            # Store group-level assignment
            self._group_assignments[lower_config] = upper_config

            # Chain/zip individual trials within matched groups
            # Chain/zip trials - handle size mismatches by cycling
            if len(lower_trials) <= len(upper_trials):
                # Lower group is smaller or equal - direct pairing
                assignments.update(dict(zip(lower_trials, upper_trials)))
            else:
                # Lower group is larger - cycle through upper trials
                upper_cycle = itertools.cycle(upper_trials)
                assignments.update({lower_trial: next(upper_cycle) for lower_trial in lower_trials})

            logger.debug(
                "Matched group %d: %d lower trials → %d upper trials (group idx=%d)",
                idx,
                len(lower_trials),
                len(upper_trials),
                upper_idx,
            )

        # Log distribution statistics
        upper_trial_usage = {}
        for upper_trial in assignments.values():
            upper_trial_usage[upper_trial.trial_id] = upper_trial_usage.get(upper_trial.trial_id, 0) + 1

        logger.info(
            "Exploitation distribution complete: %d trial pairs, %d unique upper trials used",
            len(assignments),
            len(upper_trial_usage),
        )

        if upper_trial_usage:
            usage_counts = list(upper_trial_usage.values())
            logger.debug(
                "Upper trial usage: min=%d, max=%d, avg=%.1f",
                min(usage_counts),
                max(usage_counts),
                sum(usage_counts) / len(usage_counts),
            )

        return assignments

    def reset_exploitations(self):
        """Reset exploitation assignments and optionally group cache.

        Clears both trial-level assignments (from parent) and group-level assignments.
        If recompute_groups is True, also clears the config group cache to force
        recomputation at the next perturbation.

        Additionally, ensures that each trial's config carries its stable group_key.
        This is important to restore the original key immediately after exploitation,
        avoiding copying the donor group's key.
        """
        super().reset_exploitations()
        self._group_assignments = {}

        # Restore group_key for ALL known trials, not just those in _trial_state
        for trial, frozen in self._trial_to_group.items():
            trial.config[GROUP_KEY] = self._build_group_key_from_config(frozen)
            logger.debug("Restored group_key for trial %s: %s", trial.trial_id, trial.config[GROUP_KEY])

        if self._recompute_groups:
            self._config_groups.clear()
            logger.debug("Cleared group cache for recomputation.")

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing.

        Returns:
            Dictionary containing parent state plus group-specific data.
            Trial objects are converted to trial IDs for serialization.
        """
        state = super().get_state()

        state.update(
            {
                "trial_to_group": {_trial_id(trial): group for trial, group in self._trial_to_group.items()},
                "recompute_groups": self._recompute_groups,
                # Don't serialize _config_groups or _group_assignments as they're recomputed
            }
        )

        return state

    def set_state(self, state: dict) -> None:
        """Restore scheduler state from checkpoint.

        Args:
            state: State dictionary from get_state().

        Note:
            Trial objects cannot be restored from IDs alone. The restored state
            will use trial IDs as keys until trials are added and mappings are rebuilt.
        """
        super().set_state(state)

        # Restore group-specific state
        # Note: Keys are trial IDs (strings) until trials are added via on_trial_add
        self._trial_to_group = cast("Dict[Trial, frozendict]", state.get("trial_to_group", {}))
        self._recompute_groups = state.get("recompute_groups", False)

        # Clear computed caches - will be rebuilt
        self._config_groups = {}
        self._group_assignments = {}

        logger.info(
            "Restored GroupedTopPBTTrialScheduler state: %d trial→group mappings",
            len(self._trial_to_group),
        )

    def on_trial_add(self, tune_controller, trial, **kwargs):
        """Handle trial addition by updating group mappings after parent processing.

        Ensures that a stable ``group_key`` is set on the trial's config based on
        its mapped group when available.

        Args:
            tune_controller: The TuneController managing trials.
            trial: The trial being added.
            **kwargs: Additional arguments.
        """
        if "seed" not in trial.config:
            ImportantLogger.important_warning(
                logger, "Trial %s added without 'seed' in config. Grouping may be incorrect.", trial.trial_id
            )
        if "seed" in trial.config and "env_seed" not in trial.config:
            trial.config["env_seed"] = trial.config["seed"]

        super().on_trial_add(tune_controller, trial, **kwargs)

        # After unpickling, convert trial ID keys to trial object keys
        if self._unpickled and hasattr(self, "_trial_to_group"):
            trial_id = trial.trial_id
            if trial_id in self._trial_to_group:
                # remove string id replace with real trial object
                frozen_config = self._trial_to_group.pop(trial_id)
                self._trial_to_group[trial] = frozen_config
                trial.config[GROUP_KEY] = self._build_group_key_from_config(frozen_config)
                logger.debug("Restored group mapping for trial %s", trial_id)

        # Ensure group_key is set for new trials
        # Perform immediate grouping to establish group_key
        if trial not in self._trial_to_group:
            config = trial.config.copy()
            config.pop(FORK_FROM, None)
            config.pop("env_seed", None)
            if "cli_args" in config:
                config["cli_args"] = config["cli_args"].copy()
                config["cli_args"].pop("env_seed", None)
            for key in {*CONFIG_HASH_EXCLUDE_KEYS, "seed", "env_seed"}:
                config.pop(key, None)
            for key in self.additional_config_keys:
                config.pop(key, None)

            frozen_config = deep_freeze(config)
            self._trial_to_group[trial] = frozen_config
            trial.config[GROUP_KEY] = self._build_group_key_from_config(frozen_config)
            logger.debug("Assigned initial group_key for trial %s: %s", trial.trial_id, trial.config[GROUP_KEY])

    def _checkpoint_or_exploit(
        self,
        trial: Trial,
        tune_controller: "TuneController",
        upper_quantile: List[Trial],
        lower_quantile: List[Trial],
    ):
        """Checkpoint if in upper quantile, exploits if in lower.

        Overrides parent to ensure GROUP_KEY and PERTURBATION_EPOCH are restored after the parent removes them.
        """
        # Call parent implementation which handles all PBT logic including epoch assignment
        super()._checkpoint_or_exploit(trial, tune_controller, upper_quantile, lower_quantile)

        # Restore GROUP_KEY after parent processing (parent removes all additional_config_keys)
        if trial in self._trial_to_group:
            frozen_config = self._trial_to_group[trial]
            trial.config[GROUP_KEY] = self._build_group_key_from_config(frozen_config)
            logger.debug(
                "Restored group_key for trial %s after _checkpoint_or_exploit: %s (epoch=%s)",
                trial.trial_id,
                trial.config[GROUP_KEY],
                trial.config.get(PERTURBATION_EPOCH, "unknown"),
            )
        else:
            logger.warning("Trial %s not found in _trial_to_group during _checkpoint_or_exploit", trial.trial_id)
