import importlib
import sys
import optuna
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
import hashlib
import gc
import torch
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import FuncFormatter

import numbers
import numpy as np
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype, is_timedelta64_dtype, is_bool_dtype
from pandas.io.formats.style import Styler

from typing import Callable, Dict, Optional, Sequence, Any, Literal, Iterable

import warnings
import logging
_LOGGER = logging.getLogger(__name__)
from dataclasses import dataclass, replace, asdict
from IPython.display import display


from ..pipeline.lstm import (
    LSTMRunConfig, LSTMPipeline,
    ModelConfig, DataConfig, FeatureConfig, TrainConfig, NormalizeConfig
)

# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
## GENERAL OPTUNA UTILITIES
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# -----------------------------
# Registering suggesters
# -----------------------------

@dataclass(frozen=True)
class ParamInt:
    name: str
    low: int
    high: int
    step: Optional[int] = None
    log: bool = False
    condition: Optional[str] = None  # e.g., "model.head == 'mlp'"

@dataclass(frozen=True)
class ParamFloat:
    name: str
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False
    condition: Optional[str] = None

@dataclass(frozen=True)
class ParamCat:
    name: str
    choices: Sequence[Any]
    condition: Optional[str] = None

ParamSpec = ParamInt | ParamFloat | ParamCat

@dataclass(frozen=True)
class SuggesterDoc:
    """Optional description block attached to a suggester."""
    summary: str
    params: Sequence[ParamSpec]
    notes: Optional[Sequence[str]] = None  # invariants, constraints, etc.

# ---- Registry ----

SuggesterFn = Callable[[optuna.trial.Trial, "LSTMRunConfig"], "LSTMRunConfig"]

class _Entry:
    __slots__ = ("fn", "doc")
    def __init__(self, fn: SuggesterFn, doc: Optional[SuggesterDoc]):
        self.fn = fn
        self.doc = doc

# preserve across importlib.reload
REGISTRY: Dict[str, _Entry] = globals().get("REGISTRY", {})

def register_suggester(name: str, *, doc: Optional[SuggesterDoc] = None):
    """Decorator registering a suggester under a stable name, with optional docs."""
    def _decorator(fn: SuggesterFn) -> SuggesterFn:
        if name in REGISTRY and REGISTRY[name].fn is not fn:
            _LOGGER.warning("A suggester with name '%s' was already registered. Re-writing.", name)
        REGISTRY[name] = _Entry(fn, doc)
        return fn  # no wrapping
    return _decorator

def get_registered_entry(name: str) -> _Entry:
    """
    Return the registered entry for a suggester.

    On a cache miss, this lazily imports the suggester module to populate the
    registry, then tries again.
    """
    entry = REGISTRY.get(name)
    if entry is not None:
        return entry

    importlib.invalidate_caches()
    mod = sys.modules.get("heat_forecast.suggesters")
    if mod is None:
        importlib.import_module("heat_forecast.suggesters")
    else:
        importlib.reload(mod)

    entry = REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown suggester '{name}'. Known: {sorted(REGISTRY)}")
    return entry


def get_suggester(name: str) -> SuggesterFn:
    entry = get_registered_entry(name)
    return entry.fn

def describe_suggester(name: str, *, format: str = "markdown") -> str | Dict[str, Any]:
    """Return a human-readable description of the suggester's parameter space."""
    entry = get_registered_entry(name)
    if entry.doc is None:
        return f"(no description registered for '{name}')"

    doc = entry.doc
    if format == "markdown":
        lines: list[str] = []
        lines.append(f"### {name}")
        lines.append("")
        lines.append(doc.summary.strip())
        lines.append("")
        lines.append("**Parameters:**")
        for p in doc.params:
            if isinstance(p, ParamInt):
                rng = f"[{p.low}, {p.high}]"
                step = f", step={p.step}" if p.step is not None else ""
                log  = ", log" if p.log else ""
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (int) {rng}{step}{log}{cond}")
            elif isinstance(p, ParamFloat):
                rng = f"[{p.low}, {p.high}]"
                step = f", step={p.step}" if p.step is not None else ""
                log  = ", log" if p.log else ""
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (float) {rng}{step}{log}{cond}")
            else:  # ParamCat
                choices = ", ".join(repr(c) for c in p.choices)
                cond = f" _(if {p.condition})_" if p.condition else ""
                lines.append(f"- `{p.name}` (categorical): {{{choices}}}{cond}")
        if doc.notes:
            lines.append("")
            lines.append("**Notes / constraints:**")
            for n in doc.notes:
                lines.append(f"- {n}")
        return "\n".join(lines)

    elif format == "dict":
        # simple machine-readable dump
        def _one(p: ParamSpec) -> dict:
            base = {"name": p.name, "condition": getattr(p, "condition", None)}
            if isinstance(p, ParamInt):
                base |= {"type": "int", "low": p.low, "high": p.high, "step": p.step, "log": p.log}
            elif isinstance(p, ParamFloat):
                base |= {"type": "float", "low": p.low, "high": p.high, "step": p.step, "log": p.log}
            else:
                base |= {"type": "categorical", "choices": list(p.choices)}
            return base
        return {
            "name": name,
            "summary": doc.summary,
            "params": [_one(p) for p in doc.params],
            "notes": list(doc.notes) if doc.notes else [],
        }
    else:
        raise ValueError("format must be 'markdown' or 'dict'")
    
# ---------------------------------
# To set a random seed at each trial
# ---------------------------------

def trial_based_seed(base_seed: int | None, trial: optuna.trial.Trial) -> int:
    # Seeding strategy: combine base seed (if any) with study name and trial number
    base = 0 if base_seed is None else int(base_seed)
    key = f"{trial.study.study_name}:{trial.number}:{base}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return h % (2**31 - 1)

def param_based_seed(base_seed: int | None, trial: optuna.trial.Trial) -> int:
    # Seeding strategy: combine base seed (if any) with trial params (useful for grid search with different repeat_ids)
    base = 0 if base_seed is None else int(base_seed)
    items = "|".join(f"{k}={trial.params[k]!r}" for k in sorted(trial.params))
    h = int(hashlib.sha256(f"{items}:{base}".encode()).hexdigest()[:16], 16)
    return h % (2**31 - 1)

# --------------------------------------------------------
# Optuna objective function
# --------------------------------------------------------

def cleanup_after_trial():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _duration_fmt(td: pd.Timedelta) -> str:
    if pd.isna(td): return ""
    total = int(td.total_seconds())
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02}:{m:02}:{s:02}"

def make_objective(
    target_id_df: pd.DataFrame,
    aux_id_df: pd.DataFrame | None,
    base_config: LSTMRunConfig,
    *,
    suggest_config: Callable[[optuna.trial.Trial, LSTMRunConfig], LSTMRunConfig],
    start_train: pd.Timestamp | None,
    end_train: pd.Timestamp | None,
    start_val: pd.Timestamp | None,
    end_val: pd.Timestamp | None,
    logger: logging.Logger | None = None,
):
    """
    Returns an Optuna objective(trial) that:
      1) builds a per-trial config,
      2) prepares loaders,
      3) trains with early stopping + pruning,
      4) returns the best validation loss on original scale.
    """
    logger = logger or _LOGGER
    def objective(trial: optuna.trial.Trial) -> float:
        cfg = suggest_config(trial, base_config)

        # derive a deterministic seed; decide the strategy based on the sampler (tpe -> per trial, grid -> param based)
        is_grid = isinstance(trial.study.sampler, optuna.samplers.GridSampler)
        if is_grid:
            seed = param_based_seed(base_config.seed, trial)  # stable per (params, repeat_idx)
        else:
            seed = trial_based_seed(base_config.seed, trial)  # distinct per trial in TPE
        trial.set_user_attr("device", "cuda" if torch.cuda.is_available() else "cpu")
        trial.set_user_attr("config_seed", int(seed))
        cfg = replace(cfg, seed=seed)

        pipe = None
        train_loader = None
        val_loader = None
        try:
            pipe = LSTMPipeline(
                target_df=target_id_df,
                aux_df=aux_id_df,
                config=cfg,
                logger=logger,
            )

            # loaders (computes global stats if needed)
            train_loader, val_loader = pipe.make_loaders(
                start_train=start_train, end_train=end_train,
                start_val=start_val,     end_val=end_val,
                gap_hours=cfg.data.gap_hours,
            )

            # train with pruning support (pipeline forwards `trial` to trainer)
            out = pipe.fit(train_loader, val_loader=val_loader, trial=trial)
            trial.set_user_attr("n_params", pipe.n_params) # save per-trial attribute
            trainer = pipe._trainer

            best_val = out.get('best_val_loss_orig')
            trial.set_user_attr("best_val", float(best_val) if best_val is not None else None)

            best_epoch = out.get('best_epoch')
            trial.set_user_attr("best_epoch", int(best_epoch) if best_epoch is not None else None)

            dur = out.get('duration_until_best')
            trial.set_user_attr("dur_until_best", _duration_fmt(dur) if dur is not None else None)
            trial.set_user_attr("dur_until_best_s", float(dur.total_seconds()) if dur is not None else None)

            avg_near_best = out.get('avg_near_best')
            trial.set_user_attr("avg_near_best", float(avg_near_best) if avg_near_best is not None else None)
            
            trial.set_user_attr("last_val", float(trainer.val_losses_orig[-1]) if trainer.val_losses_orig else None)
            
            return float(best_val) if best_val is not None else float("inf")
        
        except RuntimeError as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            if "out of memory" in str(e).lower():
                cleanup_after_trial()
                if isinstance(trial.study.sampler, optuna.samplers.GridSampler):
                    trial.set_user_attr("oom", True)
                    return float("inf")  # keep grid complete
                else:
                    raise optuna.TrialPruned()  # skip in TPE
            raise
        finally:
            # Drop large refs to help GC
            del pipe, train_loader, val_loader
            cleanup_after_trial()
    return objective

# --------------------------------------------------------
# Utilities to configure and run a Optuna study
# --------------------------------------------------------

def best_params_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    if study.best_trial.number == trial.number:
        n_params = trial.user_attrs.get("n_params")
        if n_params is not None:
            study.set_user_attr("best_n_params", int(n_params))

@dataclass
class OptunaStudyConfig:
    """
    Configuration for running an Optuna study. Parameters are grouped into three main categories:

    General
    -------
    study_name: str
        Name of the Optuna study (used for storage + logging).
    n_trials: Optional[int] = 50
        Number of trials to run. Ignored for grid search (all grid points run).
    timeout: Optional[float] = None
        Maximum optimization time in seconds. None = unlimited.
    seed: Optional[int] = None
        Global random seed for reproducibility (affects sampler, not training).
    storage: Optional[str] = None
        Storage backend (e.g., SQLite URL or RDB for distributed optimization).

    Pruner
    ------
    pruner: {"percentile", "median", "nop"} = "percentile"
        Strategy for trial pruning:
        - "percentile": prune if trial is worse than Xth percentile.
        - "median": prune if trial is worse than running median.
        - "nop": disable pruning.
    pruner_percentile: float = 60.0
        For percentile pruner: cutoff percentile (lower = more aggressive).
    n_warmup_steps: int = 7
        Minimum number of steps (epochs) before pruning is considered.
    n_startup_trials_pruner: Optional[int] = None
        Number of initial trials to complete before pruning is enabled.
        Defaults to ~20% of n_trials (min 15).
    interval_steps: int = 1
        Frequency (in steps/epochs) at which pruning checks are made.
    patience: Optional[int] = 3
        Wraps the pruner with a "PatientPruner":
        waits for `patience` failed checks before actually pruning.
        None disables patience wrapper.

    Sampler 
    ------
    sampler: {"tpe", "grid"} = "tpe"
        Which sampler to use:
        - "tpe": Tree-structured Parzen Estimator (Bayesian optimizer).
        - "grid": exhaustive grid search (evaluates all param combos).
    n_startup_trials_sampler: Optional[int] = None
        Number of random trials before TPE starts modeling the search space.
        Defaults to ~20% of n_trials (min 10).
    n_ei_candidates: int = 128
        Number of candidate samples evaluated at each TPE step.
        Higher = better search accuracy but more compute overhead.
    multivariate: bool = True
        If True, TPE models joint distributions (captures param interactions).
    constant_liar: bool = False
        For distributed/parallel runs:
        - If True, mark running trials with "fake" losses to avoid duplicate suggestions.
        - For sequential runs, leave False (default).

    Internals (autogenerated)
    ---------
    grid: dict | None
        Parameter grid (only set when using GridSampler).
    grid_size: int | None
        Number of parameter combinations in the grid.
    """
    study_name: str
    n_trials: Optional[int] = 50
    timeout: Optional[float] = None
    seed: Optional[int] = None
    storage: Optional[str] = None

    pruner: Literal["percentile", "median", "nop"] = "percentile"
    pruner_percentile: float = 60.0
    n_warmup_steps: int = 7
    n_startup_trials_pruner: Optional[int] = None
    interval_steps: int = 1
    patience: Optional[int] = 3  # None disables PatientPruner

    sampler: Literal["tpe", "grid"] = "tpe"  
    n_startup_trials_sampler: Optional[int] = None  # for TPE
    n_ei_candidates: int = 128                      # for TPE
    multivariate: bool = True                       # for TPE
    constant_liar: bool = False                     # for TPE, True only if parallel
    consider_endpoints: bool = True                 # for TPE

    def __post_init__(self):
        if self.pruner not in ["percentile", "median", "nop"]:
            raise ValueError("Invalid pruner: choose 'percentile', 'median' or 'nop'.")
        if self.n_warmup_steps < 0 or self.interval_steps < 1:
            raise ValueError("n_warmup_steps >= 0 and interval_steps >= 1 required.")
        if self.sampler not in ["tpe", "grid"]:
            raise ValueError("Invalid sampler: choose 'tpe' or 'grid'.")

        # Grid: ignore n_trials and force NopPruner
        if self.sampler == "grid":
            if self.n_trials is not None:
                _LOGGER.warning("GridSampler ignores n_trials; evaluating full grid.")
                self.n_trials = None
            if self.pruner != "nop":
                _LOGGER.warning("Forcing pruner='nop' for grid search.")
                self.pruner = "nop"
        
        # Defaults for TPE
        if self.sampler == "tpe":
            # 20% of budget, min 10
            if self.n_startup_trials_sampler is None:
                self.n_startup_trials_sampler = 10 if self.n_trials is None else max(10, self.n_trials // 5)
            # Pruner startup: 20% of budget, min 15
            if self.pruner != "nop" and self.n_startup_trials_pruner is None:
                self.n_startup_trials_pruner = 15 if self.n_trials is None else max(15, self.n_trials // 5)

        # internals
        self.grid = None       # for grid search, set by make_sampler
        self.grid_size = None  # for grid search, set by make_sampler

    # --- Sampler ---
    def make_sampler(self, *, suggester_name: Optional[str]) -> optuna.samplers.BaseSampler:
        if self.sampler == "tpe":
            return optuna.samplers.TPESampler(
                seed=self.seed,
                multivariate=self.multivariate,
                n_startup_trials=self.n_startup_trials_sampler,
                n_ei_candidates=self.n_ei_candidates,
                constant_liar=self.constant_liar,
                consider_endpoints=self.consider_endpoints,
            )
    
        if self.sampler == "grid":
            # infer grid from suggester documentation
            if suggester_name is None:
                raise ValueError("suggester_name must be provided when using grid sampler.")
            entry = get_registered_entry(suggester_name)
            if entry.doc is None:
                raise ValueError(f"Suggester '{suggester_name}' has no registered doc; can't infer grid.")
            # Make sure every param is categorical without conditions
            for p in entry.doc.params:
                if not isinstance(p, ParamCat):
                    raise ValueError(f"Suggester '{suggester_name}' has non-categorical param '{p.name}'; can't do grid search.")
                if p.condition is not None:
                    raise ValueError(f"Suggester '{suggester_name}' has conditional param '{p.name}'; can't do grid search.")
            # Infer grid from categorical params
            grid = {p.name: p.choices for p in entry.doc.params}
            self.grid = grid
            self.grid_size = math.prod(len(v) for v in grid.values())
            return optuna.samplers.GridSampler(grid)

    def make_pruner(self) -> optuna.pruners.BasePruner:
        if self.pruner == "nop":
            return optuna.pruners.NopPruner()
        
        # If using a pruner but suggester is grid, warn
        if self.sampler == "grid":
            _LOGGER.warning("Using a pruner with grid search is unusual; changing to pruner='nop'.")
            self.pruner = "nop"
            return optuna.pruners.NopPruner()

        if self.pruner == "percentile":
            base = optuna.pruners.PercentilePruner(
                percentile=self.pruner_percentile,
                n_startup_trials=self.n_startup_trials_pruner,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )
        elif self.pruner == "median":
            base = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials_pruner,
                n_warmup_steps=self.n_warmup_steps,
                interval_steps=self.interval_steps,
            )

        return optuna.pruners.PatientPruner(base, patience=self.patience) if self.patience is not None else base
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d
    
    def to_structured_dict(self) -> dict:
        d = self.to_dict()
        structured = {
            "General": {
                "study_name": d.pop("study_name"),
                "n_trials": d.pop("n_trials"),
                "timeout": d.pop("timeout"),
                "seed": d.pop("seed"),
                "storage": d.pop("storage"),
            },
            "Sampler": {
                "sampler": d.pop("sampler"),
                "n_startup_trials_sampler": d.pop("n_startup_trials_sampler"),
                "n_ei_candidates": d.pop("n_ei_candidates"),
                "multivariate": d.pop("multivariate"),
                "constant_liar": d.pop("constant_liar"),
                "consider_endpoints": d.pop("consider_endpoints"),
            },
            "Pruner": {
                "pruner": d.pop("pruner"),
                "pruner_percentile": d.pop("pruner_percentile"),
                "n_warmup_steps": d.pop("n_warmup_steps"),
                "n_startup_trials_pruner": d.pop("n_startup_trials_pruner"),
                "interval_steps": d.pop("interval_steps"),
                "patience": d.pop("patience"),
            },
        }
        return structured

def run_study(
        unique_id: str,
        heat_df: pd.DataFrame, 
        aux_df: pd.DataFrame | None, 
        base_cfg: LSTMRunConfig,
        *,
        start_train: pd.Timestamp | None,
        end_train: pd.Timestamp | None,
        start_val: pd.Timestamp | None,
        end_val: pd.Timestamp | None,
        optuna_cfg: OptunaStudyConfig,
        suggest_config_name: str,
    ) -> optuna.Study:
    suggest_config = get_suggester(suggest_config_name)
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)
    pruner = optuna_cfg.make_pruner()

    heat_id_df = heat_df[heat_df['unique_id'] == unique_id]
    aux_id_df = aux_df[aux_df['unique_id'] == unique_id] if aux_df is not None else None

    objective_fn = make_objective(
        heat_id_df, 
        aux_id_df, 
        base_cfg,
        suggest_config=suggest_config,
        start_train=start_train, end_train=end_train, 
        start_val=start_val, end_val=end_val,
        logger=logging.getLogger("optuna_run"),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        load_if_exists=bool(optuna_cfg.storage),
    )
    study.set_user_attr("splits", {
        "start_train": str(start_train), "end_train": str(end_train),
        "start_val": str(start_val), "end_val": str(end_val)
    })
    study.set_user_attr("unique_id", unique_id)
    study.set_user_attr("base_cfg", base_cfg.to_dict())
    study.set_user_attr("optuna_cfg", optuna_cfg.to_dict())
    study.set_user_attr("suggest_config_name", suggest_config_name)
    study.set_user_attr("env", {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.version.cuda else None,
        "cudnn": torch.backends.cudnn.version()
    })

    study.optimize(
        objective_fn,
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[best_params_callback]
    )
    return study

def continue_study(
        study_name: str,
        storage_url: str,
        *,
        n_new_trials: int,
        target_df: pd.DataFrame,
        aux_df: pd.DataFrame | None,
        suggest_config_name: str | None = None,
    ) -> optuna.Study:
    # retrieve and update pruner
    tmp = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    ) # Note: this loads the study with the default pruner and sampler; 
      # we will need to reload the study with the correct ones

    # resolve suggester
    if suggest_config_name is None:
        suggest_config_name = tmp.user_attrs.get("suggest_config_name")
        if not suggest_config_name:
            raise KeyError("study missing 'suggest_config_name' user_attr")
    suggest_config = get_suggester(suggest_config_name)

    # rebuild study with correct pruner and sampler
    optuna_cfg_dict = tmp.user_attrs.get("optuna_cfg")
    if optuna_cfg_dict is None:
        raise KeyError("study missing 'optuna_cfg' user_attr; can't rebuild pruner/sampler")
    optuna_cfg = OptunaStudyConfig(**optuna_cfg_dict)
    pruner = optuna_cfg.make_pruner()
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        pruner=pruner,
        sampler=sampler
    )
    _LOGGER.info("Continuing study '%s' with %d new trials", study_name, n_new_trials)

    sd = optuna_cfg.to_structured_dict()
    lines = ", ".join(f"{k}={v}" for k, v in sd["General"].items())
    _LOGGER.info("Using general config: %s.", lines)

    lines = ", ".join(f"{k}={v}" for k, v in sd["Pruner"].items())
    _LOGGER.info("Using pruner config: %s.", lines)

    lines = ", ".join(f"{k}={v}" for k, v in sd["Sampler"].items())
    _LOGGER.info("Using sampler config: %s.", lines)

    # retrieve base config
    base_cfg_dict = study.user_attrs["base_cfg"]
    base_cfg = LSTMRunConfig(
        model=ModelConfig(**base_cfg_dict["model"]),
        data=DataConfig(**base_cfg_dict["data"]),
        features=FeatureConfig(**base_cfg_dict["features"]),
        train=TrainConfig(**base_cfg_dict["train"]),
        norm=NormalizeConfig(**base_cfg_dict["norm"]),
        seed=base_cfg_dict["seed"]
    )

    # rebuild train sets
    unique_id = study.user_attrs["unique_id"]
    heat_id_df = target_df[target_df['unique_id'] == unique_id]
    aux_id_df = None
    if aux_df is not None:
        aux_id_df = aux_df[aux_df['unique_id'] == unique_id]

    # retrieve splits
    to_ts = lambda s: None if s in (None, "None") else pd.Timestamp(s)
    start_train = to_ts(study.user_attrs["splits"]["start_train"])
    end_train   = to_ts(study.user_attrs["splits"]["end_train"])
    start_val   = to_ts(study.user_attrs["splits"]["start_val"])
    end_val     = to_ts(study.user_attrs["splits"]["end_val"])

    objective_fn = make_objective(
        heat_id_df, 
        aux_id_df, 
        base_cfg,
        suggest_config=suggest_config,
        start_train=start_train, end_train=end_train, 
        start_val=start_val, end_val=end_val,
        logger=logging.getLogger("optuna_run"),
    )

    # continue an existing study with new trials
    study.optimize(
        objective_fn, 
        n_trials=n_new_trials, 
        gc_after_trial=True, 
        show_progress_bar=True, 
        callbacks=[best_params_callback]
    )
    return study

def copy_study_with_first_n_trials(
        *,
        src_study_name: str,
        dst_study_name: str,
        storage_url: str,
        n_trials: int
    ) -> optuna.Study:
    # Load the source study
    src = optuna.load_study(study_name=src_study_name, storage=storage_url)
    optuna_cfg_dict = src.user_attrs.get("optuna_cfg")
    suggest_config_name = src.user_attrs.get("suggest_config_name")
    if not suggest_config_name:
        raise KeyError("study missing 'suggest_config_name' user_attr")
    if optuna_cfg_dict is None:
        raise KeyError("study missing 'optuna_cfg' user_attr; can't rebuild pruner/sampler")
    optuna_cfg = OptunaStudyConfig(**optuna_cfg_dict)
    pruner = optuna_cfg.make_pruner()
    sampler = optuna_cfg.make_sampler(suggester_name=suggest_config_name)

    # Create a destination study with the same objective directions
    dst = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=dst_study_name,
        storage=optuna_cfg.storage,
    )

    # Keep the first n_trials by trial number (0..99); skip any RUNNING trials
    allowed = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    first_n = [
        t for t in src.get_trials(deepcopy=True)
        if t.number < n_trials and t.state in allowed
    ]
    dst.add_trials(first_n)
    _LOGGER.info("Copied the first %d trials (and all user_attrs) from '%s' to '%s'", len(first_n), src_study_name, dst_study_name)

    # Copy study-level user attrs
    for k, v in src.user_attrs.items():
        dst.set_user_attr(k, v)
        if k == "optuna_cfg":
            sd = optuna_cfg.to_structured_dict()
            lines = ", ".join(f"{k}={v}" for k, v in sd["General"].items())
            _LOGGER.info("Copied general config: %s.", lines)

            lines = ", ".join(f"{k}={v}" for k, v in sd["Pruner"].items())
            _LOGGER.info("Copied pruner config: %s.", lines)

            lines = ", ".join(f"{k}={v}" for k, v in sd["Sampler"].items())
            _LOGGER.info("Copied sampler config: %s.", lines)

    # After verifying everything looks right, you can remove the old study:
    # optuna.delete_study(study_name=SRC_NAME, storage=STORAGE)

    return dst

import optuna
from typing import Optional

def rename_study(
    *,
    storage_url: str,
    old_name: str,
    new_name: str,
    keep_old: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> optuna.Study:
    """
    Safely 'renames' an Optuna study by cloning it to a new study name and
    (optionally) deleting the old one.

    Parameters
    ----------
    storage_url : str
        RDB storage URL (sqlite:///..., postgresql://..., mysql://..., etc.).
    old_name : str
        Existing study name.
    new_name : str
        Desired new study name (must not exist already).
    keep_old : bool, default False
        If True, do NOT delete the old study after cloning.
    dry_run : bool, default False
        If True, perform all checks and report what would happen, but do not
        create/delete anything.
    logger : logging.Logger | None
        Optional logger for status messages.

    Returns
    -------
    optuna.Study
        The *new* study object (loaded from storage). In dry-run mode, this
        just returns the *old* study object.

    Notes
    -----
    - Refuses to proceed if the old study has RUNNING trials.
    - Copies all trials (COMPLETE/PRUNED/FAIL) and study-level user attrs.
    - System attrs are copied when possible.
    - Directions are preserved (multi-objective supported).
    - If your code depends on custom pruner/sampler, they're not persisted in
      the storage itself; you'll set them when you call `optimize()` again,
      just like in your `continue_study()` helper.
    """
    log = logger or _LOGGER

    # --- Load the source study and sanity-checks
    src = optuna.load_study(study_name=old_name, storage=storage_url)

    # Check RUNNING trials (cloning those is undefined / unsafe)
    running = [t for t in src.get_trials(deepcopy=False)
               if t.state == optuna.trial.TrialState.RUNNING]
    if running:
        log.warning("Found %d RUNNING trial(s) in study '%s'; skipping.", len(running), old_name)

    # New name must not already exist
    try:
        _ = optuna.load_study(study_name=new_name, storage=storage_url)
    except Exception:
        pass  # likely doesn't exist
    else:
        raise ValueError(f"A study named '{new_name}' already exists in this storage.")

    # Gather everything we need from src
    directions = getattr(src, "directions", None)
    if not directions:
        # Older optuna exposes .direction (single objective)
        directions = [src.direction]

    # Trials to copy (skip RUNNING by construction)
    allowed_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.WAITING,  # usually none left, but safe to include
    }
    trials_to_copy = [
        t for t in src.get_trials(deepcopy=True)
        if t.state in allowed_states
    ]

    if dry_run:
        log.info("[DRY RUN] Would create study '%s' and copy %d trial(s) and %d user_attr(s).",
                 new_name, len(trials_to_copy), len(src.user_attrs))
        if not keep_old:
            log.info("[DRY RUN] Would delete old study '%s'.", old_name)
        return src

    # --- Create destination study with the same objective directions
    if len(directions) == 1:
        dst = optuna.create_study(
            study_name=new_name,
            storage=storage_url,
            direction=directions[0].name.lower(),  # "minimize"/"maximize"
        )
    else:
        dst = optuna.create_study(
            study_name=new_name,
            storage=storage_url,
            directions=[d.name.lower() for d in directions],
        )

    # --- Copy trials
    # Note: add_trials() preserves numbers/params/values/states/timings.
    if trials_to_copy:
        dst.add_trials(trials_to_copy)

    # --- Copy study-level attributes
    for k, v in src.user_attrs.items():
        dst.set_user_attr(k, v)

    # --- Quick verification
    src_trials = [t for t in src.get_trials(deepcopy=False) if t.state in allowed_states]
    dst_trials = dst.get_trials(deepcopy=False)
    if len(src_trials) != len(dst_trials):
        raise RuntimeError(
            f"Clone verification failed: expected {len(src_trials)} trials, "
            f"found {len(dst_trials)} in destination."
        )

    # --- Optionally delete the old study
    if not keep_old:
        optuna.delete_study(study_name=old_name, storage=storage_url)
        log.info("Deleted old study '%s'.", old_name)

    log.info("Renamed study '%s' -> '%s' (copied %d trials).",
             old_name, new_name, len(dst_trials))
    # Return a loaded handle to the *new* study
    return optuna.load_study(study_name=new_name, storage=storage_url)

def create_substudy_by_param(
    *,
    storage_url: str,
    src_study_name: str,
    dst_study_name: str,
    param_name: str,
    equals: Optional[object] = None,
    in_values: Optional[Iterable[object]] = None,
    predicate: Optional[Callable[[object], bool]] = None,
    numeric_tol: Optional[float] = None,
    include_states: Optional[set[optuna.trial.TrialState]] = None,
    copy_user_attrs: bool = True,
) -> optuna.Study:
    """
    Create a new Optuna study containing only the trials from `src_study_name`
    that match a condition on parameter `param_name`.

    Matching options (first provided wins):
      - equals: exact match (with numeric tolerance if numeric_tol given)
      - in_values: membership in a set/list/tuple
      - predicate: custom callable(value) -> bool
    """
    # Load source study
    src = optuna.load_study(study_name=src_study_name, storage=storage_url)

    # Ensure destination doesn't exist
    try:
        optuna.load_study(study_name=dst_study_name, storage=storage_url)
    except Exception:
        pass
    else:
        raise ValueError(f"Study '{dst_study_name}' already exists in this storage.")

    # Decide directions (single or multi)
    try:
        directions = list(getattr(src, "directions"))
    except Exception:
        directions = [src.direction]

    # Create destination
    if len(directions) == 1:
        dst = optuna.create_study(
            study_name=dst_study_name,
            storage=storage_url,
            direction=directions[0].name.lower(),  # "minimize"/"maximize"
        )
    else:
        dst = optuna.create_study(
            study_name=dst_study_name,
            storage=storage_url,
            directions=[d.name.lower() for d in directions],
        )

    # Build matcher
    def _eq(a, b) -> bool:
        if numeric_tol is not None and isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return abs(a - b) <= numeric_tol
        return a == b

    if equals is not None:
        match = lambda v: _eq(v, equals)
        active_criteria = f"equals {equals!r}"
    elif in_values is not None:
        vals = set(in_values)
        match = lambda v: any(_eq(v, x) for x in vals)
        active_criteria = f"in {vals!r}"
    elif predicate is not None:
        match = predicate
        active_criteria = "selected by predicate"
    else:
        raise ValueError("Provide one of: equals=, in_values=, or predicate= for filtering.")

    # States to include (exclude RUNNING by default)
    if include_states is None:
        include_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.WAITING,
        }

    # Select and copy trials
    allowed = []
    for t in src.get_trials(deepcopy=True):
        if t.state not in include_states:
            continue
        if param_name in t.params and match(t.params[param_name]):
            allowed.append(t)

    if allowed:
        dst.add_trials(allowed)

    # Copy study-level user attrs + provenance
    if copy_user_attrs:
        for k, v in src.user_attrs.items():
            dst.set_user_attr(k, v)
    dst.set_user_attr("_parent_study", src_study_name)
    dst.set_user_attr("_substudy_filter", {
        "param_name": param_name,
        "equals": equals,
        "in_values": list(in_values) if in_values is not None else None,
        "numeric_tol": numeric_tol,
        "states": [s.name for s in include_states],
    })

    txt = f"Constraint: {param_name} {active_criteria}"
    _LOGGER.info(
        "Created substudy '%s' from '%s' with %d matching trial(s). %s",
        dst_study_name, src_study_name, len(allowed), txt
    )

    if not allowed:
        _LOGGER.warning("No trials matched %s in '%s'. Substudy created empty.", txt, src_study_name)

    return dst


# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
## FOR STUDY INSPECTION
# ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯

# ---------------------------
# Basics
# ---------------------------

def trials_df(
        study: optuna.study.Study,
        states: tuple[str] = ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"),
    ) -> pd.DataFrame:
    """Collect trial rows with params_ columns and the objective value."""
    df = study.trials_dataframe()
    df = df[df["state"].isin(states)].copy()
    return df


def _param_kinds_from_study(study: optuna.study.Study) -> dict[str, str]:
    """
    Map param name -> 'numeric' | 'categorical' from Optuna distributions.
    Falls back to dtype later where distributions are missing.
    """
    kinds: dict[str, str] = {}
    for t in study.trials:
        for name, dist in t.distributions.items():
            if isinstance(dist, (FloatDistribution, IntDistribution)):
                kinds[name] = "numeric"
            elif isinstance(dist, CategoricalDistribution):
                kinds[name] = "categorical"
    return kinds

def _guess_param_kind(series: pd.Series) -> str:
    # treat integer/float categoricals as numeric
    return "numeric" if is_numeric_dtype(series) and (not is_bool_dtype(series)) and (not series.nunique() <= 5) else "categorical"


def study_minimize(study: optuna.study.Study) -> bool:
    """True if the first objective is MINIMIZE; sensible default True."""
    try:
        if hasattr(study, "direction"):
            return study.direction == optuna.study.StudyDirection.MINIMIZE
        if hasattr(study, "directions"):
            return study.directions[0] == optuna.study.StudyDirection.MINIMIZE
    except Exception:
        pass
    return True


def _param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("params_")]

def _param_meta(study: optuna.study.Study) -> dict[str, dict]:
    """param name -> metadata extracted from Optuna distributions (bounds, log, step, choices)."""
    out: dict[str, dict] = {}
    # scan trials to collect distributions (some trials may not have all)
    for t in study.trials:
        for name, dist in t.distributions.items():
            if name in out:
                continue
            if isinstance(dist, IntDistribution) or isinstance(dist, FloatDistribution):
                out[name] = dict(kind="numeric", low=float(dist.low), high=float(dist.high),
                                 log=getattr(dist, "log", False), step=getattr(dist, "step", None))
            elif isinstance(dist, CategoricalDistribution):
                choices = list(dist.choices)
                out[name] = dict(kind="categorical", choices=choices, n_choices=len(choices))
    return out

def _float_fmt(x):
    """Scientific if |x|<1e-2 or very large; fixed otherwise."""
    if pd.isna(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    return f"{v:.2e}" if (abs(v) < 1e-2 or abs(v) >= 1e6) else f"{v:.3f}"

def _int_fmt(x):
    if pd.isna(x): return ""
    return f"{int(np.round(float(x))):,d}"

def _bool_fmt(x):
    if pd.isna(x):
        return ""
    return "True" if bool(x) else "False"

def _pct_fmt(x):
    if pd.isna(x): return ""
    return f"{100*float(x):.1f}%"

def _is_integerish_series(s: pd.Series, tol: float = 1e-12) -> bool:
    if is_integer_dtype(s):
        return True
    if is_float_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr)]
        return arr.size > 0 and np.all(np.abs(arr - np.round(arr)) <= tol)
    return False

def _best_num_formatter_for_series(s):
    """Return appropriate formatter (_int_fmt, _float_fmt or _bool_fmt) for Series or 1D array."""
    if isinstance(s, np.ndarray):
        s = pd.Series(s)
    elif isinstance(s, pd.Index):
        s = s.to_series().reset_index(drop=True)
    elif not isinstance(s, pd.Series):
        raise TypeError(f"Expected pandas Series or numpy ndarray, got {type(s)}")

    if is_bool_dtype(s):
        return _bool_fmt
    elif _is_integerish_series(s):
        return _int_fmt
    else:
        return _float_fmt

def _param_col(p: str) -> str:
    if "user_attrs_" in p or "system_attrs_" in p:
        return p
    # Treat as parameter name
    if p.startswith("params_"):
        return p
    return f"params_{p}"

def _get_param(df: pd.DataFrame, param: str) -> pd.Series:
    if param in df.columns:
        return df[param]
    col = _param_col(param)
    if col in df.columns:
        return df[col]
    raise KeyError(f"Parameter '{col}' not found in DataFrame.")

# ---------------------------
# Parameter tables (high-level)
# ---------------------------

def trials_df_for_display(
    df: pd.DataFrame,               
    to_exclude: tuple[str, ...] | None = None,
    value_col: str = "value",
    ascending: bool = True
) -> pd.DataFrame:
    """Display a trials DataFrame with params_ columns and the objective value."""
    # create a working copy
    df = df.copy()

    # exclude uninformative columns
    if to_exclude is None:
        to_exclude = (
            "user_attrs_repeat_idx",
            "system_attrs_grid_id",
            "system_attrs_search_space",
            "datetime_start",
            "datetime_complete",
        )
    df = df.drop(columns=[c for c in to_exclude if c in df.columns], errors="ignore")

    # nicer state
    if "state" in df.columns:
        df["state"] = df["state"].astype(str)

    # order by value
    df = df.sort_values(by="value", ascending=ascending).reset_index(drop=True)

    # format duration
    if "duration" in df.columns and is_timedelta64_dtype(df["duration"]):
        df["duration"] = df["duration"].apply(_duration_fmt)

    # sort by original 'value', then rename to value_col
    if "value" in df.columns:
        df.sort_values(by="value", ascending=ascending, inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
        if value_col != "value":
            df.rename(columns={"value": value_col}, inplace=True)
    else:
        pass

    # format numeric params
    param_cols = _param_cols(df)
    for c in param_cols:
        if is_numeric_dtype(df[c]):
            _fmt = _best_num_formatter_for_series(df[c])
            df[c] = df[c].map(_fmt)

    # format objective
    if value_col in df.columns and is_numeric_dtype(df[value_col]):
        fmt_val = _best_num_formatter_for_series(df[value_col])
        df[value_col] = df[value_col].map(fmt_val)

    # format user_attrs columns
    user_attrs_cols = [c for c in df.columns if c.startswith("user_attrs_")]
    for c in user_attrs_cols:
        if is_numeric_dtype(df[c]):
            _fmt = _best_num_formatter_for_series(df[c])
            df[c] = df[c].map(_fmt)

    # strip 'params_' prefix for display
    if param_cols:
        df.rename(columns={c: c.replace("params_", "", 1) for c in param_cols}, inplace=True)

    return df

def _style_coverage_tables(
    num_df: pd.DataFrame | None,
    cat_df: pd.DataFrame | None,
    *,
    cmap_good: str = "Greens",
    cmap_bad: str = "Reds",
) -> tuple[Styler | None, Styler | None]:
    """Return styled versions of numeric + categorical coverage tables."""

    # --- Numeric coverage
    if num_df is None or num_df.empty:
        num_sty = None
    else:
        num_fmt = {}

        # integer-ish count columns
        for col in [
            "n_with_value", "n_complete", "n_pruned",
            "n_fail", "n_running", "unique", "non_empty_bins"
        ]:
            if col in num_df:
                num_fmt[col] = _int_fmt

        # percentage-like columns
        for col in ["unique_ratio", "span_ratio", "bin_coverage"]:
            if col in num_df:
                num_fmt[col] = _pct_fmt

        # continuous metrics
        for col in [
            "min", "p25", "median", "p75", "max",
            "search_low", "search_high", "search_step"
        ]:
            if col in num_df:
                num_fmt[col] = _float_fmt

        num_sty = (
            num_df.style
            .format(num_fmt, na_rep="")
            .hide(axis="index")
            .set_properties(subset=["parameter"],
                            **{"font-weight": "600", "white-space": "nowrap"})
        )

        # heatmaps / bars
        if "span_ratio" in num_df:
            num_sty = num_sty.background_gradient(
                subset=["span_ratio"], cmap=cmap_good, vmin=0, vmax=1
            )
        for col in ["unique_ratio", "bin_coverage"]:
            if col in num_df:
                num_sty = num_sty.bar(subset=[col], vmin=0, vmax=1, color=cmap_good)

    # --- Categorical coverage
    if cat_df is None or cat_df.empty:
        cat_sty = None
    else:
        cat_fmt = {}

        # integer-ish counts
        for col in [
            "n_with_value", "n_complete", "n_pruned",
            "n_fail", "n_running", "unique_levels", "choices_declared"
        ]:
            if col in cat_df:
                cat_fmt[col] = _int_fmt

        # ratios / continuous
        if "coverage_ratio" in cat_df:
            cat_fmt["coverage_ratio"] = _pct_fmt
        if "imbalance_ratio" in cat_df:
            cat_fmt["imbalance_ratio"] = _float_fmt
        if "entropy_ratio" in cat_df:
            cat_fmt["entropy_ratio"] = _pct_fmt

        cat_sty = (
            cat_df.style
            .format(cat_fmt, na_rep="")
            .hide(axis="index")
            .set_properties(subset=["parameter"],
                            **{"font-weight": "600", "white-space": "nowrap"})
        )

        # visual cues
        if "coverage_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["coverage_ratio"], cmap=cmap_good, vmin=0.0, vmax=1.0)
        if "imbalance_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["imbalance_ratio"], cmap=cmap_bad, vmin=1.0)
        if "entropy_ratio" in cat_df:
            cat_sty = cat_sty.background_gradient(subset=["entropy_ratio"], cmap=cmap_good, vmax=1.0, vmin=0.0)

    return num_sty, cat_sty

def summarize_params_coverage(
    study: optuna.study.Study,
    df: pd.DataFrame,
    value_col: str = "value",
    max_levels: int = 8,
    bins_for_gap: int = 10,
) -> tuple[Styler | None, Styler | None]:
    """
    Return (num_df, cat_df) coverage diagnostics.

    Coverage counts use ALL trials; performance stats (median/IQR/mean/std) use COMPLETE trials only.
    Adds per-state counts, missing rate, and comparison vs the declared search space when available.
    """
    kinds = _param_kinds_from_study(study)
    meta = _param_meta(study)
    param_cols = _param_cols(df)

    # state masks
    st = df["state"].astype(str)
    m_complete = (st == "COMPLETE") & df[value_col].notna()
    m_pruned   = (st == "PRUNED")
    m_fail     = (st == "FAIL")
    m_running  = (st == "RUNNING") if "RUNNING" in st.unique() else pd.Series(False, index=df.index)

    num_rows, cat_rows = [], []

    for col in sorted(param_cols):
        pname = col.replace("params_", "", 1)
        # coverage series: use all trials (any state)
        s_all = df[col]
        has_val = s_all.notna()
        n_any = int(has_val.sum())
        if n_any == 0:
            # no value ever sampled for this param
            continue

        # counts by state where param present
        n_complete_param = int((m_complete & has_val).sum())
        n_pruned_param = int((m_pruned & has_val).sum())
        n_fail_param = int((m_fail & has_val).sum())
        n_running_param = int((m_running & has_val).sum())

        # determine kind (prefer distribution info)
        kind = _guess_param_kind(df[col]) # or meta.get(pname, {}).get("kind") or kinds.get(pname)

        if kind == "numeric":
            vals_all = pd.to_numeric(s_all[has_val], errors="coerce").astype(float).dropna()
            if vals_all.empty:
                continue

            # bounds from distribution if available
            low = meta.get(pname, {}).get("low")
            high = meta.get(pname, {}).get("high")
            log_flag = bool(meta.get(pname, {}).get("log", False))
            step = meta.get(pname, {}).get("step")

            vmin = float(vals_all.min())
            v25  = float(vals_all.quantile(0.25))
            v50  = float(vals_all.median())
            v75  = float(vals_all.quantile(0.75))
            vmax = float(vals_all.max())
            nunique = int(vals_all.nunique())
            unique_ratio = nunique / n_any

            # span ratio vs declared space
            span_ratio = np.nan
            if low is not None and high is not None and high > low:
                span_ratio = (vmax - vmin) / (high - low)

            # gap/coverage via quantile bins: fraction of non-empty bins
            non_empty_bins = np.nan
            bin_coverage = np.nan
            try:
                q = min(bins_for_gap, max(1, vals_all.nunique()))
                cats = pd.qcut(vals_all, q=q, duplicates="drop")
                non_empty_bins = int(cats.cat.categories.size)
                bin_coverage = non_empty_bins / q
            except Exception:
                pass

            num_rows.append({
                "parameter": col,
                "n_with_value": n_any,
                "n_complete": n_complete_param,
                "n_pruned": n_pruned_param,
                "n_fail": n_fail_param,
                "n_running": n_running_param,
                "unique": nunique,
                "unique_ratio": float(unique_ratio),
                "min": vmin, "p25": v25, "median": v50, "p75": v75, "max": vmax,
                "search_low": low, "search_high": high, "search_log": log_flag, "search_step": step,
                "span_ratio": float(span_ratio),
                "non_empty_bins": non_empty_bins, "bin_coverage": bin_coverage,
            })

        else:
            # categorical
            levels = s_all[has_val]
            counts = levels.value_counts(dropna=False)  # include actual nulls if any slipped
            nunique_obs = int(counts.size)

            # distribution info
            n_choices_decl = meta.get(pname, {}).get("n_choices")
            # Use declared choices if available; otherwise observed uniques
            n_base = int(n_choices_decl) if n_choices_decl else nunique_obs

            # coverage ratio
            coverage_ratio = np.nan
            if n_choices_decl:
                coverage_ratio = nunique_obs / n_choices_decl

            # imbalance metrics
            imbalance_ratio = float(counts.max() / max(1.0, counts.mean()))
            probs = (counts / counts.sum()).values
            ent = float(-(probs * np.log(probs + 1e-12)).sum())  # natural log
            max_ent = np.log(n_base) if n_base and n_base > 0 else np.nan
            ent_ratio = float(ent / max_ent) if (max_ent and np.isfinite(max_ent) and max_ent > 0) else np.nan

            # string of top levels
            top = [f"{lvl}({int(counts[lvl])})" for lvl in counts.index[:max_levels]]
            if counts.size > max_levels:
                top.append(f"… +{counts.size - max_levels}")

            cat_rows.append({
                "parameter": col,
                "n_with_value": n_any,
                "n_complete": n_complete_param,
                "n_pruned": n_pruned_param,
                "n_fail": n_fail_param,
                "n_running": n_running_param,
                "unique_levels": nunique_obs,
                "choices_declared": n_choices_decl,
                "coverage_ratio": coverage_ratio,
                "imbalance_ratio": imbalance_ratio,
                "entropy_ratio": ent_ratio,
                "top_levels(count)": ", ".join(top),
            })

    num_df = pd.DataFrame(num_rows)
    if "parameter" in num_df.columns and not num_df.empty:
        num_df = num_df.sort_values("parameter").reset_index(drop=True)

    cat_df = pd.DataFrame(cat_rows)
    if "parameter" in cat_df.columns and not cat_df.empty:
        cat_df = cat_df.sort_values("parameter").reset_index(drop=True)
    return _style_coverage_tables(num_df, cat_df)

# ---------------------------
# 1D marginals
# ---------------------------

@dataclass
class Binned:
    bins: pd.Categorical              # ordered categorical mapping row -> bin
    level: list[str]                  # human-readable labels in bin order
    x_center: np.ndarray              # centers aligned to bin order
    x_left: np.ndarray                # left edges (float) or NaNs
    x_right: np.ndarray               # right edges (float) or NaNs
    effective: str                    # "categorical" | "quantile" | "uniform" | "unique" | "degenerate"
    treat_as_categorical: bool
    degenerate: bool
    degenerate_info: dict | None      # for the <=1 unique numeric case

def _interval_midpoints(index: pd.Index) -> np.ndarray:
    mids = np.full(len(index), np.nan, float)
    if isinstance(index, pd.IntervalIndex):
        mids = (index.left.astype(float) + index.right.astype(float)) / 2.0
    return mids

def _bin_param(
    d: pd.DataFrame,
    param_col: str,
    *,
    binning: str = "quantile",   # "quantile" | "uniform" | "unique"
    bins: int = 10,
    param_kind: str | None = None  # 'categorical' or 'numeric' (override dtype)
) -> Binned:
    x = _get_param(d, param_col)

    if param_kind not in (None, "categorical", "numeric"):
        raise ValueError("param_kind must be one of: None, 'categorical', 'numeric'")
    
    # Decide treatment 
    if param_kind == "categorical":
        treat_as_categorical = True
    elif param_kind == "numeric":
        treat_as_categorical = False
    else: # param_kind is None
        treat_as_categorical = (not is_numeric_dtype(x)) or is_bool_dtype(x)

    # --------- Categorical treatment ----------
    if treat_as_categorical:
        # Boolean special-case: enforce [False, True] order and include both levels
        if is_bool_dtype(x) or set(pd.Series(x).dropna().unique()).issubset({0, 1, True, False, "True", "False"}):
            x_norm = (
                pd.Series(x)
                .map({"True": True, "False": False, 1: True, 0: False, True: True, False: False})
            )
            bins_ordered = pd.Categorical(x_norm, categories=[False, True], ordered=True)
            level_index = bins_ordered.categories
            x_center = np.array([0.0, 1.0])
            x_left = np.array([np.nan, np.nan])
            x_right = np.array([np.nan, np.nan])
            level = [str(v) for v in level_index]

            return Binned(
                bins=bins_ordered,
                level=level,
                x_center=x_center,
                x_left=x_left,
                x_right=x_right,
                effective="categorical",
                treat_as_categorical=True,
                degenerate=False,
                degenerate_info=None,
            )
        
        # General categorical path (non-boolean)
        # Each distinct level is its own bin; preserve numeric order if labels are numeric-like
        bins_idx = x.astype("category")
        # Try to order by numeric value of the labels; else keep label order
        try:
            level_values = pd.to_numeric(bins_idx.cat.categories, errors="raise")
            order = np.argsort(level_values)
            ordered_levels = bins_idx.cat.categories[order]
            level_index = pd.CategoricalIndex(ordered_levels, ordered=True)
        except Exception:
            ordered_levels = bins_idx.cat.categories
            level_index = pd.CategoricalIndex(ordered_levels, ordered=True)

        # Recode bins to the ordered categories 
        bins_ordered = bins_idx.cat.set_categories(level_index, ordered=True)

        level_index = bins_ordered.cat.categories
        x_center = np.arange(len(level_index), dtype=float)
        x_left = np.full(len(level_index), np.nan)
        x_right = np.full(len(level_index), np.nan)
        level = level_index.astype(str).to_list()

        return Binned(
            bins=bins_ordered,
            level=level,
            x_center=x_center,
            x_left=x_left,
            x_right=x_right,
            effective="categorical",
            treat_as_categorical=True,
            degenerate=False,
            degenerate_info=None,
        )

    # --------- Numeric treatment ----------
    xn = pd.to_numeric(x, errors="coerce")
    n_unique = xn.dropna().nunique()

    if n_unique <= 1:
        single_center = float(xn.median())
        degenerate_info = {
            "level": ["all"],
            "x_center": np.array([single_center], dtype=float),
            "x_left": np.array([np.nan]),
            "x_right": np.array([np.nan]),
        }
        # ordered categorical with a single category
        bins_c = pd.Categorical(["all"] * len(d), categories=["all"], ordered=True)
        return Binned(
            bins=bins_c,
            index=pd.CategoricalIndex(["all"], ordered=True),
            level=["all"],
            x_center=np.array([single_center], dtype=float),
            x_left=np.array([np.nan]),
            x_right=np.array([np.nan]),
            effective="degenerate",
            treat_as_categorical=False,
            degenerate=True,
            degenerate_info=degenerate_info,
        )

    # Choose effective binning
    effective = "unique" if (binning == "quantile" and bins >= n_unique) else binning

    if effective == "quantile":
        # Quantile bins via qcut, and make sure it's ordered
        b = pd.qcut(xn, q=min(bins, n_unique), duplicates="drop")
        b = b.cat.set_categories(b.cat.categories, ordered=True)

        intervals = b.cat.categories  # IntervalIndex
        centers = _interval_midpoints(intervals)
        x_left = intervals.left.astype(float)
        x_right = intervals.right.astype(float)
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        level = [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left, x_right)]

        return Binned(
            bins=b,
            level=level,
            x_center=centers,
            x_left=x_left,
            x_right=x_right,
            effective="quantile",
            treat_as_categorical=False,
            degenerate=False,
            degenerate_info=None,
        )

    elif effective == "uniform":
        # Same heuristic cap + uniform bins via cut
        nb = min(bins, max(2, int(np.sqrt(n_unique))))
        b = pd.cut(xn, bins=nb)
        b = b.cat.set_categories(b.cat.categories, ordered=True)

        intervals = b.cat.categories  # IntervalIndex
        centers = _interval_midpoints(intervals)
        x_left = intervals.left.astype(float)
        x_right = intervals.right.astype(float)
        _fmtL = _best_num_formatter_for_series(x_left)
        _fmtR = _best_num_formatter_for_series(x_right)
        level = [f"({_fmtL(L)}, {_fmtR(R)}]" for L, R in zip(x_left, x_right)]
        return Binned(
            bins=b,
            level=level,
            x_center=centers,
            x_left=x_left,
            x_right=x_right,
            effective="uniform",
            treat_as_categorical=False,
            degenerate=False,
            degenerate_info=None,
        )

    else:  # "unique"
        # One bin per unique numeric value; keep numeric order in the index
        idx = pd.Index(xn.dropna().unique()).astype(float).sort_values()
        bins_c = pd.Categorical(xn, categories=idx, ordered=True)

        # Geometry for plotting: centers = the value; no edges
        x_center = idx.to_numpy(dtype=float)
        x_left = np.full(len(idx), np.nan)
        x_right = np.full(len(idx), np.nan)
        _fmt = _best_num_formatter_for_series(xn)
        level = [_fmt(v) for v in idx]
        return Binned(
            bins=bins_c,
            level=level,
            x_center=x_center,
            x_left=x_left,
            x_right=x_right,
            effective="unique",
            treat_as_categorical=False,
            degenerate=False,
            degenerate_info=None,
        )

def marginal_1d(
    df: pd.DataFrame,
    param_col: str,
    *,
    objective: str = "value",
    binning: str = "quantile",
    bins: int = 10,
    min_count: int = 2,
    compute_shares: bool = True,
    top_k: int = 10,
    top_frac: float = 0.20,
    minimize: bool = True,
    param_kind: str | None = None
) -> pd.DataFrame:
    param_col = _param_col(param_col)
    if param_col not in df.columns:
        raise KeyError(f"Parameter column '{param_col}' not found in DataFrame.")
    if objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame.")
    d = df[(df["state"] == "COMPLETE") & df[objective].notna() & df[param_col].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "level","x_center","x_left","x_right","count","mean","median","std","p25","p75",
            "share_topK","share_topFrac"
        ])

    y = d[objective]

    binfo = _bin_param(
        d, param_col,
        binning=binning,
        bins=bins,
        param_kind=param_kind
    )

    if binfo.degenerate:
        out = pd.DataFrame({
            "level": binfo.degenerate_info["level"],
            "x_center": binfo.degenerate_info["x_center"],
            "x_left": binfo.degenerate_info["x_left"],
            "x_right": binfo.degenerate_info["x_right"],
            "count": [len(d)],
            "mean": [y.mean()],
            "median": [y.median()],
            "std": [y.std(ddof=0)],
            "p25": [y.quantile(0.25)],
            "p75": [y.quantile(0.75)],
            "share_topK": [np.nan],
            "share_topFrac": [np.nan],
        })
        out.loc[out["count"] < int(min_count), ["mean","median","std","p25","p75"]] = np.nan
        return out

    d["__bin__"] = binfo.bins  # ordered categorical
    grp = d.groupby("__bin__", dropna=False, observed=False)

    # Full category index for all bins, including empty ones
    bin_index = d["__bin__"].cat.categories

    # Aggregations, reindexed so lengths match the bin metadata
    count  = grp.size().reindex(bin_index, fill_value=0)
    mean   = grp[objective].mean().reindex(bin_index)
    median = grp[objective].median().reindex(bin_index)
    std    = grp[objective].std(ddof=0).reindex(bin_index)
    p25    = grp[objective].quantile(0.25).reindex(bin_index)
    p75    = grp[objective].quantile(0.75).reindex(bin_index)

    out = pd.DataFrame({
        "level": binfo.level,
        "x_center": np.asarray(binfo.x_center, dtype=float),
        "x_left": np.asarray(binfo.x_left, dtype=float),
        "x_right": np.asarray(binfo.x_right, dtype=float),
        "count": count.to_numpy(),
        "mean": mean.to_numpy(),
        "median": median.to_numpy(),
        "std": std.to_numpy(),
        "p25": p25.to_numpy(),
        "p75": p75.to_numpy(),
    })

    if compute_shares:
        d_sorted = d.sort_values(objective, ascending=minimize)

        if "number" in d_sorted.columns:
            k = max(1, min(top_k, len(d_sorted)))
            best_ids = set(d_sorted.head(k)["number"])
            d["__is_topK"] = d["number"].isin(best_ids).astype(float)
        else:
            d["__is_topK"] = np.nan

        frac = max(1e-6, min(1.0, float(top_frac)))
        k_cut = max(1, int(np.ceil(frac * len(d_sorted))))
        cutoff = d_sorted.iloc[k_cut - 1][objective]
        d["__is_topFrac"] = (
            (d[objective] <= cutoff) if minimize else (d[objective] >= cutoff)
        ).astype(float)

        shares = d.groupby("__bin__", observed=False)[["__is_topK", "__is_topFrac"]].mean()
        out["share_topK"] = shares["__is_topK"].to_numpy()
        out["share_topFrac"] = shares["__is_topFrac"].to_numpy()
    else:
        out["share_topK"] = np.nan
        out["share_topFrac"] = np.nan

    out.loc[out["count"] < int(min_count), ["mean","median","std","p25","p75","share_topK","share_topFrac"]] = np.nan
    out = out.sort_values("x_center", kind="mergesort").reset_index(drop=True)
    return out

def _build_xticklabels_from_table(
        tbl: pd.DataFrame | dict, 
        *,
        max_labels: int = 16, 
        force_levels_for_labels: bool = False
    ) -> tuple[np.ndarray, list[str]]:
    """Return (xticks, xlabels) built from table's x_center and interval edges if present."""
    x = tbl["x_center"].to_numpy()

    if force_levels_for_labels:
        labels = tbl["level"].astype(str).tolist()
    else:
        # treat as categorical (and use levels) if no edges, else use centers with best numeric formatting
        if "x_left" not in tbl.columns or tbl["x_left"].isna().all() or \
           "x_right" not in tbl.columns or tbl["x_right"].isna().all():
            labels = tbl["level"].astype(str).tolist()
        else:
            _fmt = _best_num_formatter_for_series(pd.Series(x))
            labels = [_fmt(v) for v in x]

    # downsample tick labels to avoid clutter
    n = len(x)
    if n > max_labels and n > 0:
        step = math.ceil(n / max_labels)
        keep_idx = np.arange(0, n, step, dtype=int)
        return x[keep_idx], [labels[i] for i in keep_idx]
    return x, labels

def plot_marginal_1d_on_ax(
    ax: plt.Axes,
    tbl: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "objective",
    use_semilogx: bool = False,
    use_median: bool = True,
    show_std: bool = True,
):
    """Same as plot_marginal_1d_from_table, but draws on a provided Matplotlib Axes."""
    if tbl.empty or tbl["x_center"].isna().all():
        ax.set_axis_off()
        return

    y = (tbl["median"] if use_median else tbl["mean"]).to_numpy()
    x = tbl["x_center"].to_numpy()
    s = tbl["std"].to_numpy()

    if use_semilogx:
        ax.set_xscale("symlog", linthresh=1e-2)
    ax.plot(x, y, marker="o")
    if show_std and np.isfinite(s).any():
        ax.fill_between(x, y - s, y + s, alpha=0.20, linewidth=0)

    xticks, xlabels = _build_xticklabels_from_table(tbl, max_labels=16)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def _make_grid(n_panels, *, panel_size=None, ncols=None):
    # auto columns
    if ncols is None:
        ncols = 1 if n_panels == 1 else (2 if n_panels == 2 else 3)
    nrows = math.ceil(n_panels / ncols)

    if panel_size is None:
        panel_size = (8,4) if ncols == 1  else (4,3)

    # compute global figsize
    fig_w = panel_size[0] * ncols
    fig_h = panel_size[1] * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained"   
    )
    return fig, axes

def _compute_allowed_cols(
    df: pd.DataFrame,
    params: Iterable[str] | None,
    non_params_to_allow: Iterable[str]
):
    if params is not None:
        params = [_param_col(p) for p in params]
    else:
        params = _param_cols(df)
    params_allowed = params
    
    non_params_allowed = []
    param_cols_df = _param_cols(df)
    for p in non_params_to_allow:
        if p not in df.columns:
            print(f"Warning: non-parameter column '{p}' not found in DataFrame; ignoring.")
        elif p in param_cols_df or _param_col(p) in param_cols_df:
            print(f"Warning: '{p}' looks like a parameter column but was included in `non_params_to_allow`; ignoring.")
        else:
            non_params_allowed.append(p)

    # all allowed columns
    allowed = params_allowed + non_params_allowed
    return allowed

def plot_marginals_1d(
    df: pd.DataFrame,
    *,
    objective: str = "value",
    params: Iterable[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    ncols: int | None = None,              # auto: 1 if 1 param, 2 if 2, else 3
    panel_size: tuple[float, float] | None = None,
    # marginal/plot settings
    binning_numeric: str = "quantile",
    bins_numeric: int = 8,
    min_count: int = 2,
    use_median: bool = True,
    show_std: bool = True,
    use_semilogx: list[str] = ["train.learning_rate"],
    param_kinds: dict[str, str] | None = None,
    title_prefix: str = "Marginal of ",
):
    # Ensure using full params names
    use_semilogx_cols = [_param_col(p) for p in use_semilogx]

    # Compute allowed cols
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if not allowed:
        print("No parameters or non-parameter columns to plot.")
        return

    # Prepare grid
    n = len(allowed)
    fig, axes = _make_grid(n, panel_size=panel_size, ncols=ncols)
    nrows, ncols = axes.shape

    plotted = False
    for i, pcol in enumerate(allowed):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        kind = (param_kinds or {}).get(pcol, _guess_param_kind(_get_param(df, pcol)))
        tbl = marginal_1d(
            df, pcol, 
            objective=objective,
            binning=(binning_numeric if kind == "numeric" else "unique"),
            bins=bins_numeric, min_count=min_count,
            compute_shares=False, minimize=True,
            param_kind=kind
        )

        semilog_this = (pcol in use_semilogx_cols) and (kind == "numeric")
        plot_marginal_1d_on_ax(
            ax, tbl,
            title=f"{title_prefix}{pcol.replace('params_', '', 1).replace('user_attrs_', '', 1)}",
            xlabel="",
            ylabel="value" if c == 0 else "",
            use_median=use_median,
            show_std=show_std,
            use_semilogx=semilog_this,
        )
        if not tbl.empty:
            plotted = True

    # hide unused
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_axis_off()

    if plotted:
        fig.suptitle(f"Marginal Parameter Effects - obj: {objective}")
        plt.show()
    else:
        plt.close(fig)
        print("Nothing to plot.")

def display_marginal_1d(
    tbl: pd.DataFrame,
    *,
    minimize: bool = True,
    include_spread: bool = True,   # include p25/p75/std if supported
    include_shares: bool = True,   # include share_topK/share_topFrac if present
) -> Styler:
    """
    Pretty display for a 1D marginal table produced by `marginal_1d`.

    - Auto-detects kind when not provided.
    - Hides irrelevant columns.
    - Formats floats compactly; percentages for share columns.
    - Green gradient on the chosen stat, blue gradient on Std.
    - For interval-binned numerics, Level is rendered as "(left, right]" with formatted bounds.
    """

    # helper
    def _has(col):
        return (col in tbl.columns) and (not tbl[col].isna().all())

    # Build a working copy
    df = tbl.copy()

    # --- pick columns
    cols = ["level", "count", "mean", "median"]

    # spread
    if include_spread:
        for c in ("p25","p75","std"):
            if _has(c): cols.append(c)

    # shares
    if include_shares:
        for c in ("share_topK","share_topFrac"):
            if _has(c): cols.append(c)

    # ensure existence & filter
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # nice display names
    rename_map = {
        "level": "Level",
        "count": "Count",
        "mean": "Mean Value",
        "median": "Median Value",
        "p25": "P25",
        "p75": "P75",
        "std": "Std",
        "share_topK": "Top-K share",
        "share_topFrac": "Top-Frac share",
    }
    df = df.rename(columns=rename_map)

    # formatting
    fmt_map = {}
    for c in df.columns:
        if c in ("Level",):
            continue
        if c in ("Count",):
            fmt_map[c] = "{:,.0f}".format
        elif c in ("Top-K share","Top-Frac share"):
            fmt_map[c] = _pct_fmt
        else:
            fmt_map[c] = _float_fmt

    # color gradients
    styler = df.style.hide(axis="index")
    # green on chosen stat
    to_prettify = [rename_map[s] for s in ("median", "mean", "p25", "p75") if s in cols]
    if to_prettify:
        styler = styler.background_gradient(
            subset=to_prettify,
            cmap=("Greens_r" if minimize else "Greens")
        )
    to_prettify_shares = [rename_map[s] for s in ("share_topK","share_topFrac") if s in cols]
    if to_prettify_shares:
        styler = styler.background_gradient(
            subset=to_prettify_shares,
            cmap=("Greens")
        )

    # blue on Std if present
    if "Std" in df.columns:
        styler = styler.background_gradient(subset=["Std"], cmap="Blues")

    return styler

def display_marginals_1d(
    df: pd.DataFrame,
    *,
    params: Iterable[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    objective: str = "value",
    bins_numeric: int = 8,
    top_k: int = 10,
    top_frac: float = 0.20,
    minimize: bool = True,
    include_spread: bool = True,
    include_shares: bool = True,
    param_kinds: dict[str, str] | None = None,
    display_tbls: bool = True,
) -> dict[str, Styler]:
    """
    Return dict of {param_name: Styler} for 1D marginals of specified params.

    If no params specified, all param columns in df are used.
    """
    # all allowed columns
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if not allowed:
        print("No parameters or non-parameter columns to display.")
        return {}

    out = {}
    if display_tbls:
        _LOGGER.info(f"Displaying 1D marginals for objective: '{objective}'.")
    for pcol in allowed:
        kind = (param_kinds or {}).get(pcol, _guess_param_kind(df[pcol]))
        tbl = marginal_1d(
            df, pcol, 
            objective=objective,
            binning=("quantile" if kind == "numeric" else "unique"),
            bins=bins_numeric, min_count=2,
            compute_shares=True, 
            top_k=top_k, top_frac=top_frac,
            minimize=minimize,
            param_kind=kind
        )
        sty = display_marginal_1d(
            tbl, minimize=minimize,
            include_spread=include_spread,
            include_shares=include_shares
        )
        if display_tbls:
            _LOGGER.info(f" === {pcol.replace('params_', '', 1)} === ")
            display(sty)
        out[pcol.replace("params_", "", 1)] = sty
    return out

def plot_param_importances(
    imps: pd.Series,
    *,
    top_n: int | None = None,
    normalize: bool = True,
    annotate: bool = True,
):
    # make sure we have importances as an ordered pd.Series
    s = pd.Series(imps, dtype=float).sort_values(ascending=False)

    if top_n is not None:
        s = s.iloc[:top_n]

    if normalize:
        total = s.sum()
        if total > 0:
            s = s / total

    # plot (height scales with number of params)
    n = len(s)
    fig_h = max(2.5, min(0.45 * n + 1.0, 12))  # compact but readable
    fig, ax = plt.subplots(figsize=(8, fig_h), constrained_layout=True)

    ax.barh(s.index, s.values) 
    ax.invert_yaxis()           # largest on top
    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importances")

    # x axis as percent if normalized
    if normalize:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.0%}"))
        ax.set_xlim(0, 1)

    # annotate bars
    if annotate:
        x_max = ax.get_xlim()[1]
        for y, v in enumerate(s.values):
            label = f"{v:.1%}" if normalize else f"{v:.3g}"
            ax.text(min(v, x_max) + 0.01 * x_max, y, label, va="center")

    return fig, ax

# ---------------------------
# Two-parameter interaction views
# ---------------------------

def marginal_2d(
    df: pd.DataFrame,
    param_a: str,
    param_b: str,
    *,
    objective: str = "value",
    binning: str = "quantile",   # "quantile" | "uniform" | "unique"
    bins_a: int = 5,
    bins_b: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Build pivot tables of objective statistics for interactions between two params.
    Returns dict with keys: 'median', 'mean', 'std', 'count'.
    Index: bins/levels of A. Columns: bins/levels of B.
    """
    if objective not in df.columns:
        raise KeyError(f"Objective column '{objective}' not found in DataFrame.")
    
    # filter rows
    param_a = _param_col(param_a)
    param_b = _param_col(param_b)
    d = df[
        (df["state"] == "COMPLETE")
        & df[objective].notna()
        & df[param_a].notna()
        & df[param_b].notna()
    ].copy()
    if d.empty:
        empty = {"median": pd.DataFrame(), "mean": pd.DataFrame(), "std": pd.DataFrame(), "count": pd.DataFrame()}
        return empty

    # bin both params (ordered categoricals)
    binfo_a = _bin_param(d, param_a, binning=binning, bins=bins_a)
    binfo_b = _bin_param(d, param_b, binning=binning, bins=bins_b)

    d["__A__"] = binfo_a.bins
    d["__B__"] = binfo_b.bins

    # observed=False -> include empty categories (full A×B grid)
    grp = d.groupby(["__A__", "__B__"], dropna=False, observed=False)
    agg = grp[objective].agg(["median", "mean", "std", "count"])

    # pivot each stat (no reset_index / no manual reindex needed)
    piv_median = agg["median"].unstack("__B__")
    piv_mean   = agg["mean"].unstack("__B__")
    piv_std    = agg["std"].unstack("__B__")
    piv_count  = agg["count"].unstack("__B__").fillna(0).astype(int)

    # replace axis tick labels with human-readable levels & rename clumns/index
    pa_name = param_a.replace("params_", "", 1)
    pb_name = param_b.replace("params_", "", 1)
    def _relabel_axis(piv: pd.DataFrame) -> pd.DataFrame:
        # Row labels (A)
        if len(binfo_a.level) == len(piv.index):
            piv.index = pd.Index(binfo_a.level, name=piv.index.name)
        # Column labels (B)
        if len(binfo_b.level) == len(piv.columns):
            piv.columns = pd.Index(binfo_b.level, name=piv.columns.name)
        piv = piv.rename_axis(index=pa_name)
        piv = piv.rename_axis(columns=pb_name)
        return piv

    return {
        "median": _relabel_axis(piv_median.copy()),
        "mean":   _relabel_axis(piv_mean.copy()),
        "std":    _relabel_axis(piv_std.copy()),
        "count":  _relabel_axis(piv_count.copy()),
        "binfo": (binfo_a, binfo_b),
    }

import plotly.graph_objects as go

def plot_marginal_2d(
    pivots: dict[str, Any],
    statistic: str,
    *,
    title: str,
    minimize: bool = True,
    colorscale_minimize: str = "Viridis",
    colorscale_maximize: str = "Viridis_r",
    colorbar_title: str | None = None,
    show_text: bool = False,
    text_fmt: str = ".3g",
) -> go.Figure:
    """
    Simple Plotly heatmap for a two_param_summary pivot.

    - Tick *positions* use bin centers from pivots["binfo"] = (binfo_a, binfo_b), else positional centers.
    - Tick *labels* prefer centers (formatted) when available, else the pivot's level labels.
    - Hover shows param names + *level labels* (not centers).
    - Axis titles use the param names when available.
    """
    if statistic not in pivots:
        raise ValueError(f"Statistic '{statistic}' not in pivots: {list(pivots.keys())}")

    pivot: pd.DataFrame = pivots[statistic]
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — (empty)")
        return fig

    Z = pivot.to_numpy(dtype=float)
    n_rows, n_cols = Z.shape

    # infer param names (A=rows/index, B=cols/columns)
    pname_a = pivot.index.name if pivot.index.name else "param A"
    pname_b = pivot.columns.name if pivot.columns.name else "param B"

    # centers: prefer from binfo (binfo_a = rows, binfo_b = cols), else positional
    binfo = pivots.get("binfo")

    def _safe_centers(b) -> np.ndarray | None:
        if not hasattr(b, "x_center"):
            return None
        arr = np.asarray(getattr(b, "x_center"), dtype=float)  # coerce to float array
        if arr.ndim != 1 or not np.isfinite(arr).all():
            return None
        return arr

    x_centers = y_centers = None
    centers_from_binfo_x = centers_from_binfo_y = False

    if isinstance(binfo, tuple) and len(binfo) == 2:
        binfo_a, binfo_b = binfo  # rows=A, cols=B
        yc = _safe_centers(binfo_a)
        xc = _safe_centers(binfo_b)

        if yc is not None and len(yc) == n_rows:
            y_centers = yc
            centers_from_binfo_y = True
        if xc is not None and len(xc) == n_cols:
            x_centers = xc
            centers_from_binfo_x = True

    if x_centers is None:
        x_centers = np.arange(n_cols, dtype=float)  # positional fallback
    if y_centers is None:
        y_centers = np.arange(n_rows, dtype=float)  # positional fallback

    if len(x_centers) != n_cols or len(y_centers) != n_rows:
        raise ValueError("x_centers / y_centers lengths must match pivot shape.")

    # z-range & colorscale
    if not np.isfinite(Z).any():
        fig = go.Figure()
        fig.update_layout(title=f"{title} — (all values are NaN)")
        return fig

    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    if zmin == zmax:
        eps = 1e-12
        zmin, zmax = zmin - eps, zmax + eps

    colorscale = colorscale_minimize if minimize else colorscale_maximize

    # Optional per-cell text
    text = None
    if show_text:
        text = np.empty_like(Z, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                text[i, j] = (format(Z[i, j], text_fmt) if np.isfinite(Z[i, j]) else "")

    # tick labels: centers if available, else level labels
    def _fmt_centers(vals: np.ndarray) -> list[str]:
        # compact numeric formatting for axis labels, using same helpers as elsewhere
        _fmt = _best_num_formatter_for_series(vals)
        return [_fmt(v) for v in vals]

    x_ticktext = _fmt_centers(x_centers) if centers_from_binfo_x else [str(c) for c in pivot.columns]
    y_ticktext = _fmt_centers(y_centers) if centers_from_binfo_y else [str(r) for r in pivot.index]

    # hover uses param names + level labels (not centers)+
    customdata = np.stack(
        [
            np.broadcast_to(np.array(pivot.index, dtype=object)[:, None], Z.shape),   # row labels (A)
            np.broadcast_to(np.array(pivot.columns, dtype=object)[None, :], Z.shape)  # col labels (B)
        ],
        axis=-1
    )
    hovertemplate = (
        f"{pname_a}: %{{customdata[0]}}<br>"
        f"{pname_b}: %{{customdata[1]}}<br>"
        f"{statistic}: %{{z:.4g}}<extra></extra>"
    )

    # --- heatmap ---
    hm = go.Heatmap(
        x=x_centers, y=y_centers, z=Z,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title or f"objective ({statistic})"),
        hovertemplate=hovertemplate,
        customdata=customdata,
        text=text,
        texttemplate="%{text}" if show_text else None,
        showscale=True,
    )
    fig = go.Figure(hm)

    # --- axes: ticks at centers; titles from param names; labels from x/y_ticktext ---
    fig.update_xaxes(
        title_text=pname_b,
        tickmode="array", tickvals=x_centers, ticktext=x_ticktext,
        tickangle=45
    )
    fig.update_yaxes(
        title_text=pname_a,
        tickmode="array", tickvals=y_centers, ticktext=y_ticktext
    )

    fig.update_layout(
        title=title,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig

import itertools
from plotly.subplots import make_subplots

def plot_marginals_2d(
    df: pd.DataFrame,
    *,
    params: Iterable[str] | None = None,
    non_params_to_allow: Iterable[str] = [],
    objective: str = "value",
    as_first: str | None = None,          # param to pin to first axis (if present)
    statistic: Literal["median", "mean", "std", "count"] = "median",
    title: str | None = None,
    # binning / pivot controls (passed into marginal_2d)
    binning: Literal["quantile", "uniform", "unique"] = "quantile",
    bins_a: int = 5,
    bins_b: int | None = None,  # if None -> bins_a
    # plotting controls (passed into plot_marginal_2d)
    minimize: bool = True,
    colorscale_minimize: str = "Viridis",
    colorscale_maximize: str = "Viridis_r",
    show_text: bool = False,
    text_fmt: str = ".3g",
    # layout controls
    ncols: int = 2,
    share_colorscale: bool = True,      # uniform zmin/zmax across all subplots
    show_single_colorbar: bool = True,  # show one colorbar (right side)
) -> tuple[go.Figure, dict[tuple[str, str], dict[str, pd.DataFrame]]]:
    """
    Build a grid of heatmaps for all pairwise combos of `params` using `marginal_2d`
    and `plot_marginal_2d` (your single-heatmap helper).

    Returns (figure, pivots_by_pair). Keys of pivots_by_pair are (param_a, param_b).
    """
    bins_b = bins_b if bins_b is not None else bins_a

    # all allowed columns
    allowed = _compute_allowed_cols(df, params, non_params_to_allow)
    if len(allowed) < 2:
        _LOGGER.warning("Need at least two parameters or non-parameter columns to plot pairwise marginals.")
        fig = go.Figure()
        fig.update_layout(title=title or "Pairwise heatmaps (not enough parameters)")
        return fig, {}

    # --- build & order pairs, with optional pinning of one param to an axis ---
    all_pairs = list(itertools.combinations(allowed, 2))

    def _pin(pair: tuple[str, str]) -> tuple[str, str]:
        a, b = pair
        if as_first is None:
            return a, b
        if a == as_first:
            return b, a
        return a, b

    if as_first:
        as_first = _param_col(as_first)
        if as_first not in params:
            _LOGGER.warning(f"as_first param '{as_first}' not found in params; ignoring pinning.")
            as_first = None
            ordered_pairs = all_pairs
        else:
            priority = []
            others = []
            for p in all_pairs:
                if as_first in p:
                    priority.append(_pin(p))
                else:
                    others.append(p)
            # keep priority first (already pinned), then the rest (original orientation)
            ordered_pairs = priority + others
    else:
        ordered_pairs = all_pairs

    if not ordered_pairs:
        fig = go.Figure()
        fig.update_layout(title=title or "Pairwise heatmaps (no parameter pairs)")
        return fig, {}

    # --- compute pivots + global z-range ---
    pivots_by_pair: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    global_min = float("inf")
    global_max = float("-inf")

    for a, b in ordered_pairs:
        piv = marginal_2d(
            df,
            param_a=a,
            param_b=b,
            objective=objective,
            binning=binning,
            bins_a=bins_a,
            bins_b=bins_b,
        )
        pivots_by_pair[(a, b)] = piv

        if statistic in piv and not piv[statistic].empty:
            Z = piv[statistic].to_numpy(dtype=float)
            if np.isfinite(Z).any():
                zmin = float(np.nanmin(Z))
                zmax = float(np.nanmax(Z))
                global_min = min(global_min, zmin)
                global_max = max(global_max, zmax)

    # fallback if everything was empty/NaN or flat
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        global_min, global_max = 0.0, 1.0

    # --- create subplot grid ---
    n = len(ordered_pairs)
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)

    subplot_titles = [f"{a} x {b}" for (a, b) in ordered_pairs]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
    )

    # --- add each subplot using your helper ---
    for k, (a, b) in enumerate(ordered_pairs, start=1):
        r = (k - 1) // ncols + 1
        c = (k - 1) % ncols + 1

        piv = pivots_by_pair[(a, b)]
        subfig = plot_marginal_2d(  # your single-heatmap function
            pivots=piv,
            statistic=statistic,
            title="",  # we'll use subplot_titles
            minimize=minimize,
            colorscale_minimize=colorscale_minimize,
            colorscale_maximize=colorscale_maximize,
            colorbar_title=statistic,
            show_text=show_text,
            text_fmt=text_fmt,
        )

        # Empty pivot → annotate and still propagate axis titles if present
        if len(subfig.data) == 0:
            fig.add_annotation(row=r, col=c, text="(empty)", showarrow=False, font=dict(size=12))
            fig.update_xaxes(title_text=subfig.layout.xaxis.title.text or b, row=r, col=c)
            fig.update_yaxes(title_text=subfig.layout.yaxis.title.text or a, row=r, col=c)
            continue

        trace = subfig.data[0]

        # Share colorscale: either via per-trace zmin/zmax, or a true shared coloraxis
        if share_colorscale and not show_single_colorbar:
            trace.update(zmin=global_min, zmax=global_max)
        if show_single_colorbar:
            # we'll attach all traces to a single coloraxis later
            trace.update(showscale=False)

        fig.add_trace(trace, row=r, col=c)

        # Copy axis formatting (tickvals/ticktext, titles) from the subfig
        xax = subfig.layout.xaxis
        yax = subfig.layout.yaxis
        fig.update_xaxes(
            title_text=(xax.title.text or b),
            tickmode=getattr(xax, "tickmode", "array"),
            tickvals=getattr(xax, "tickvals", None),
            ticktext=getattr(xax, "ticktext", None),
            tickangle=getattr(xax, "tickangle", 45),
            automargin=True,                 # <<< let Plotly make room for ticks/title
            title_standoff=8,                # <<< small gap between axis and title
            row=r, col=c,
        )
        fig.update_yaxes(
            title_text=(yax.title.text or a),
            tickmode=getattr(yax, "tickmode", "array"),
            tickvals=getattr(yax, "tickvals", None),
            ticktext=getattr(yax, "ticktext", None),
            automargin=True,
            title_standoff=8,
            row=r, col=c,
        )

    # --- layout + spacing polish ---
    main_title = title or f"Pairwise heatmaps - obj: {objective} - stat: {statistic}"
    fig.update_layout(
        title=dict(text=main_title),
        height=375*nrows,   
    )
    fig.update_xaxes(automargin=True, title_standoff=20)
    fig.update_yaxes(automargin=True, title_standoff=20)

    # --- de-collide subplot titles (they're annotations) ---
    # anchor from the bottom and push up a bit
    for ann in (fig.layout.annotations or []):
        ann.update(yanchor="bottom", yshift=5)

    # --- single shared colorbar via coloraxis (cleaner spacing) ---
    if show_single_colorbar:
        colorscale = colorscale_minimize if minimize else colorscale_maximize
        fig.update_layout(
            coloraxis=dict(
                cmin=global_min, cmax=global_max,
                colorscale=colorscale,
                colorbar=dict(title=statistic, x=1.02)
            )
        )
        # Attach all heatmaps to that shared coloraxis
        for tr in fig.data:
            if isinstance(tr, go.Heatmap):
                tr.update(coloraxis="coloraxis")

    return fig, pivots_by_pair

# ---------------------------
# Wrappers around Optuna visualizations
# ---------------------------

import optuna.visualization as ov

def _fmt_params(*, params: dict, max_param_items: int, sort_params: bool) -> str:
    items = sorted(params.items()) if sort_params else list(params.items())
    if max_param_items and len(items) > max_param_items:
        items = items[:max_param_items] + [("…", f"+{len(params)-max_param_items} more")]
    return "<br>".join(f"{k}: {v!r}" for k, v in items)

def plot_intermediate_values(
    study: optuna.study.Study,
    *,
    target_name: str = "value",
    max_param_items: int = 30,
    sort_params: bool = True,
    include_params: Optional[Dict[str, Any]] = None,
    exclude_params: Optional[Dict[str, Any]] = None,
    predicate: Optional[Callable[[optuna.trial.FrozenTrial], bool]] = None,
    dim_excluded: bool = False,
    dim_factor: float = 0.1
):
    """
    Plot intermediate values with hover text showing trial params, with optional filtering.

    Parameters
    ----------
    study : optuna.study.Study
        The Optuna study to visualize.
    target_name : str, default "value"
        Label for target on the Y axis (also shown in hover).
    max_param_items : int, default 30
        Maximum number of params to display in tooltip (truncated if exceeded).
    sort_params : bool, default True
        Sort params by key name in tooltip.
    include_params : dict[str, Any], optional
        Keep only trials whose params match ALL key/value pairs (==).
    exclude_params : dict[str, Any], optional
        Drop (or dim) trials whose params match ANY key/value pairs (==).
    predicate : callable(trial)->bool, optional
        Custom boolean filter applied after include/exclude dicts.
    dim_excluded : bool, default False
        If True, excluded trials are shown at opacity=0.3 instead of removed.
    dim_factor : float, default 0.1
        If dim_excluded is True, opacity of excluded trials is set to this value.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Interactive Plotly figure.
    """
    # Base plot from Optuna
    fig = ov.plot_intermediate_values(study)

    # Align trials with traces
    trials_with_iv = [t for t in study.trials if t.intermediate_values]

    def _passes_include(trial: optuna.trial.FrozenTrial) -> bool:
        if not include_params:
            return True
        for k, v in include_params.items():
            if trial.params.get(k, None) != v:
                return False
        return True

    def _passes_exclude(trial: optuna.trial.FrozenTrial) -> bool:
        if not exclude_params:
            return True
        for k, v in exclude_params.items():
            if trial.params.get(k, object()) == v:
                return False
        return True

    def _passes_predicate(trial: optuna.trial.FrozenTrial) -> bool:
        return True if predicate is None else bool(predicate(trial))

    keep_mask = [
        (_passes_include(t) and _passes_exclude(t) and _passes_predicate(t))
        for t in trials_with_iv
    ]

    # If not dimming, just drop excluded trials
    if not dim_excluded:
        if keep_mask and (not all(keep_mask)):
            new_traces = [tr for keep, tr in zip(keep_mask, fig.data) if keep]
            fig.data = tuple(new_traces)
            trials_with_iv = [t for t, keep in zip(trials_with_iv, keep_mask) if keep]
    else:
        # Keep all traces, but reduce opacity on excluded ones
        for keep, trace in zip(keep_mask, fig.data):
            if not keep:
                trace.update(opacity=dim_factor)

    # Attach hover text to each (remaining or all) trial
    for trial, trace in zip(trials_with_iv, fig.data):
        params_str = _fmt_params(
            params=trial.params,
            max_param_items=max_param_items,
            sort_params=sort_params
        ) if trial.params else "(no params)"
        td = getattr(trial, "duration", None)

        hover_text = []
        for step, val in trial.intermediate_values.items():
            parts = [
                f"Trial {trial.number}",
                f"state: {trial.state.name}",
                f"step: {step}",
            ]
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                parts.append(f"{target_name}: {val:g}")
            parts.append(f"best_epoch: {trial.user_attrs.get('best_epoch', 'n/a')}")
            parts.append(f"duration: {_duration_fmt(td)}" if td else "duration: n/a")
            parts.append("--- params ---")
            parts.append(params_str)
            hover_text.append("<br>".join(parts))

        if hover_text:
            trace.hovertext = hover_text
            trace.hoverinfo = "text"
            trace.hovertemplate = "%{hovertext}<extra></extra>"

    return fig


def plot_optimization_history(
    study: optuna.study.Study,
    *,
    target=None,
    target_name: str = "value",
    max_param_items: int = 30,
    sort_params: bool = True,
):
    """
    Like optuna.visualization.plot_optimization_history, but each point's hover shows
    the trial's params (and optionally state/duration).

    Parameters
    ----------
    study : optuna.study.Study
        Source study.
    target : Callable[[FrozenTrial], float] | None
        Same as in Optuna's plot_optimization_history (for custom metrics).
    target_name : str
        Label for target on the Y axis (also shown in hover).
    max_param_items : int
        Limit number of param key/values shown (helps with very large spaces).
    sort_params : bool
        If True, params are sorted by key for stable display.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The augmented Plotly figure. Call `fig.show()` to display.
    """
    fig = ov.plot_optimization_history(study, target=target, target_name=target_name)

    trial_by_num = {t.number: t for t in study.trials}

    # Go through traces; add hover text for marker traces (skip best-line etc.)
    for tr in fig.data:
        # Only update scatter-like traces with per-point markers
        mode = getattr(tr, "mode", "") or ""
        if "markers" not in mode:
            continue
        xs = list(getattr(tr, "x", []))
        ys = list(getattr(tr, "y", []))
        if not xs or not ys:
            continue

        texts = []
        for x, y in zip(xs, ys):
            # X is trial number for Optuna's optimization history
            try:
                num = int(x)
            except (ValueError, TypeError):
                num = None
            t = trial_by_num.get(num) if num is not None else None

            parts = []
            parts.append(f"Trial {num}" if num is not None else "Trial")
            
            if t is not None:
                parts.append(f"state: {t.state.name}") 
            
            if y is not None and (isinstance(y, (int, float)) and math.isfinite(y)):
                parts.append(f"{target_name}: {y:g}") 
            
            if t is not None:
                best_epoch = t.user_attrs.get("best_epoch", None)
                parts.append(f"best_epoch: {best_epoch if best_epoch is not None else 'n/a'}")
                td = getattr(t, "duration", None)
                parts.append(f"duration: {_duration_fmt(td)}" if td else "duration: n/a")    
                parts.append("--- params ---")
                params_str = _fmt_params(params=t.params, max_param_items=max_param_items, sort_params=sort_params) if t.params else "(no params)"
                parts.append(params_str)

            texts.append("<br>".join(parts))

        # Attach as hover text; hide the default extra box
        tr.hovertext = texts
        tr.hoverinfo = "text"
        tr.hovertemplate = "%{hovertext}<extra></extra>"

    return fig
