# tft_pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Literal, Dict, Any, List
import logging
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler

from heat_forecast.pipeline.lstm import set_global_seed, is_for_endog_fut

# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFTModelConfig:
    """
    Architecture/fit hyperparameters for Darts' TFTModel.
    """
    input_chunk_length: int = 168
    output_chunk_length: int = 24
    hidden_size: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    num_attention_heads: int = 4
    loss_fn: Optional[Any] = None          
    random_state: Optional[int] = None
    batch_size: int = 64
    n_epochs: int = 20
    torch_device_str: Optional[str] = None  # "cuda" | "cpu" | None (auto)
    show_warnings: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass
class TrainConfig:
    """
    Training options.
    """
    use_es: bool = False   # early stopping
    es_patience: int = 5
    es_min_delta: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class DataConfig:
    """
    Data and split options (similar semantics to your pipeline).
    """
    stride: int = 1
    gap_hours: int = 0
    batch_size: int = 64
    num_workers: int = 0
    shuffle_train: bool = True
    pin_memory: Optional[bool] = None      # None => auto on CUDA

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class FeatureConfig:
    """
    Endog/exog feature engineering switches.

    Conventions
    -----------
    • We treat "future-safe" features as future_covariates (deterministic calendar or known-in-advance exog).
    • All others go to past_covariates.
    """
    exog_vars: Tuple[str, ...] = ("temperature",)
    endog_hour_lags: Tuple[int, ...] = ()
    include_exog_lags: bool = True
    time_vars: Tuple[str, ...] = ("hod", "dow", "moy", "wss")

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class NormalizeConfig:
    """
    Scaling policy (train-only, then apply to val/test).
    """
    scaler: Optional[Any] = None # to pass to Darts' Scaler()
    scale_time_vars: bool = False
    verbose: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TFTRunConfig:
    model: TFTModelConfig = field(default_factory=TFTModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    norm: NormalizeConfig = field(default_factory=NormalizeConfig)
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "features": self.features.to_dict(),
            "norm": self.norm.to_dict(),
            "seed": self.seed,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TFTPipeline:
    """
    End-to-end TFT pipeline using Darts.

    Input
    -----
    target_df: DataFrame with columns ['unique_id','ds','y']
    aux_df:    DataFrame with the same ['unique_id','ds'] index cols and exogenous features.
    """

    def __init__(
        self,
        target_df: pd.DataFrame,
        config: TFTRunConfig,
        aux_df: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._target_df = target_df
        self._aux_df = aux_df
        self.config = config
        self._logger = logger or logging.getLogger(__name__)

        # internal attributes initialized during data preparation
        self._endog_components = []
        self._time_components = []
        self._climate_components = []
        self._y = None
        self._past_covs = None
        self._endog_vars_not_for_future = None
        self._future_covs = None

        # internal attributes initialized during data preparation / loading
        self._start_train = None
        self._end_train = None
        self._scaler_t = None
        self._scaler_p = None
        self._scaler_f = None

        # internal attributes initialized during fit
        self._trainer = None
        self._history_df = None
        self._model = None
        self._n_params = None
        self._train_losses_orig = None
        self._val_losses_orig = None


        # --- Set seeds if requested ---
        lines_to_log = []
        if self.config.seed is not None:
            lines_to_log.append(f"setting random seed: {self.config.seed}")
            set_global_seed(self.config.seed)

        # --- Check coherence of params ---

        # Make sure a single id is used
        if not target_df['unique_id'].nunique() == 1:
            raise ValueError("The target_df must contain a single unique_id. Found: {}".format(target_df['unique_id'].unique()))
        
        # Make sure aux_df is provided if exog_vars is used
        if self.config.features.exog_vars and aux_df is None:
            raise ValueError("aux_df must be provided when exog_vars is used.")

        # Make sure the aux_df has the same unique_id
        if aux_df is not None:
            if not aux_df['unique_id'].nunique() == 1:
                raise ValueError("The aux_df must contain a single unique_id. Found: {}".format(aux_df['unique_id'].unique()))
            if not aux_df['unique_id'].unique()[0] == target_df['unique_id'].unique()[0]:
                raise ValueError("The unique_id in aux_df must match the one in target_df. Found: {} and {}".format(
                    aux_df['unique_id'].unique()[0], target_df['unique_id'].unique()[0]
                ))
        unique_id = target_df['unique_id'].unique()[0]
        if not unique_id in ['F1', 'F2', 'F3', 'F4', 'F5']:
            raise ValueError(f"Invalid unique_id: {unique_id}. Expected one of ['F1', 'F2', 'F3', 'F4', 'F5'].")
        
        # Save unique_id and target merged with aux
        self.unique_id = unique_id
        if aux_df is not None:
            self._target_plus_aux_df = target_df.merge(
                aux_df, 
                on=['unique_id', 'ds'], 
                how='inner', 
            )
        else:
            self._target_plus_aux_df = target_df.copy()

        # Set ds as index in target_plus_aux_df
        self._target_plus_aux_df['ds'] = pd.to_datetime(self._target_plus_aux_df['ds'])
        self._target_plus_aux_df = self._target_plus_aux_df.sort_values('ds').set_index('ds')

        if self.config.features.exog_vars:
            # Make sure exog vars exist in aux_df
            missing_vars = [var for var in self.config.features.exog_vars if var not in aux_df.columns]
            if missing_vars:
                raise ValueError(f"The following exogenous variables are missing in aux_df: {missing_vars}")
        
        tv = set(self.config.features.time_vars)
        allowed_time_vars = {"hod", "dow", "moy", "wss"}
        invalid_time_vars = tv - allowed_time_vars
        if invalid_time_vars:
            raise ValueError(f"Invalid time_vars: {invalid_time_vars}. Allowed values are {allowed_time_vars}.")
        allowed_and_present = tv & allowed_time_vars
        if not allowed_and_present:
            raise ValueError("At least one valid time_var must be specified.")

        self._logger.info("[pipe init] " + "; ".join(lines_to_log) + '.')
            
    # ------------------------------
    # Data Preparation
    # ------------------------------

    def generate_vars(self) -> None:
        """
        Create endogenous/exogenous matrices and target series.

        Steps
        -----
        - Lagged features (endog and, optionally, exog).
        - Time features (sin/cos of HOD/DOW/MOY, WSS, cold-season flag).
        - Head-trim to drop rows made invalid by lags/diffs/rolls.
        - NaN row drop with a warning.
        - Track endog columns not safe for decoder (leakage control).
        """
        df = self._target_plus_aux_df

        # Base frames
        endog_df = df[['y']].copy()
        exog_df  = df[list(self.config.features.exog_vars)].copy() if self.config.features.exog_vars else pd.DataFrame(index=df.index)
        self._climate_components = list(exog_df.columns) if not exog_df.empty else []

        # --- Lagged endogenous features ---
        lag_cols = {}
        max_lag = 0
        for lag in self.config.features.endog_hour_lags or []:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError(f"endog_hour_lags must contain positive ints, got {lag}")
            max_lag = max(max_lag, lag)
            lag_cols[f"y_lag{lag}"] = endog_df['y'].shift(lag)
            self._endog_components.append(f"y_lag{lag}")

        if lag_cols:
            endog_df = pd.concat([endog_df, pd.DataFrame(lag_cols, index=endog_df.index)], axis=1)
        
        # --- Lagged exogenous features ---
        base_exog_for_lags = list(self.config.features.exog_vars)
        if self.config.features.include_exog_lags and not exog_df.empty:
            exog_lag_cols = {}
            for lag in (self.config.features.endog_hour_lags or []):
                for c in base_exog_for_lags:
                    exog_lag_cols[f"{c}_lag{lag}"] = exog_df[c].shift(lag)
                    self._climate_components.append(f"{c}_lag{lag}")

            if exog_lag_cols:
                exog_df = pd.concat([exog_df, pd.DataFrame(exog_lag_cols, index=exog_df.index)], axis=1)

        # --- Trim head once (for lags) ---
        head_trim = max_lag
        if head_trim > 0:
            endog_df = endog_df.iloc[head_trim:]
            exog_df  = exog_df.iloc[head_trim:]

        # --- Time features (cyclical) ---
        idx = endog_df.index
        assert idx.equals(exog_df.index), "Indexes must be aligned."

        def sincos(x, period):
            x = np.asarray(x, dtype=np.float32)
            ang = 2.0 * np.pi * (x / period)
            return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)
        
        # Figure out which time primitives we need
        tv = set(self.config.features.time_vars)
        need_hod = "hod" in tv
        need_dow = ("dow" in tv) or ("wss" in tv)  # wss is derived from dow
        need_moy = "moy" in tv

        if need_hod:
            hour = idx.hour.values
        if need_dow:
            dow = idx.dayofweek.values
        if need_moy:
            month = idx.month.values

        feats_dict = {}

        if "hod" in self.config.features.time_vars:
            hour_sin, hour_cos = sincos(hour, 24)
            feats_dict["hour_sin"], feats_dict["hour_cos"] = hour_sin, hour_cos
        if "dow" in self.config.features.time_vars:
            dow_sin, dow_cos = sincos(dow, 7)
            feats_dict["dow_sin"], feats_dict["dow_cos"] = dow_sin, dow_cos
        if "moy" in self.config.features.time_vars:
            month_sin, month_cos = sincos(month - 1, 12)
            feats_dict["month_sin"], feats_dict["month_cos"] = month_sin, month_cos
        if "wss" in self.config.features.time_vars:
            wss   = np.where(dow == 5, 1, np.where(dow == 6, 2, 0)).astype(np.int32)  # weekday/sat/sun
            wss_sin, wss_cos = sincos(wss, 3)
            feats_dict["wss_sin"], feats_dict["wss_cos"] = wss_sin, wss_cos

        if feats_dict:
            feats_df = pd.DataFrame(feats_dict, index=idx).astype(np.float32)
            exog_df  = pd.concat([exog_df, feats_df], axis=1)
            self._time_components = list(feats_df.columns)

        # --- Final cleanup & assignments ---
        if endog_df.isna().any().any() or exog_df.isna().any().any():
            n_endog = int(endog_df.isna().sum().sum())
            n_exog  = int(exog_df.isna().sum().sum())
            raise ValueError(f"NaNs detected after feature generation (endog: {n_endog}, exog: {n_exog}). "
                             f"Please check your configuration and input data.")

        # --- Build TimeSeries objects ---
        endog_df = endog_df.astype(np.float32) if not endog_df.empty else endog_df
        exog_df  = exog_df.astype(np.float32) if not exog_df.empty else exog_df
        
        self._target = TimeSeries.from_dataframe(endog_df[['y']])

        self._endog_vars_not_for_future = [
            c for c in endog_df.columns
            if not (is_for_endog_fut(c, self.config.model.output_chunk_length) or c == 'y')
        ]
        fut_endog_df  = endog_df[self._endog_vars_not_for_future]
        past_endog_df = endog_df[[c for c in endog_df.columns if c not in self._endog_vars_not_for_future and c != 'y']]
        fut_covs_df   = exog_df if fut_endog_df.empty else pd.concat([fut_endog_df, exog_df], axis=1)

        self._future_covs = None if fut_covs_df.shape[1] == 0 else TimeSeries.from_dataframe(fut_covs_df)
        self._past_covs   = None if past_endog_df.shape[1] == 0 else TimeSeries.from_dataframe(past_endog_df)

        
        n_endog_past, n_endog_fut = past_endog_df.shape[-1], fut_endog_df.shape[-1]
        n_time = len(self.config.features.time_vars)*2
        n_climate_fut = len(exog_df.columns) - n_time
        self._logger.info(f"[gvars] features ready: "
            f"past_covs={self._past_covs.width} (endog={n_endog_past}, climate=0) | future_covs={self._future_covs.width} (endog={n_endog_fut}, climate={n_climate_fut}, time={n_time})")

    @property
    def target(self) -> Optional[TimeSeries]: return self._target
    @property
    def past_covs(self) -> Optional[TimeSeries]: return self._past_covs
    @property
    def future_covs(self) -> Optional[TimeSeries]: return self._future_covs


    def _fit_transformers(self, ts_train: TimeSeries, past_train: Optional[TimeSeries], fut_train: Optional[TimeSeries]) -> None:        
        """fit scalers on training data only. Return already transformed train data."""
        # Inizialize transformers
        transformer_t = Scaler(self.config.norm.scaler, **self.config.norm.kwargs)

        need_p = past_train is not None and past_train.width > 0
        transformer_p = Scaler(self.config.norm.scaler, **self.config.norm.kwargs) if need_p else None

        fc_set = set(self._future_covs.components) if self._future_covs is not None else set()
        tf_set = set(self._time_components)
        need_f = (len(fc_set - tf_set) > 0) or \
                (len(fc_set & tf_set) > 0 and self.config.norm.scale_time_vars) 
        transformer_f = Scaler(self.config.norm.scaler, **self.config.norm.kwargs) if need_f else None

        # Optionally mask time vars from scaling
        mask_f = None
        if need_f and not self.config.norm.scale_time_vars and self._future_covs is not None:
            mask_f = np.array([c not in self._time_components for c in self._future_covs.components])

        # Fit scalers on training data only and apply to train data only
        ts_tt   = transformer_t.fit_transform(ts_train)
        past_tt = transformer_p.fit_transform(past_train) if need_p else past_train
        fut_tt  = transformer_f.fit_transform(fut_train, component_mask=mask_f) if need_f else fut_train

        # Persist in self
        self._need_p = need_p
        self._need_f = need_f
        self._mask_f = mask_f
        self._transformer_t = transformer_t
        self._transformer_p = transformer_p if need_p else None
        self._transformer_f = transformer_f if need_f else None

        return ts_tt, past_tt, fut_tt

    def _apply_transforms(self):
        """Apply fitted scalers to store data, and persists transformed data to self."""
        self._target_transf = self._transformer_t.transform(self._target)
        if self._need_p and self._past_covs is not None:
            self._past_covs_transf = self._transformer_p.transform(self._past_covs)
        else:
            self._past_covs_transf = self._past_covs
        if self._need_f and self._future_covs is not None:
            self._future_covs_transf = self._transformer_f.transform(self._future_covs, component_mask=self._mask_f)
        else:
            self._future_covs_transf = self._future_covs

    def _build_model(self, pl_trainer_kwargs: dict) -> TFTModel:
        mc = self.config.model
        self._model = TFTModel(
            input_chunk_length=mc.input_chunk_length,
            output_chunk_length=mc.output_chunk_length,
            hidden_size=mc.hidden_size,
            lstm_layers=mc.lstm_layers,
            dropout=mc.dropout,
            num_attention_heads=mc.num_attention_heads,
            random_state=mc.random_state,
            batch_size=mc.batch_size,
            n_epochs=mc.n_epochs,
            pl_trainer_kwargs=pl_trainer_kwargs,
            loss_fn=mc.loss_fn,
        )

    def fit(self, end_train: pd.Timestamp, end_val: Optional[pd.Timestamp] = None) -> None:
        if self._target is None:
            self.generate_vars()

        mc = self.config.model
        tc = self.config.train

        pl_trainer_kwargs = {
            "accelerator": ("auto" if mc.torch_device_str is None else ("gpu" if mc.torch_device_str=="cuda" else "cpu"))
        }
        if (end_val is not None) and tc.use_es:
            from pytorch_lightning.callbacks import EarlyStopping
            early_stop = EarlyStopping(
                monitor="val_loss",   
                mode="min",
                patience=tc.es_patience,
                min_delta=tc.es_min_delta
            )
            pl_trainer_kwargs["callbacks"] = [early_stop]

        self._build_model(pl_trainer_kwargs)
            
        h = self.config.model.output_chunk_length
        ts_train, _ = self._target.split_after(end_train)
        past_train, _ = self._past_covs.split_after(end_train) if self._past_covs is not None else (None, None)
        fut_train, _ = self._future_covs.split_after(end_train + pd.Timedelta(hours=h)) if self._future_covs is not None else (None, None)

        ts_tt, past_tt, fut_tt = self._fit_transformers(ts_train, past_train, fut_train)
        self._apply_transforms()
        self._model.fit(
            series = ts_tt,
            past_covariates = past_tt,
            future_covariates = fut_tt,
            verbose = True
        )
        self._end_train = end_train

    def predict(self, n: int, cutoff: pd.Timestamp, alias: str = "TFT") -> pd.DataFrame:
        """
        Forecast next `n`, from `cutoff`+ 1h to `cutoff` + `n` hours. 
        Uses `n`= `output_chunk_length`, `cutoff` = last training timestamp by default.
        """
        if self._model is None:
            raise ValueError("Model is not trained yet. Please call fit() before predict().")
        
        if n is None:
            n = self.config.model.output_chunk_length
        if cutoff is None:
            cutoff = self._end_train

        ts_context, _ = self._target_transf.split_after(cutoff)
        past_context, _ = self._past_covs_transf.split_after(cutoff) if self._past_covs_transf is not None else (None, None)
        fut_context, _ = self._future_covs_transf.split_after(cutoff + pd.Timedelta(hours=n)) if self._future_covs_transf is not None else (None, None)

        pred_t = self._model.predict(
            n=n,
            series=ts_context,
            past_covariates=past_context,
            future_covariates=fut_context,
        )

        # invert transform
        pred = self._transformer_t.inverse_transform(pred_t)

        # Back to DataFrame
        s = pred.pd_series()
        pred_df = s.reset_index()
        pred_df.columns = ["ds", alias]  # robust
        pred_df.insert(0, "unique_id", self.unique_id)
        return pred_df

    def predict_many(
        self,
        n: int,
        start: pd.Timestamp,          # first cutoff to evaluate
        end: pd.Timestamp | None = None,   # optional last cutoff
        stride_hours: int = 1,        # gap between consecutive cutoffs
        alias: str = "TFT",
    ) -> pd.DataFrame:
        """
        Produce forecasts of length `n` at multiple cutoffs in [start, end],
        spaced by `stride_hours`. Returns a DataFrame with columns:
        unique_id, cutoff, ds, <alias>.
        """
        if self._model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        # contexts are already transformed in your pipeline
        ts = self._target_transf
        pc = self._past_covs_transf
        fc = self._future_covs_transf

        # run rolling forecasts
        fcsts = self._model.historical_forecasts(
            series=ts,
            past_covariates=pc,
            future_covariates=fc,
            start=start,
            forecast_horizon=n,
            stride=stride_hours,
            last_points_only=False,   # keep full n-step forecasts for each cutoff
            retrain=False,            # reuse the fitted weights
            verbose=True,
        )

        # fcsts is a list of TimeSeries, one per cutoff
        out_frames = []
        for f in fcsts:
            cutoff = f.start_time() - pd.Timedelta(hours=1)  # first pred is cutoff + 1 step
            s = f.to_series().rename(alias).reset_index()
            s.insert(0, "cutoff", cutoff)
            s.insert(0, "unique_id", self.unique_id)
            s.columns = ["unique_id", "cutoff", "ds", alias]
            out_frames.append(s)

        return pd.concat(out_frames, ignore_index=True)


    # ------------------------------
    # Rolling backtest (CV style)
    # ------------------------------

    def cross_validation(
        self,
        *,
        test_size: int,
        end_test: Optional[pd.Timestamp] = None,
        step_size: int = 1,
        alias: str = "TFT",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Rolling evaluation:
          - cutoffs spaced by step_size hours
          - each predicts the next T_out hours
        """
        if self._series is None:
            self.generate_vars()

        h = self.config.model.output_chunk_length
        idx = self._target.time_index
        end_test = end_test or idx[-1]
        steps = list(range(-test_size, -h + 1, step_size))
        results = []
        # Ensure model is trained once on full available training range prior to CV
        self.fit()

        for off in steps:
            cutoff = end_test + pd.Timedelta(hours=off)
            fc = self.predict_single_cutoff(cutoff=cutoff, alias=alias)
            # join truth
            mask = (fc["ds"] > cutoff) & (fc["ds"] <= cutoff + pd.Timedelta(hours=h))
            horizon = fc.loc[mask, ["ds", alias]].copy()
            truth = (self._target_df.set_index("ds")
                     .loc[horizon["ds"]]["y"]
                     .reset_index())
            merged = truth.merge(horizon, on="ds", how="left")
            merged.insert(0, "unique_id", self.unique_id)
            merged["cutoff"] = cutoff
            results.append(merged.rename(columns={"y": "y"}))

        out = pd.concat(results, ignore_index=True)
        if verbose:
            try:
                err = mape(TimeSeries.from_dataframe(out[["ds","y"]], time_col="ds", value_cols="y"),
                           TimeSeries.from_dataframe(out[["ds", alias]], time_col="ds", value_cols=alias))
                self._logger.info(f"[cv] MAPE over CV points: {err:.3f}")
            except Exception:
                pass
        return out

    # ------------------------------
    # Summaries
    # ------------------------------

    def describe_dataset(self) -> str:
        if self._series is None:
            self.generate_vars()
        parts = [
            f"Rows={len(self._series)}",
            f"Past covs={self._past_covs.width if self._past_covs is not None else 0}",
            f"Future covs={self._future_covs.width if self._future_covs is not None else 0}",
            f"T_in={self.config.model.input_chunk_length}",
            f"T_out={self.config.model.output_chunk_length}",
            f"Δ24={'yes' if self._diff is not None else 'no'}",
        ]
        return "[dataset] " + " | ".join(parts)

    def describe_model(self) -> str:
        mc = self.config.model
        parts = [
            "TFTModel",
            f"hidden={mc.hidden_size}",
            f"lstm_layers={mc.lstm_layers}",
            f"heads={mc.num_attention_heads}",
            f"dropout={mc.dropout}",
            f"epochs={mc.n_epochs}",
            f"batch_size={mc.batch_size}",
            f"lr={mc.lr}",
        ]
        return "[model] " + " | ".join(parts)
