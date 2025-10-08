
## Suggesters for LSTM

import optuna
import copy

from heat_forecast.utils.optuna import (
    register_suggester, 
    SuggesterDoc,
    ParamCat, ParamInt, ParamFloat
)
from heat_forecast.pipeline.lstm import LSTMRunConfig

_lstm_v1_doc = SuggesterDoc(
    summary="Baseline LSTM search space for 24-hour ahead forecasting.",
    params=[
        # Model
        ParamCat("model.head", ["linear", "mlp"]),
        ParamCat("model.hidden_size", [32, 64, 128]),
        ParamCat("model.num_layers", [1, 2]),
        ParamCat("model.dropout", [0.0, 0.15, 0.3]),

        # Windows
        ParamCat("model.input_len", [72, 168]),

        # Data
        ParamCat("data.batch_size", [64, 128]),

        # Training
        ParamFloat("train.learning_rate", 1e-4, 5e-3, log=True),
    ],
    notes=[
        "`norm.mode` is fixed to `global`.",
        "`train.n_epochs` is fixed to 35; early stopping trims excess epochs.",
        "The heavy combo of `model.hidden_size=128`, `model.num_layers=2`, and `model.input_len=168` is skipped (pruned early).",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number (and base_cfg.seed if provided).",
        "Other params use defaults.",
    ],
)

@register_suggester("lstm_v1", doc=_lstm_v1_doc)
def suggest_config_v1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model ----
    cfg.model.input_len  = trial.suggest_categorical("model.input_len", [72, 168])
    cfg.model.num_layers = trial.suggest_categorical("model.num_layers", [1, 2])
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [32, 64, 128])

    # Forbid a heavy combo, prune early
    if cfg.model.num_layers == 2 and cfg.model.input_len == 168 and cfg.model.hidden_size == 128:
        trial.set_user_attr("invalid_combo", True)
        raise optuna.TrialPruned("Skip hidden_size=128 for 2 layers & 168 input_len")

    cfg.model.head    = trial.suggest_categorical("model.head", ["linear", "mlp"])
    cfg.model.dropout = trial.suggest_categorical("model.dropout", [0.0, 0.15, 0.3])

    # ---- Training/Data ----
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 5e-4, 5e-3, log=True)
    cfg.train.n_epochs = 35
    cfg.data.batch_size = trial.suggest_categorical("data.batch_size", [64, 128])
    cfg.norm.mode = "global"

    return cfg

_lstm_v2_F1_doc = SuggesterDoc(
    summary="Constrained LSTM v2 search space for series F1 (feature-focused).",
    params=[
        # --- Model (paired via key) ---
        ParamCat("model.head_hidden_key", ["linear_64", "mlp_128"]),
        ParamCat("model.dropout", [0.0, 0.05, 0.10]),

        # --- Data ---
        ParamCat("data.batch_size", [32, 64]),

        # --- Training ---
        ParamFloat("train.learning_rate", 4e-4, 2e-3, log=True),

        # --- Features (via key) ---
        ParamCat("features.exog_vars_key", ["t", "t_h", "t_p", "t_w", "t_hpw"]),
        ParamCat("features.use_cold_season", [True, False]),
    ],
    notes=[
        "`model.input_len` is fixed to 72.",
        "`model.num_layers` is fixed to 2.",
        "`model.head` and `model.hidden_size` are derived from `model.head_hidden_key` "
        "with the mapping: linear_64 → ('linear', 64), mlp_128 → ('mlp', 128).",
        "`features.exog_vars` is derived from `features.exog_vars_key` with the mapping: "
        "t → ('temperature',), t_h → ('temperature','humidity'), t_p → ('temperature','pressure'), "
        "t_w → ('temperature','wind_speed'), t_hpw → ('temperature','humidity','pressure','wind_speed').",
        "`norm.mode` is fixed to `global`.",
        "`train.n_epochs` is fixed to 35; early stopping trims excess epochs.",
        "`features.cold_temp_threshold` is fixed to 13.8 (the 'best' threshold found while developing SARIMAX).",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number "
        "(and `base_cfg.seed` if provided).",
        "Other params use defaults.",
    ],
)


@register_suggester("lstm_v2_F1", doc=_lstm_v2_F1_doc)
def suggest_config_v2_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2

    # Enforce the allowed (head, hidden_size) pairs
    head_hidden_options = {
        "linear_64": ("linear", 64),
        "mlp_128":   ("mlp",   128),
    }
    head_hidden_key = trial.suggest_categorical(
        "model.head_hidden_key",
        list(head_hidden_options.keys())
    )
    cfg.model.head, cfg.model.hidden_size = head_hidden_options[head_hidden_key]

    # Low dropout only
    cfg.model.dropout = trial.suggest_categorical("model.dropout", [0.0, 0.05, 0.10])

    # ---- Training/Data ----
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 4e-4, 2e-3, log=True)
    cfg.train.n_epochs = 35
    cfg.data.batch_size = trial.suggest_categorical("data.batch_size", [32, 64])
    cfg.norm.mode = "global"

    # ---- Features ----
    exog_options = {
        "t":       ("temperature",),
        "t_h":     ("temperature", "humidity"),
        "t_p":     ("temperature", "pressure"),
        "t_w":     ("temperature", "wind_speed"),
        "t_hpw":   ("temperature", "humidity", "pressure", "wind_speed"),
    }
    exog_key = trial.suggest_categorical("features.exog_vars_key", list(exog_options.keys()))
    cfg.features.exog_vars = exog_options[exog_key]

    cfg.features.use_cold_season = trial.suggest_categorical("features.use_cold_season", [True, False])
    cfg.features.cold_temp_threshold = 13.8 # using best_thresholds[id]

    return cfg

_lstm_v3_F1_doc = SuggesterDoc(
    summary="Constrained LSTM v3 search space for series F1 (feature-focused).",
    params=[
        # --- Model (paired via key) ---
        ParamCat("model.head_hidden_key", ["linear_64", "mlp_128"]),

        # --- Features ---
        ParamCat("features.use_differences", [True, False]),
        ParamCat("features.hour_averages_key", ["none", "2days"]),
        ParamCat("features.lag_key", ["7days", "7day +ex", "7days_1day", "7day_1day +ex"]),
    ],
    notes=[
        "`model.input_len` is fixed to 72.",
        "`model.num_layers` is fixed to 2.",
        "`model.dropout` is fixed to 0.0.",
        "`model.head` and `model.hidden_size` are derived from `model.head_hidden_key` "
        "with the mapping: linear_64 → ('linear', 64), mlp_128 → ('mlp', 128).",
        "`train.learning_rate` is fixed to 5e-4.",
        "`train.n_epochs` is fixed to 35; early stopping may trim excess epochs.",
        "`norm.mode` is fixed to `global`.",
        "`features.hour_averages` is derived from `features.hour_averages_key` with the mapping: "
        "none → (), 2days → (48,).",
        "`features.endog_hour_lags` and `features.include_exog_lags` are derived from `features.lag_key` "
        "with the mapping: "
        "7days → ((168,), False); "
        "7day +ex → ((168,), True); "
        "7days_1day → ((168, 24), False); "
        "7day_1day +ex → ((168, 24), True).",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number "
        "(and `base_cfg.seed` if provided).",
        "Other params use defaults from `base`.",
    ],
)

@register_suggester("lstm_v3_F1", doc=_lstm_v3_F1_doc)
def suggest_config_v3_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2
    cfg.model.dropout    = 0.0

    # Enforce the allowed (head, hidden_size) pairs
    head_hidden_options = {
        "linear_64": ("linear", 64),
        "mlp_128":   ("mlp",   128),
    }
    head_hidden_key = trial.suggest_categorical(
        "model.head_hidden_key",
        list(head_hidden_options.keys())
    )
    cfg.model.head, cfg.model.hidden_size = head_hidden_options[head_hidden_key]

    # ---- Training/Data ----
    cfg.train.learning_rate = 5e-4
    cfg.train.n_epochs = 35
    # cfg.data.batch_size = 64 (default)
    cfg.norm.mode = "global"

    # ---- Features ----
    cfg.features.use_differences = trial.suggest_categorical("features.use_differences", [True, False])

    hour_avg_options = {
        "none": (),
        "2days": (48,),
    }
    hour_averages_key = trial.suggest_categorical("features.hour_averages_key", list(hour_avg_options.keys()))
    cfg.features.hour_averages = hour_avg_options[hour_averages_key]

    lag_options = {
        "7days": ((168,), False),
        "7day +ex": ((168,), True),
        "7days_1day": ((168, 24), False),
        "7day_1day +ex": ((168, 24), True),
    }
    lag_key = trial.suggest_categorical("features.lag_key", list(lag_options.keys()))
    cfg.features.endog_hour_lags, cfg.features.include_exog_lags = lag_options[lag_key]
    return cfg

_lstm_v4_F1_doc = SuggesterDoc(
    summary="Constrained LSTM v4 search space for series F1 (lean model, tune regularization & LR).",
    params=[
        # --- Model ---
        ParamFloat("model.dropout", 0.0, 0.20),  # tuned continuously

        # --- Training/Data ---
        ParamFloat("train.learning_rate", 2e-4, 2e-3, log=True),
        ParamCat("data.batch_size", [32, 64]),
        ParamCat("train.tf_drop_epochs", [10, 15]),  
    ],
    notes=[
        "`model.input_len` is fixed to 72.",
        "`model.num_layers` is fixed to 2.",
        "`model.head` and `model.hidden_size` are fixed to ('linear', 64).",
        "`norm.mode` is fixed to `global`.",
        "`train.n_epochs` is fixed to 35; early stopping may trim excess epochs.",
        "`features.include_exog_lags` is fixed to True.",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number "
        "(and `base_cfg.seed` if provided).",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("lstm_v4_F1", doc=_lstm_v4_F1_doc)
def suggest_config_v4_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = 2
    cfg.model.head        = "linear" 
    cfg.model.hidden_size = 64   

    cfg.model.dropout    = trial.suggest_float("model.dropout", 0.0, 0.20)

    # ---- Training/Norm/Data ----
    cfg.train.learning_rate = trial.suggest_float("train.learning_rate", 2e-4, 2e-3, log=True)
    cfg.train.n_epochs = 35
    cfg.norm.mode = "global"
    cfg.data.batch_size = trial.suggest_categorical("data.batch_size", [32, 64])
    cfg.train.tf_drop_epochs = trial.suggest_categorical("train.tf_drop_epochs", [10, 15])

    # ---- Features ----
    cfg.features.include_exog_lags = True

    return cfg

_lstm_const_F1_doc = SuggesterDoc(
    summary="Constant LSTM config for series F1 (only seed varies across trials).",
    params=[],
    notes=[
        "`model.input_len` is fixed to 72.",
        "`model.num_layers` is fixed to 2.",
        "`model.head` and `model.hidden_size` are fixed to ('linear', 64).",
        "`model.dropout` is fixed to 0.0.",
        "`train.learning_rate` is fixed to 5e-4.",
        "`train.n_epochs` is fixed to 35 (upper bound; early stopping picks best).",
        "`data.batch_size` is fixed to 32.",
        "`norm.mode` is fixed to `global`.",
        "`features.hour_averages` is fixed to ().",
        "`features.endog_hour_lags` is fixed to (168, 24).",
        "`features.include_exog_lags` is fixed to True.",
        "The only varying element across trials is the random seed.",
        "At each trial, the chosen cfg receives a seed derived from study name and trial number "
        "(and `base_cfg.seed` if provided).",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("lstm_const_F1", doc=_lstm_const_F1_doc)
def suggest_config_const_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Model (all fixed) ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = 2
    cfg.model.head        = "linear"
    cfg.model.hidden_size = 64
    cfg.model.dropout     = 0.0

    # ---- Training/Data (all fixed) ----
    cfg.train.learning_rate = 5e-4
    cfg.train.n_epochs      = 35          # upper bound; early stopping picks best
    cfg.norm.mode           = "global"
    cfg.data.batch_size     = 32

    # ---- Features (all fixed) ----
    cfg.features.endog_hour_lags   = (168, 24)
    cfg.features.include_exog_lags = True

    return cfg

#--------------------------------
# PRELIMINARY STUDY SUGGESTERS
# --------------------------------

preliminary_v2_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v2 (t vs thpw comparison).",
    params=[
        ParamCat("features.exog_vars_key", ["t", "thpw"]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, hidden_size=64, head='linear', "
        "dropout=0.1, lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on).",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v2", doc=preliminary_v2_doc)
def suggester_preliminary_v2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v2') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    # ---- Features ----
    exog_vars_key = trial.suggest_categorical("features.exog_vars_key", ["t", "thpw"])
    cfg.features.exog_vars = ("temperature",) if exog_vars_key == "t" else ("temperature", "humidity", "pressure", "wind_speed")
    trial.set_user_attr("repeat_idx", idx)

    return cfg

preliminary_v2_second_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v2 for F4 (t vs thpw comparison; es activated at epoch 12).",
    params=[
        ParamCat("features.exog_vars_key", ["t", "thpw"]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, hidden_size=64, head='linear', "
        "dropout=0.1, lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25, es_start_epoch=12s.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v2_second", doc=preliminary_v2_doc)
def suggester_preliminary_v2_second(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v2_second') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len  = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 12
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    # ---- Features ----
    exog_vars_key = trial.suggest_categorical("features.exog_vars_key", ["t", "thpw"])
    cfg.features.exog_vars = ("temperature",) if exog_vars_key == "t" else ("temperature", "humidity", "pressure", "wind_speed")
    trial.set_user_attr("repeat_idx", idx)

    return cfg

preliminary_v3_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v3 (long context vs short context + lags).",
    params=[
        ParamCat("model.input_len", [24, 72, 168, 504]),
        ParamCat("features.endog_hour_lags", [(), (168,), (168, 24)]),
        ParamCat("model.hidden_size", [64, 128]),
        ParamCat("repeat_idx", [0, 1, 2]),
    ],
    notes=[
        "`repeat_idx` repeats each config 3×.",
        "Other fixed params: input_len=72, num_layers=2, head='linear', dropout=0.1, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on).",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)
@register_suggester("preliminary_v3", doc=preliminary_v3_doc)
def suggester_preliminary_v3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v3') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model (fixed + constrained) ----
    cfg.model.input_len = trial.suggest_categorical("model.input_len", [24, 72, 168, 504])
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [64, 128])

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.endog_hour_lags = trial.suggest_categorical("features.endog_hour_lags", [(), (168,), (168, 24)])
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

preliminary_v4_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v4 (autoregressive vs non-autoregressive dec).",
    params=[
        ParamCat("model.use_ar_prev", [False, True]),
        ParamCat("repeat_idx", [0, 1, 2, 3]),
    ],
    notes=[
        "`repeat_idx` repeats each config 4×.",
        "Other fixed params: input_len=72, num_layers=2, head='linear', dropout=0.1, hidden_size=64, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on), es_start_epoch=12.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("preliminary_v4", doc=preliminary_v4_doc)
def suggester_preliminary_v4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v4') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1, 2, 3])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers = 2
    cfg.model.head       = "linear"
    cfg.model.hidden_size = 64
    cfg.model.use_ar_prev = trial.suggest_categorical("model.use_ar_prev", [False, True])

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 12
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

preliminary_v5_doc = SuggesterDoc(
    summary="LSTM preliminary sugg v5 (appropriate complexity study).",
    params=[
        ParamCat("model.hidden_size", [2, 4, 8, 12, 16, 32, 48, 64, 96, 128, 160, 196]),
        ParamCat("repeat_idx", [0, 1]),
    ],
    notes=[
        "`repeat_idx` repeats each config 2x.",
        "Other fixed params: input_len=72, num_layers=1, head='linear', dropout=0.1, hidden_size=64, "
        "lr=7e-4, batch_size=64, norm.mode='global', n_epochs=25 (early stopping on), es_start_epoch=6.",
        "At each trial, the chosen cfg receives a seed derived from config parameters and `repeat_idx`",
        "Other params use defaults from `base`."
    ],
)

@register_suggester("preliminary_v5", doc=preliminary_v5_doc)
def suggester_preliminary_v5(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('preliminary_v5') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", [0, 1])  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers = 1
    cfg.model.head       = "linear"
    cfg.model.hidden_size = trial.suggest_categorical("model.hidden_size", [2, 4, 8, 12, 16, 32, 48, 64, 96, 128, 160, 196])
    cfg.model.use_ar = "none"

    # Low dropout only
    cfg.model.dropout = 0.1

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate = 7e-4
    cfg.train.n_epochs = 25
    cfg.train.es_start_epoch = 7
    cfg.data.batch_size = 64
    cfg.norm.mode = "global"

    return cfg

final_v1_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v1 (F1)",
    params=[
        ParamInt("model.num_layers", 1, 4),
        ParamInt("model.hidden_size", 32, 128, step=32),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
        ParamFloat("train.weight_decay_pos", 1e-6, 1e-2, log=True, condition="train.use_weight_decay == True"),
        ParamInt("train.drop_epoch", 5, 8),
        ParamInt("data.batch_size", 32, 96, step=32),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', use_ar=24h, "
        "use_lr_drop=True (factor=0.3), norm.mode='global', "
        "n_epochs=25 with es (es_start_epoch=6), max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.lr_drop_at_epoch and train.tf_drop_epochs are both set to `train.drop_epoch`.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v1_F1", doc=final_v1_F1_doc)
def suggester_final_v1_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 4)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=32)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay_pos", 1e-6, 1e-2, log=True)
    )
    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 5, 8)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.data.batch_size      = trial.suggest_int("data.batch_size", 32, 96, step=32)
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v1_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v1 (F2)",
    params=[
        ParamInt("model.num_layers", 1, 3),
        ParamInt("model.hidden_size_exp", 4, 7),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
        ParamFloat("train.weight_decay_pos", 1e-6, 1e-2, log=True, condition="train.use_weight_decay == True"),
        ParamInt("train.drop_epoch", 5, 8),
        ParamInt("data.batch_size", 32, 96, step=32),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', use_ar=24h, "
        "use_lr_drop=True (factor=0.3), norm.mode='global', "
        "n_epochs=25 with es (es_start_epoch=5, es_rel_delta=0.5%), max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.lr_drop_at_epoch and train.tf_drop_epochs are both set to `train.drop_epoch`.",
        "model.hidden_size` is set to 2^`model.hidden_size_exp`.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v1_F2", doc=final_v1_F2_doc)
def suggester_final_v1_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 3)
    cfg.model.head        = "linear"
    hidden_size_exp = trial.suggest_int("model.hidden_size_exp", 4, 7)
    cfg.model.hidden_size = 2 ** hidden_size_exp
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay_pos", 1e-6, 1e-2, log=True)
    )
    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 5, 8)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta = 0.005
    cfg.data.batch_size      = trial.suggest_int("data.batch_size", 32, 96, step=32)
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F1)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 32, 128, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F1", doc=final_v2_F1_doc)
def suggester_final_v2_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_NAR_F1_doc = SuggesterDoc(
    summary="Final LSTM suggester v2, non-ar (F1)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 32, 128, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_NAR_F1", doc=final_v2_NAR_F1_doc)
def suggester_final_v2_NAR_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 32, 128, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F2)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 16, 112, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F2", doc=final_v2_F2_doc)
def suggester_final_v2_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 16, 112, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_NAR_F2_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (NAR F2)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 16, 112, step=16),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_NAR_F2", doc=final_v2_NAR_F2_doc)
def suggester_final_v2_NAR_F2(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 16, 112, step=16)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v2_F3_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F3)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 32, step=8),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
        ParamFloat("train.weight_decay", 1e-6, 1e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)


@register_suggester("final_v2_F3", doc=final_v2_F3_doc)
def suggester_final_v2_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=8)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay", 1e-6, 1e-2, log=True)
    )
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch 

    return cfg

final_v2_NAR_F3_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 non-autoregressive (F3)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 32, step=8),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
        ParamFloat("train.weight_decay", 1e-6, 1e-3, log=True),
        ParamCat("train.use_weight_decay", [False, True]),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=none, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_NAR_F3", doc=final_v2_NAR_F3_doc)
def suggester_final_v2_NAR_F3(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 32, step=8)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    use_wd = trial.suggest_categorical("train.use_weight_decay", [False, True])
    cfg.train.weight_decay = (
        0.0 if not use_wd else trial.suggest_float("train.weight_decay", 1e-6, 1e-2, log=True)
    )
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "none"
    cfg.train.tf_drop_epochs = drop_epoch 

    return cfg

final_v2_F4_doc = SuggesterDoc(
    summary="Final LSTM suggester v2 (F4)",
    params=[
        ParamInt("model.num_layers", 1, 2),
        ParamInt("model.hidden_size", 8, 56, step=8),
        ParamFloat("model.dropout", 0.0, 0.3),
        ParamFloat("train.learning_rate", 1e-4, 2e-3, log=True),
        ParamInt("train.drop_epoch", 4, 7, step=3),
    ],
    notes=[
        "Fixed params: input_len=72, head='linear', output_len=168 (7 days ahead), "
        "use_ar=24h, norm.mode='global', "
        "n_epochs=25 with early stopping (es_start_epoch=5, es_rel_delta=0.5%), "
        "use_lr_drop=True (factor=0.3, lr_drop_at_epoch=train.drop_epoch), "
        "max_walltime_sec=600 (10 minutes per trial).",
        "Fixed features: include_exog_lags=True.",
        "train.tf_drop_epochs is set to `train.drop_epoch`.",
        "data.batch_size fixed to 64.",
        "Other params inherit defaults from `base`.",
    ],
)

@register_suggester("final_v2_F4", doc=final_v2_F4_doc)
def suggester_final_v2_F4(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.train.max_walltime_sec = 600  # 10 minutes per trial max
    cfg.model.output_len = 168 # 7days ahead

    # ---- Model ----
    cfg.model.input_len   = 72
    cfg.model.num_layers  = trial.suggest_int("model.num_layers", 1, 2)
    cfg.model.head        = "linear"
    cfg.model.hidden_size = trial.suggest_int("model.hidden_size", 8, 56, step=8)
    cfg.model.dropout     = trial.suggest_float("model.dropout", 0.0, 0.3)

    # ---- Features ----
    cfg.features.include_exog_lags = True

    # ---- Training/Data ----
    cfg.train.learning_rate  = trial.suggest_float("train.learning_rate", 1e-4, 2e-3, log=True)
    cfg.train.grad_clip_max_norm = 10.0 # (default)

    cfg.train.use_lr_drop    = True 
    drop_epoch   = trial.suggest_int("train.drop_epoch", 4, 7, step=3)
    cfg.train.lr_drop_at_epoch = drop_epoch
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.005
    cfg.data.batch_size      = 64
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"
    cfg.train.tf_drop_epochs = drop_epoch

    return cfg

final_v3_F1_doc = SuggesterDoc(
    summary="LSTM final sugg v3 (final 3 candidate configurations with repeats).",
    params=[
        ParamCat("config", ["best_anb", "candidate_1", "candidate_2"]),
        ParamCat("repeat_idx", list(range(10))),
    ],
    notes=[
        "`repeat_idx` repeats each config 10×.",
        "Configuration 'best_anb' achieved the best `avg_near_best` (trial #34).",
        "Other configurations are candidates derived from the tuning analysis.",
        "`es_rel_delta` is set to 0.0 to milden early stopping.",
    ],
)

@register_suggester("final_v3_F1", doc=final_v3_F1_doc)
def suggester_final_v3_F1(trial: optuna.trial.Trial, base: LSTMRunConfig) -> LSTMRunConfig:
    cfg = copy.deepcopy(base)
    cfg.model.output_len = 168 # 7days ahead

    # ---- Sanity check ----
    if not isinstance(trial.study.sampler, optuna.samplers.GridSampler):
        raise ValueError("This suggester ('final_v3_F1') is designed for grid search (GridSampler). Load the study with the appropriate sampler.")

    # ---- Repeat index ----
    idx = trial.suggest_categorical("repeat_idx", list(range(10)))  # dummy to repeat trials
    trial.set_user_attr("repeat_idx", idx)

    # ---- Config ----
    config = trial.suggest_categorical("config", ["best_anb", "candidate_1", "candidate_2"])
    trial.set_user_attr("config", config)
    cfg.data.batch_size = 64
    cfg.model.input_len   = 72
    cfg.model.head        = "linear"
    cfg.features.include_exog_lags = True
    cfg.train.grad_clip_max_norm = 10.0 # (default)
    cfg.train.use_lr_drop    = True
    cfg.train.lr_drop_factor   = 0.3
    cfg.train.n_epochs       = 25
    cfg.train.es_start_epoch = 5
    cfg.train.es_rel_delta   = 0.0
    cfg.norm.mode            = "global"
    cfg.model.use_ar         = "24h"

    if config == "best_anb":
        cfg.model.dropout = 0.016
        cfg.model.hidden_size = 64
        cfg.model.num_layers = 2
        cfg.train.lr_drop_at_epoch = 6
        cfg.train.tf_drop_epochs = 6
        cfg.train.learning_rate = 5.69e-04
    else:
        cfg.model.dropout = 0.0
        cfg.model.hidden_size = 64 if config == "candidate_1" else 80
        cfg.model.num_layers = 1
        cfg.train.lr_drop_at_epoch = 4
        cfg.train.tf_drop_epochs = 4
        cfg.train.learning_rate = 7e-04

    return cfg


