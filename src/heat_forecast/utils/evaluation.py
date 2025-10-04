from functools import partial
from typing import Sequence, List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, mape, mase


# -------------------------------------------------------------------------------
# FUNCTIONS FOR EVALUATING FORECASTS AND DISPLAY / PLOT EVALUATION RESULTS
# -------------------------------------------------------------------------------

def me(
    df: pd.DataFrame,
    models: Sequence[str],
    id_col: str = "unique_id",
    target_col: str = "y",
):
    """
    Mean Error (ME) — average signed error per series.

    Positive ME ⇒ the model *over-forecasts* on average  
    Negative ME ⇒ the model *under-forecasts* on average

    The function returns one row per series (``id_col`` value) and one
    column per model in *models*, plus the constant column ``'metric'``
    set to ``'me'`` so the output is compatible with the wider
    ``evaluate`` workflow.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that contains the ground-truth column *target_col*,
        the identifier column *id_col*, and one forecast column for each
        entry in *models*.

    models : Sequence[str]
        List or tuple of column names holding the model forecasts whose
        ME should be computed.

    id_col : str, default ``"unique_id"``
        Name of the column that uniquely identifies each series.

    target_col : str, default ``"y"``
        Name of the ground-truth column.

    Returns
    -------
    wide : pandas.DataFrame
        Wide-format table with columns::

            [id_col, "metric", *models]

        One row per series; each model column contains that model's ME.

    Examples
    --------
    >>> wide_me = me(cv_slice, models=["model_a", "model_b"])
    >>> wide_me.head()
       unique_id metric  model_a  model_b
    0         S1     me   0.1234  -0.0150
    """
    # Compute errors for every model at once
    err_block = pd.DataFrame(
        {m: df[m].to_numpy() - df[target_col].to_numpy() for m in models},
        index=df.index,
    )
    err_df = pd.concat([df[[id_col]], err_block], axis=1)

    # Average over the time dimension
    wide = (
        err_df
        .groupby(id_col, as_index=False)
        .mean(numeric_only=True)          # mean error per series & model
        .assign(metric="me")              # required by evaluate
        [[id_col, "metric", *models]]     # enforce column order
    )
    return wide

def nmae(
    df: pd.DataFrame,
    models: Sequence[str],
    id_col: str = "unique_id",
    target_col: str = "y",
):
    r"""
    Normalised Mean Absolute Error (NMAE) **per series**.

    For each series (identified by ``id_col``) the MAE of every model is
    divided by that series' own normalisation constant:

    ```       
        NMAE_{i,m} = mean_t(|ŷ_{i,t}^{(m)} - y_{i,t}|)
                     ─────────────────────────────────
                            mean_t(|y_{i,t}|)
    ```

    where *i* indexes the series and *m* the model.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain
        * the ground-truth column *target_col*,
        * the identifier column *id_col*, and
        * one forecast column for each model in *models*.

    models : Sequence[str]
        Names of the forecast columns to evaluate.

    id_col : str, default ``"unique_id"``
        Column identifying each series.

    target_col : str, default ``"y"``
        Ground-truth column.

    Returns
    -------
    wide : pandas.DataFrame
        Shape: *(n_series, 2 + n_models)* with columns::

            [id_col, "metric", *models]

        Every model column holds that model's **per-series NMAE**.
        ``"metric"`` is always the string ``"nmae"``.
    """
    # Compute absolute errors for every model at once
    err_block = pd.DataFrame(
        {m: np.abs(df[m].to_numpy() - df[target_col].to_numpy())
         for m in models},
        index=df.index,
    )
    err_df = pd.concat([df[[id_col]], err_block], axis=1)

    # Compute average mae and average |y| per series
    mae_per_series = (
        err_df
        .groupby(id_col, as_index=False)
        .mean(numeric_only=True)            # per-series MAE
    )
    abs_y_mean = (
        df.groupby(id_col, as_index=False)[target_col]
        .apply(lambda s: np.abs(s).mean())  # per-series ⟨|y|⟩
        .rename(columns={target_col: "norm"})
    )

    # Merge and normalise
    wide = (
        mae_per_series
        .merge(abs_y_mean, on=id_col)
        .assign(**{m: lambda d, m=m: d[m] / d["norm"] for m in models})
        .drop(columns="norm")
        .assign(metric="nmae")
        [[id_col, "metric", *models]]
)
    return wide

def compute_pct_increase(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each model's percentage difference from the *Naive24h* baseline.

    The function reshapes *evaluation_df* to long form, calculates  
    ``(model - Naive24h) / Naive24h * 100`` for every non-baseline model,
    appends those percentage rows (with ``'_pct_inc'`` added to *metric*)
    and pivots back to a wide layout.

    The only metric that is not transformed is ``'me'``; it is
    excluded from the output.

    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        Must contain
        * ``'unique_id'`` - series identifier  
        * ``'metric'``    - name of the accuracy metric (e.g. 'MAE')  
        * ``'Naive24h'``  - baseline values  
        * one column per additional model

        Example structure::

            unique_id  metric    Naive24h     ARIMA      ETS
            F1         MAE       12.3         10.1       11.0
            F1         RMSE      20.5         18.7       19.2
            …

    Returns
    -------
    pandas.DataFrame
        Wide-format frame with the original metrics plus new rows where
        *metric* is suffixed by ``'_pct_inc'`` and each model column holds
        the percentage increase (baseline rows use ``0`` for convenience).

        Example output::

            unique_id   metric        Naive24h   ARIMA   ETS
            F1          MAE               12.3   10.1   11.0
            F1          MAE_pct_inc        0.0  -17.9  -10.6
            F1          RMSE              20.5   18.7   19.2
            F1          RMSE_pct_inc       0.0   -8.7   -6.3
            …
    """
    # Melt to long form
    long = evaluation_df.melt(
        id_vars=['unique_id','metric'],
        var_name='model',
        value_name='value'
    )

    # Pull out the Naive24h baseline
    baseline = (
        long[long['model']=='Naive24h']
          .rename(columns={'value':'naive'})
          .loc[:, ['unique_id','metric','naive']]
    )

    # Merge baseline back onto all rows
    merged = long.merge(baseline, on=['unique_id','metric'])

    # Compute pct-inc only for non-Naive
    pct = merged[merged['model']!='Naive24h'].copy()
    pct['value'] = (pct['value'] - pct['naive']) / pct['naive'] * 100
    pct['metric'] = pct['metric'] + '_pct_inc'

    # Stack originals + pct rows
    all_long = pd.concat([long, pct[['unique_id','metric','model','value']]], ignore_index=True)

    # One final pivot back to wide
    wide = (
        all_long
        .pivot_table(
            index=['unique_id','metric'],
            columns='model',
            values='value',
            fill_value=0 # fill with 0s for missing values (corresponding to the pct inc of the Naive24h model itself)
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(['unique_id','metric'])
        .reset_index(drop=True)
    )

    # Drop me_pct_inc rows if they exist
    wide = wide[~(wide['metric']=='me_pct_inc')].copy()

    return wide

def custom_evaluate(
    forecast_df: pd.DataFrame,  
    target_df: pd.DataFrame,  
    insample_size: Optional[int] = None,  
    metrics: Optional[List[str]] = None,  
    with_naive: bool = True,  
    with_pct_increase: bool = False,  # Whether to compute percentage increase compared to Naive24h
) -> pd.DataFrame:
    """
    Evaluate forecasts against the ground truth and return a
    tidy table of accuracy metrics.

    Workflow
    --------
    1. Merge *forecast_df* with the true targets from *target_df*.
    2. Optionally fit a 24-hour seasonal naive benchmark (alias
       'Naive24h') on the training portion that precedes the forecast
       horizon and append its predictions.
    3. Call *statsforecast.evaluate* with the requested metrics.
    4. Optionally add percentage-difference rows that show each model's
       change relative to the Naive24h baseline.

    Parameters
    ----------
    forecast_df : pandas.DataFrame
        Model output with columns
        * 'unique_id' - series identifier  
        * 'ds'        - forecast timestamp  
        * one column per model containing point forecasts
    target_df : pandas.DataFrame
        Complete history of the target variable.  Must include
        'unique_id', 'ds', 'y'.
    insample_size : int, optional
        Limit the Naive24h training window to the last *insample_size*
        observations per series when computing MASE.  Ignored when
        *with_naive* is False and when 'mase' is not among *metrics*.
    metrics : list[str], optional
        Accuracy measures to compute.  Choices are
        ['mae', 'rmse', 'mase', 'nmae', 'mape', 'me'].
        Defaults to ['mae', 'rmse', 'mase', 'nmae'].
    with_naive : bool, default True
        Fit and evaluate a 24-hour seasonal naive benchmark unless it is
        already present in *forecast_df*.
    with_pct_increase : bool, default False
        Append extra rows where *metric* is suffixed by '_pct_inc' and the
        values are percentage differences from Naive24h.  Requires
        *with_naive* to be True.

    Returns
    -------
    pandas.DataFrame
        Wide-format table with columns

        ['unique_id', 'metric', <model1>, <model2>, …].

        When *with_pct_increase* is True, every original metric gains an
        additional '_pct_inc' counterpart.
    """
    available_metrics = ['mae', 'rmse', 'mase', 'nmae', 'mape', 'me']
    if metrics is not None:
        if not set(metrics).issubset(set(available_metrics)):
            raise ValueError(f"metrics must be a subset of {available_metrics}.")
    else:
        metrics = ['mae', 'rmse', 'mase', 'nmae']

    str_to_func = {
        'mae': mae,
        'rmse': rmse,
        'mase': partial(mase, seasonality=24),  # Assuming a 24-hour seasonality for the naive model
        'nmae': nmae,
        'mape': mape,
        'me': me
    }
    metrics_func = [str_to_func[key] for key in metrics if key in str_to_func]

    # Add the true values to the forecast DataFrame for evaluation
    forecast_and_val_df = forecast_df.merge(
        target_df[['unique_id', 'ds', 'y']],
        on=['unique_id', 'ds'],
        how='left'
    )

    if 'mase' in metrics or (with_naive and 'Naive24h' not in forecast_and_val_df.columns):
        start_test = forecast_df['ds'].min()
        heat_train_df = target_df[target_df['ds'] < start_test].copy()
        if insample_size is not None:
            # If insample_size is provided, limit the training data to the last `insample_size` hours
            heat_train_df = heat_train_df.groupby('unique_id').apply(
                lambda x: x.tail(insample_size)
            ).reset_index(drop=True)

    if with_naive and 'Naive24h' not in forecast_and_val_df.columns:
        h = forecast_df['ds'].nunique()  # Number of hours to forecast
        
        # Compute forecasts using the naive method
        naive_model24 = SeasonalNaive(season_length=24, alias='Naive24h')
        naive_forecast_df = StatsForecast(
            models=[naive_model24], 
            freq='h'
        ).forecast(h, heat_train_df)

        # Merge the forecasts into a single DataFrame
        forecast_and_val_df = (
            forecast_and_val_df
            .merge(naive_forecast_df, on=['unique_id', 'ds'], how='left')
        )

    # Evaluate
    if 'mase' in metrics: 
        evaluation_df = evaluate(df=forecast_and_val_df, metrics=metrics_func, train_df=heat_train_df)
    else:
        evaluation_df = evaluate(df=forecast_and_val_df, metrics=metrics_func)
    
    if with_pct_increase:
        if not with_naive:
            raise ValueError("with_naive must be True to compute percentage increase compared to Naive24h.")
        # Compute percentage increase compared to Naive24h
        evaluation_df = compute_pct_increase(evaluation_df)

    return evaluation_df

def custom_evaluate_cv(
    cv_df: pd.DataFrame,  # DataFrame with the forecasts of the models, with columns 'unique_id', 'ds', 'cutoff', and forecast columns
    metrics: Optional[List[str]] = None,  # List of metrics to compute
    target_df: Optional[pd.DataFrame] = None,  # Full training data with 'ds', 'unique_id', and 'y' columns, for mase if requested
    target_col: str = "y",  # Column name in `target_df` for the true values
) -> pd.DataFrame:
    """
    Evaluate multi-model, multi-window cross-validation results and
    return both per-window scores and an aggregated summary.

    Parameters
    ----------
    cv_df : pandas.DataFrame
        DataFrame that holds the forecasts of every model for every
        back-testing window.  It **must** contain the columns:

        * ``'unique_id'`` - series identifier  
        * ``'ds'``        - timestamps of the forecasts  
        * ``'cutoff'``    - training-window end date  
        * one column per forecast model (e.g., ``'model_a'``, ``'model_b'`` …)

    metrics : list of str, optional
        Names of the metrics to compute. Allowed values are
        ``['mae', 'rmse', 'mase', 'nmae', 'mape', 'me']``.
        Default is ``['mae', 'rmse', 'mase', 'nmae']``.
        Passing an empty list raises ``ValueError``.

    target_df : pandas.DataFrame, optional
        Full training data with columns ``'ds'``, ``'unique_id'`` and the
        ground-truth column specified by ``target_col``.  Required **only**
        when ``'mase'`` is in ``metrics``; otherwise it is ignored.

    target_col : str, default ``"y"``
        Name of the ground-truth column in ``target_df`` and the column that
        ``evaluate`` uses to compare forecasts against.

    Returns
    -------
    summary : pandas.DataFrame
        Aggregated metrics: one row per ``unique_id`` and metric, with
        two columns for every model - ``'<model>_mean'`` and
        ``'<model>_std'`` - summarising performance across cutoffs.
    
    all_results : pandas.DataFrame
        Window-level metrics: one row per ``unique_id`` x metric x cutoff,
        with a column for every model's score in that window.
    """
    uids = cv_df['unique_id'].unique()

    # select only the sieries that are in cv_df
    target_df = target_df[target_df['unique_id'].isin(uids)].copy() if target_df is not None else None

    available = ['mae', 'rmse', 'mase', 'nmae', 'mape', 'me']
    if metrics is None:
        metrics = ['mae', 'rmse', 'mase', 'nmae']
    else:
        if not metrics:                         # forbids empty list
            raise ValueError("metrics list may not be empty.")
        bad = set(metrics) - set(available)
        if bad:
            raise ValueError(
                f"Unknown metric(s): {sorted(bad)}. "
                f"Choose from {available}."
            )
    
    if 'mase' in metrics and target_df is None:
        raise ValueError("`target_df` is required when 'mase' is requested.")

    str_to_func = {
        'mae': mae,
        'rmse': rmse,
        'mase': partial(mase, seasonality=24),  # Assuming a 24-hour seasonality for the naive model
        'nmae': nmae,
        'mape': mape,
        'me': me
    }
    metrics_func = [str_to_func[key] for key in metrics if key in str_to_func]
    
    cv_df = cv_df.copy()
    all_res = []

    for cutoff in sorted(cv_df['cutoff'].unique()):
        window_df = cv_df[cv_df['cutoff'] == cutoff].drop(columns='cutoff')

        kw = dict(df=window_df, metrics=metrics_func)
        if 'mase' in metrics:                          # needs training slice
            kw['train_df'] = target_df[target_df['ds'] <= cutoff]

        eval_df = evaluate(**kw, target_col=target_col).copy()       # one call per window
        eval_df = pd.concat([eval_df, pd.DataFrame({'cutoff': [cutoff] * len(eval_df)})], axis=1)
        all_res.append(eval_df)

    all_results = pd.concat(all_res, ignore_index=True).copy()

    # Get model columns (exclude 'unique_id', 'metric', 'cutoff')
    model_cols = [c for c in all_results.columns
              if c not in ("unique_id", "metric", "cutoff")]
    summary = (
        all_results
        .groupby(["unique_id", "metric"])[model_cols]    # ← keep cutoff out
        .agg(["mean", "std"])
        .pipe(lambda df: df.set_axis(
            [f"{col}_{stat}" for col, stat in df.columns], axis=1))
        .reset_index()
    )

    return summary.copy(), all_results.copy()

def display_metrics(
    evaluation_df: pd.DataFrame,  # DataFrame with evaluation metrics
):
    """
    Render an interactive table that summarises model-evaluation results.

    The input must be the wide table returned by ``custom_evaluate`` (or
    of identical structure).  The function reshapes the data so that each
    metric becomes a column and each model a row, adds an overall
    "Average" row when more than one ``unique_id`` is present, and uses
    ``pandas.io.formats.style.Styler`` to apply visual cues:
    * absolute-error metrics (mae, rmse, mape, mase, nmae) receive centred
      bar charts;
    * percentage-difference metrics (``'<metric>_pct_inc'``) are coloured
      on a green—white—red diverging scale where green means the model
      beats the Naive24h baseline and red means it under-performs.

    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        Must contain the columns

        * ``'unique_id'`` - identifier of the time series
        * ``'metric'``    - name of the error measure
        * one column per model being compared

    Returns
    -------
    None
        The styled table is displayed as a side effect.
    """
    long = evaluation_df.melt(
        id_vars=['unique_id','metric'],
        var_name='model',
        value_name='value'
    )
    wide = long.pivot(
        index=['unique_id','model'],
        columns=['metric'],
        values='value'
    ).sort_index(axis=0, level=[0,1])
    wide.index.names = ['facility','model']
    wide.columns.names = ['metric']

    # Add average rows
    if evaluation_df['unique_id'].nunique() > 1:
        model_means = wide.groupby(level='model').mean()
        model_means.index = pd.MultiIndex.from_product(
            [['Average'], model_means.index],
            names=wide.index.names
        )
        combined = pd.concat([wide, model_means])
    else:
        combined = wide

    # Define the custom green—white—red colormap
    cmap = LinearSegmentedColormap.from_list(
        "GreenWhiteRed", 
        ["green", "lightgray", "red"]
    )

    bar_cols  = ['mae', 'rmse', 'mape', 'mase', 'nmae']           # bars
    grad_cols = [c + '_pct_inc' for c in bar_cols]                # add _pct_inc suffix

    grad_cols_actual = [c for c in grad_cols if c in combined.columns]
    bar_cols_actual  = [c for c in bar_cols if c in combined.columns]

    styled = combined.style

    for col in grad_cols_actual:
        rng   = combined[col].abs().max()
        styled = styled.background_gradient(
            cmap=cmap,
            subset=[col],        # ← single column
            vmin=-rng, vmax=rng  # ← scalar, so no error
        )

    styled = (
        styled
        .set_properties(subset=grad_cols_actual, **{'color': 'white'})
        .bar(subset=bar_cols_actual, align='mid')
        .format("{:.2f}")
    )

    display(styled)

def display_cv_summary(
    summary: pd.DataFrame,
    sort_metric: Optional[str] = None,   # e.g. "rmse", "mae", …
    sort_stat: str = "mean",             # "mean" or "std"
    ascending: bool = True,
    by_panel: bool = False,              # True → sort within each unique_id
    show_row_numbers: bool = False,      # True → add a “#” column (0-based)
) -> pd.DataFrame:
    """
    Pretty-print cross-validation metrics (wide table) with optional sorting
    and an optional plain integer row index.

    Parameters
    ----------
    summary : DataFrame
        Output of `custom_evaluate_cv` (wide columns: "<model>_{mean,std}").
    sort_metric : str | None
        Metric to sort by (e.g. "rmse").  Pass None to keep original order.
    sort_stat : {"mean", "std"}
        Which statistic to use when sorting. Default is "mean".
    ascending : bool
        Usual pandas sort order. Default is True (ascending).
    by_panel : bool
        If True, each `unique_id` block is sorted independently;
        otherwise models are sorted across the whole table. Default is False.
    show_row_numbers : bool
        If True, insert a "#" column (0 … n-1) at the left of the table.
        Default is False.

    Returns
    -------
    DataFrame
        The wide table that was displayed (with or without the "#" column).
    """
    # Reshape to a nicer (unique_id, model) × (metric, stat) cube 
    long = (
        summary
        .melt(id_vars=["unique_id", "metric"],
              var_name="tmp", value_name="value")
        .assign(
            model=lambda d: d["tmp"].str.rsplit("_", n=1).str[0],
            stat=lambda d: d["tmp"].str.rsplit("_", n=1).str[1],
        )
        .drop(columns="tmp")
    )

    wide = (
        long
        .pivot(index=["unique_id", "model"],
               columns=["metric", "stat"],
               values="value")
        .sort_index(axis=0)                # rows: unique_id > model
        .sort_index(axis=1, level=[0, 1])  # cols: metric > stat
    )
    wide.index.names   = ["unique_id", "model"]
    wide.columns.names = ["metric", "stat"]

    # Optional sorting 
    if sort_metric is not None:
        sort_key = (sort_metric, sort_stat)
        if sort_key not in wide.columns:
            raise ValueError(
                f"{sort_metric!r} with stat {sort_stat!r} not present in summary."
            )

        if by_panel:
            wide = (
                wide
                .groupby(level="unique_id", group_keys=False)
                .apply(lambda df: df.sort_values(sort_key, ascending=ascending))
            )
        else:
            wide = wide.sort_values(sort_key, ascending=ascending)

    # Optional "#" column 
    if show_row_numbers:
        wide_disp = wide.copy()
        wide_disp.insert(0, "#", range(len(wide_disp)))
    else:
        wide_disp = wide

    # Styler (skip the "#" column when colouring) 
    metric_cols = [c for c in wide_disp.columns if isinstance(c, tuple)]
    mean_cols   = [c for c in metric_cols if c[1] == "mean"]
    std_cols    = [c for c in metric_cols if c[1] == "std"]

    # pick columns to represent as percentages
    perc_cols = [c for c in metric_cols if c[0] == "nmae"]
    other_cols = [c for c in metric_cols if c not in perc_cols]

    styler = (
        wide_disp.style
            .bar(color="lightcoral", subset=mean_cols, align="mid")
            .background_gradient(cmap="Blues", subset=std_cols)
            .format("{:.2%}", subset=perc_cols)
            .format(precision=2, subset=other_cols)
    )
    display(styler)

    return wide_disp

def barplot_cv(df):
    """
    Draw a grid of bar charts that compares model performance across
    series and error metrics.

    For each combination of ``unique_id`` (row) and ``metric`` (column),
    the function plots the distribution of model scores observed at
    different back-test cut-off dates.  Bars show the mean value and the
    error bars represent an 80 percent empirical interval
    (`errorbar=('pi', 80)` in seaborn).

    Expected input structure
    ------------------------
    The data frame *df* must contain

    * ``'unique_id'`` - identifier of the time-series
    * ``'cutoff'``    - forecast origin used to aggregate the scores
    * ``'metric'``    - name of the error measure (e.g. 'MAE', 'RMSE')
    * one column per model holding the numeric metric value

    Example::

        unique_id   cutoff        metric   ARIMA   ETS   Naive24h
        F1          2024-11-01    MAE      10.2    11.0  12.4
        F1          2024-11-01    RMSE     18.5    19.3  20.8
        F1          2024-11-08    MAE       9.8    10.7  12.0
        …

    Parameters
    ----------
    df : pandas.DataFrame
        Cross-validation results in the format described above.

    Returns
    -------
    None
        The figure is displayed via ``plt.show()`` as a side effect.
    """
    long = df.melt(
        id_vars=['unique_id', 'cutoff', 'metric'],
        var_name='model', value_name='value'
    )

    ids     = long['unique_id'].unique()
    metrics = long['metric'].unique()

    nrows, ncols = len(ids), len(metrics)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3 * ncols, 3 * nrows),
        sharey=False
    )

    # axes is 2-D even if nrows or ncols == 1
    axes = np.atleast_2d(axes)

    for i, uid in enumerate(ids):
        for j, m in enumerate(metrics):
            ax  = axes[i, j]
            sub = long.query("unique_id == @uid and metric == @m")

            sns.barplot(
                data=sub,
                x='model', y='value',
                errorbar=('pi', 80),
                ax=ax, dodge=True,
                color="#3A78EA",
            )

            if i == 0: # top row: column titles
                ax.set_title(m, weight='bold')
            if j == 0: # first column: row labels
                ax.set_ylabel(uid)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("") # model names already on ticks

    fig.suptitle("Mean performance ± 80 % interval per series / metric",
                 y=1.02)
    fig.tight_layout()
    plt.show()

def plot_cv_metric_by_cutoff(
        combined_results: pd.DataFrame,
        metric: str = "mae",
        figsize: Optional[tuple[int, int]] = None,
        models: Optional[list[str]] = None,
    ) -> plt.Figure:
    """
    Plot a grid of bar charts that visualise cross-validation scores by
    cut-off date and model.

    For every time-series (`unique_id`) a separate subplot is created.
    Inside each subplot, bars display the chosen error metric for each
    model at every back-test cut-off date.

    Parameters
    ----------
    combined_results : pandas.DataFrame
        Cross-validation results.  Must include the columns
        * ``'unique_id'`` (str) - series identifier  
        * ``'metric'``    (str) - name of the metric  
        * ``'cutoff'``    (datetime64) - forecast origin  
        * one column per model (numeric scores)
    metric : str, default ``"mae"``
        Metric to plot.  Rows with other metrics are filtered out before
        plotting.
    figsize : tuple[int, int], default ``None``
        Size of the entire figure in inches *(width, height)*.
    models : list[str] | None, default ``None``
        Subset of model columns to display.  If ``None``, all model
        columns found in *combined_results* are used.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure object.
    """
    
    # Filter for metric only
    metric_df = combined_results[combined_results['metric'] == metric].copy()
    
    # Get unique facilities and models
    facilities = sorted(metric_df['unique_id'].unique())
    
    # Get model columns (exclude metadata columns)
    model_cols = [col for col in metric_df.columns 
                  if col not in ['unique_id', 'metric', 'cutoff']]
    if models is not None:
        if not isinstance(models, list):
            raise ValueError("`models` should be None or a list of model names.")
        for m in models:
            if not isinstance(m, str):
                raise ValueError(f"Model name '{m}' should be a string.")
            if m not in model_cols:
                raise ValueError(f"Model '{m}' not found in the results. Available models: {model_cols}")
        model_cols = models
    
    # Create subplots - one for each facility
    fig, axes = plt.subplots(nrows=len(facilities), ncols=1, figsize=(15, len(facilities) * 7) if figsize is None else figsize,
                             sharex=True)
    axes = np.atleast_1d(axes)
    
    # Color palette for models
    colors = sns.color_palette("tab10", len(model_cols))
    
    for i, facility in enumerate(facilities):
        ax = axes[i]
        
        # Filter data for this facility
        facility_data = metric_df[metric_df['unique_id'] == facility].copy()
        
        # Sort by cutoff for proper ordering
        facility_data = facility_data.sort_values('cutoff')
        
        # Convert cutoff to string for better x-axis labels
        facility_data['cutoff_str'] = facility_data['cutoff'].dt.strftime('%Y-%m-%d')
        
        # Create bar positions
        x_positions = range(len(facility_data))
        bar_width = 0.8 / len(model_cols)

        # Set y-axis limits
        bottom = min(facility_data[model_cols].min().min(), 0)
        y_max = facility_data[model_cols].max().max()
        margin = 0.05 * y_max            # 5 % head-room
        ax.set_ylim(bottom, y_max + margin)   # text now lives inside the axes

        # Offset for test on bars
        offset = 0.01 * y_max
        
        # Plot bars for each model
        for j, model in enumerate(model_cols):
            x_pos = [x + bar_width * (j - len(model_cols)/2 + 0.5) for x in x_positions]
            bars = ax.bar(x_pos, facility_data[model], 
                         width=bar_width, 
                         label=model,
                         color=colors[j],
                         alpha=0.8)
            
            # Add value labels on bars 
            for bar, val in zip(bars, facility_data[model]):
                if not pd.isna(val):
                    #ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    #       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
                    ax.text(bar.get_x() + bar.get_width()/2,
                            val + offset,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=8)
        
        # Customize subplot
        ax.set_title(f'Series {facility}')
        ax.set_xlabel('Cutoff Date')
        ax.set_ylabel(f'{metric}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(facility_data['cutoff_str'], rotation=45, ha='right')

    # build a single legend for the whole figure
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, lab in zip(h, l):
            if lab not in labels:
                handles.append(hi)
                labels.append(lab)

    # place the legend above all subplots
    fig.legend(handles, labels,
               loc='upper right',
               ncol=1,
               bbox_to_anchor=(1, 0.97))

    fig.suptitle(f'Cross-Validation {metric} Results by Cutoff and Model', y=0.98, fontsize=18, fontweight='bold')     
    fig.tight_layout(rect=[0, 0, 1, 0.97])                     
    
    return fig

def compute_loss_diff_stats(
        combined_results: pd.DataFrame,
        baseline_model: str = 'Naive24h'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute and summarise loss-difference (LD) statistics versus a
    baseline model.

    For every model column in *combined_results* (except the baseline),
    the function subtracts the baseline's losses to create new columns
    named ``"LD-<model>"``.  
    It returns:

    1. **summary_ld** - per-series/per-metric mean and standard deviation
       of each LD column.
    2. **combined_ld** - the full table containing the original metadata
       columns (``'unique_id'``, ``'metric'``, ``'cutoff'``) plus every
       LD column for all windows.

    Parameters
    ----------
    combined_results : pandas.DataFrame
        Wide CV-evaluation table produced by your earlier pipeline.  It
        must contain:
            * ``'unique_id'`` - series identifier  
            * ``'metric'``    - the metric name (e.g. ``'mae'``)  
            * ``'cutoff'``    - window end date  
            * one numeric column per forecast model

    baseline_model : str, default ``"Naive24h"``
        Name of the model column to use as the reference whose loss is
        subtracted from every other model's loss.

    Returns
    -------
    summary_ld : pandas.DataFrame
        One row per ``unique_id`` x ``metric``; columns::

            [ 'unique_id', 'metric',
              'LD-<model1>_mean', 'LD-<model1>_std',
              'LD-<model2>_mean', … ]

    combined_ld : pandas.DataFrame
        Same shape as *combined_results* plus **all** LD columns, one
        row per original window.

    Raises
    ------
    ValueError
        * If *baseline_model* is missing from *combined_results*.  
        * If no other model columns are present to compare against.

    Notes
    -----
    * Positive LD ⇒ the candidate model performs **worse** than the
      baseline (larger loss).  
    * Negative LD ⇒ the candidate model outperforms the baseline.  
    * The aggregation step uses ``mean`` and ``std`` with
      ``numeric_only=True`` for safety.

    Examples
    --------
    >>> summary, ld_full = display_loss_diff_stats(results_df, baseline_model="Stat")
    >>> summary.head()
       unique_id metric  LD-MSTL_mean  LD-MSTL_std  LD-SARIMAX_mean  LD-SARIMAX_std
    0        S1    mae       -0.021       0.004        0.019         0.007
    """

    # Check if the baseline model exists in the results
    if baseline_model not in combined_results.columns:
        raise ValueError(
            f"baseline_model '{baseline_model}' not found in results."
        )
    # Get model columns (exclude 'unique_id', 'metric', 'cutoff')
    meta_cols  = {"unique_id", "metric", "cutoff"}
    models     = [c for c in combined_results.columns if c not in meta_cols]
    other_mdl  = [m for m in models if m != baseline_model]
    if not other_mdl: raise ValueError("No models other than the baseline—nothing to compare.")

    # Compute loss diff
    ld_block = (
        combined_results[other_mdl]
        .sub(combined_results[baseline_model], axis=0)   
        .add_prefix("LD-")                              
    )
    combined_ld = pd.concat(
        [combined_results[["unique_id", "metric", "cutoff"]], ld_block],
        axis=1,
    )

    # Summarise
    agg = (
        combined_ld
        .groupby(["unique_id", "metric"])[ld_block.columns] 
        .agg(["mean", "std"])
    )
    agg.columns = [f"{mdl}_{stat}" for mdl, stat in agg.columns]
    summary_ld = agg.reset_index() # bring the keys back as columns
    summary_ld = summary_ld[["unique_id","metric",*agg.columns]]  # reorder columns

    return summary_ld, combined_ld

def adj_r2_score(y, y_hat, T, k):
    """
    Compute the adjusted R-squared score.

    Adjusted R² accounts for the number of predictors in the model and 
    penalizes excessive use of non-informative features.

    Parameters
    ----------
    y : array-like
        True target values.
    y_hat : array-like
        Predicted target values.
    T : int
        Number of observations (sample size).
    k : int
        Number of explanatory variables (not including intercept).

    Returns
    -------
    adj_r2 : float
        Adjusted R-squared score.
    """
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - k - 1)
    return adj_r2

def aic_score(y, y_hat, T, k):
    """
    Compute the Akaike Information Criterion (AIC) score.

    AIC is a model selection metric that balances goodness of fit and model complexity.

    Parameters
    ----------
    y : array-like
        True target values.
    y_hat : array-like
        Predicted target values.
    T : int
        Number of observations (sample size).
    k : int
        Number of explanatory variables (not including intercept).

    Returns
    -------
    aic : float
        Akaike Information Criterion value.
    """
    sse = np.sum((y - y_hat) ** 2)
    aic = T * np.log(sse / T) + 2 * (k + 2)
    return aic

def aicc_score(y, y_hat, T, k):
    """
    Compute the corrected Akaike Information Criterion (AICc).

    AICc adjusts the AIC for small sample sizes, adding a bias correction term.

    Parameters
    ----------
    y : array-like
        True target values.
    y_hat : array-like
        Predicted target values.
    T : int
        Number of observations (sample size).
    k : int
        Number of explanatory variables (not including intercept).

    Returns
    -------
    aicc : float
        Corrected Akaike Information Criterion value.
    """
    aic = aic_score(y, y_hat, T, k)
    aicc = aic + (2 * (k + 2) * (k + 3)) / (T - k - 3)
    return aicc

def bic_score(y, y_hat, T, k):
    """
    Compute the Bayesian Information Criterion (BIC) score.

    BIC penalizes model complexity more strongly than AIC and is used for model comparison.

    Parameters
    ----------
    y : array-like
        True target values.
    y_hat : array-like
        Predicted target values.
    T : int
        Number of observations (sample size).
    k : int
        Number of explanatory variables (not including intercept).

    Returns
    -------
    bic : float
        Bayesian Information Criterion value.
    """
    sse = np.sum((y - y_hat) ** 2)
    bic = T * np.log(sse / T) + (k + 2) * np.log(T)
    return bic

def overforecast_over_th_score(y, y_hat, y_th):
    """
    Compute the average over-forecast error (y_hat > y).
    """
    error = y_hat - y
    over_mask = (error > 0) & (y > y_th)
    if np.any(over_mask):
        return np.mean(np.abs(error[over_mask]))
    else:
        return 0.0

def underforecast_over_th_score(y, y_hat, y_th):
    """
    Compute the average under-forecast error (y_hat < y).
    """
    error = y_hat - y
    under_mask = (error < 0) & (y > y_th)  
    if np.any(under_mask):
        return np.mean(np.abs(error[under_mask]))
    else:
        return 0.0
    
def mae_over_thr_score(y, y_hat, y_th):
    """
    Compute MAE limited to high y periods (y > y_th).
    """
    error = y_hat - y
    mask = (y > y_th)  
    if np.any(mask):
        return np.mean(np.abs(error[mask]))
    else:
        return 0.0



