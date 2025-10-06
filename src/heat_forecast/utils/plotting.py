from pydoc import html
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
from typing import Optional, Sequence, Dict, Any, Tuple, List, Callable, Iterable, Mapping
from datetime import datetime
import warnings
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from IPython.display import display
from matplotlib import colors
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter, MultipleLocator
import ipywidgets as widgets
from IPython.display import HTML

from .transforms import make_is_winter, get_lambdas, make_transformer, transform_column

# -------------------------------------------------------------------------------
# FOR PLOTTING FORECASTS / TARGET & EXOGENOUS SERIES
# -------------------------------------------------------------------------------

# Helper function to format the axes for series plots based on the displayed period
def configure_time_axes(
        axes: matplotlib.axes.Axes | Sequence[matplotlib.axes.Axes],
        period: Sequence[datetime | pd.Timestamp],
        *,
        global_legend: bool = False,
        legend_fig: Optional[matplotlib.figure.Figure] = None,
        legend_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
    """
    Format time axes of one or more matplotlib axes for time-series
    plots, adapting tick density and labels to the length of the period
    shown.

    The function chooses appropriate major and minor tick locators/
    formatters based on the total number of days between the first and
    last date in *period*.

    All axes receive identical formatting.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or Sequence[matplotlib.axes.Axes]
        One axis or an iterable of axes to be formatted.

    period : Sequence[datetime | pandas.Timestamp]
        Two-element sequence ``[start, end]`` (or any iterable whose first
        and last elements represent the visible date range).  The function
        treats the elements as inclusive.
    
    global_legend : bool, default False
        If True, the function will create a global legend for all axes
    
    legend_fig : matplotlib.figure.Figure, optional
        If *global_legend* is True, this figure will be used to place
        the global legend. If None, the legend will be placed in the first
        axis of *axes*.

    Returns
    -------
    None
        The function works by side-effect; it modifies each axis in
        *axes* and returns nothing.
    """
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    else:
        try:                         
            iter(axes)
        except TypeError:
            raise ValueError(
                "`axes` must be a Matplotlib Axes instance "
                "or an iterable of Axes objects."
        )
    period = sorted(period)  # Ensure period is sorted
    start, end = period[0], period[-1]
    period_days = (end - start).days + 1  # Include the last day

    # Choose locator and formatter based on the period length
    if period_days < 1:
        locator_major = mdates.HourLocator(byhour=range(0, 24, 3))
        formatter_major = mdates.DateFormatter('%H:%M')
        locator_minor = mdates.MinuteLocator(interval=30)
        formatter_minor = None                       
    elif period_days < 11:
        locator_major = mdates.DayLocator(interval=1)
        formatter_major = mdates.DateFormatter('%d %b')
        locator_minor = mdates.HourLocator(byhour=[0, 12])
        formatter_minor = None
    elif period_days < 31:
        locator_major = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_major = mdates.DateFormatter('Wk %W\n%d %b')
        locator_minor = mdates.DayLocator(interval=1)
        formatter_minor = None
    elif period_days < 91:
        locator_major = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_major = mdates.DateFormatter('%d %b')
        locator_minor = mdates.DayLocator(interval=2)
        formatter_minor = None
    elif period_days < 366:
        locator_major = mdates.MonthLocator()
        formatter_major = mdates.DateFormatter('%b\n%Y')
        locator_minor = mdates.WeekdayLocator(byweekday=mdates.MO)
        formatter_minor = mdates.DateFormatter('%d')
    elif period_days < 366 * 2:
        locator_major = mdates.MonthLocator(interval=3)
        formatter_major = mdates.DateFormatter('%b\n%Y')
        locator_minor = mdates.MonthLocator(interval=1)
        formatter_minor = None
    elif period_days < 366 * 5:
        locator_major = mdates.MonthLocator(bymonth=[1,4,7,10])  
        def month_formatter(x, pos):
            dt = mdates.num2date(x)
            if dt.month == 1: return dt.strftime("%b\n%Y")
            else: return dt.strftime("%b")   
        formatter_major = FuncFormatter(month_formatter)
        locator_minor = None
        formatter_minor = None
    else:
        locator_major = mdates.YearLocator(base=2)      
        formatter_major = mdates.DateFormatter('%Y')
        locator_minor = mdates.YearLocator(base=1)
        formatter_minor = None

    with_minor = locator_minor is not None

    # Apply the locator and formatter to all axes
    for ax in axes:
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_major_formatter(formatter_major)
        if with_minor:
            ax.xaxis.set_minor_locator(locator_minor)
            if formatter_minor is not None: ax.xaxis.set_minor_formatter(formatter_minor)
        pad = 12 if with_minor and (formatter_minor is not None) else 2
        ax.tick_params(axis='x', which='major', pad=pad, labelbottom=True)
        ax.tick_params(axis='x', which='minor', pad=2, labelbottom=True)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if global_legend:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    
    # If global_legend is True, create a global legend
    if global_legend:
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if not isinstance(hh, matplotlib.artist.Artist): 
                    continue
                if ll not in labels:           
                    handles.append(hh)
                    labels.append(ll)
        if len(handles) > 1:
            legend_fig.legend(handles, labels,
                              **(legend_kwargs or dict(loc="upper right",
                                                       bbox_to_anchor=(1, 1))))

def plot_cutoff_results(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    cutoffs: Optional[Sequence[pd.Timestamp]] = None,
    cutoffs_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    levels: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    highlight_dayofweek: bool = False,
    order_of_models: Optional[Sequence[str]] = None,
    alpha: float = 0.9,
) -> plt.Figure:
    """
    Produce a multi-panel plot that overlays historical data with
    cross-validation forecasts for selected cut-off dates.

    The function works in four stages:

    1. Select the cut-offs of interest, either explicitly via *cutoffs* or
       by providing an inclusive date range with *cutoffs_period*.
    2. Filter *cv_df* to keep only the requested models, prediction-interval
       levels and series identifiers (*ids*).
    3. Combine those forecasts with the raw observations in *target_df*,
       extending the x-axis by *start_offset* and *end_offset* hours.
    4. Draw one subplot per unique series, including optional weekday
       colouring and shaded prediction intervals.

    Parameters
    ----------
    target_df : pandas.DataFrame
        Historical data with at least the columns
        'unique_id', 'ds' and 'y'.
    cv_df : pandas.DataFrame
        Cross-validation output.  Required columns:
        - 'unique_id', 'ds', 'cutoff'
        - *target_col*
        - one column per model containing the mean forecast
        - optional PI columns named  ``f"{model}-lo-{level}"`` /
          ``f"{model}-hi-{level}"``.
    start_offset, end_offset : int
        Number of hours to pad the left and right sides of the time axis.
    cutoffs : sequence of pandas.Timestamp, optional
        Exact cut-off dates to plot. If omitted, the function will
        use *cutoffs_period* instead. If both are omitted, the function
        will plot all cut-offs found in *cv_df*.
    cutoffs_period : (Timestamp, Timestamp), optional
        Inclusive start and end of a continuous cut-off window. If provided together
        with *cutoffs*, the function will ignore  it and use *cutoffs* instead.
    levels : sequence of int, optional
        Prediction-interval coverages such as ``[80, 95]``.  If omitted,
        intervals are not drawn.
    models : sequence of str, optional
        Subset of model columns to include.  Defaults to all model columns
        present in *cv_df*.
    ids : sequence of str, optional
        Subset of 'unique_id' values to plot.  Defaults to all those found
        after cut-off filtering.
    highlight_dayofweek : bool, default False
        If True, scatter-points of the target series are colour-coded by
        weekday.
    order_of_models : sequence of str, optional
        Desired ordering of models within the legend.  Any models not
        listed are appended afterwards.
    alpha : float, default 0.9
        Opacity for forecast lines (0 = fully transparent, 1 = opaque).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing *len(ids)* vertically stacked subplots.
    """
    target_col = "y"  # Default target column
    if not isinstance(target_df, pd.DataFrame) or not isinstance(cv_df, pd.DataFrame):
        raise TypeError("Both target_df and cv_df must be pandas DataFrames.")
    if not all(col in target_df.columns for col in ['unique_id', 'ds', 'y']):
        raise ValueError(f"target_df must contain 'unique_id', 'ds', and 'y' columns.")
    if not all(col in cv_df.columns for col in ['unique_id', 'ds', 'cutoff']):
        raise ValueError(f"cv_df must contain 'unique_id', 'ds', and 'cutoff' columns.")

    if (cutoffs is None) == (cutoffs_period is None) == False:
        raise warnings.warn(
            "Both `cutoffs` and `cutoffs_period` are provided. "
            "Using `cutoffs` only, ignoring `cutoffs_period`."
        )

    levels = list(levels or [])

    # Model discovery & validation 
    all_models = [
        c for c in cv_df.columns
        if c not in {"ds", "unique_id", "y", "cutoff"}
        and "-lo-" not in c and "-hi-" not in c
    ]

    if models is None:
        models = all_models
    else:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown model columns: {', '.join(unknown)}")

    if order_of_models:
        extra = [m for m in models if m not in order_of_models]
        ordered_models: List[str] = list(order_of_models) + extra
    else:
        ordered_models = list(models)

    # Choose cut-offs
    if cutoffs is not None:
        cutoffs = pd.to_datetime(cutoffs).tolist()
        missing = set(cutoffs) - set(cv_df["cutoff"].unique())
        if missing:
            raise ValueError(f"Cut-offs not found: {', '.join(map(str, missing))}")
        mask_cutoff = cv_df["cutoff"].isin(cutoffs)

    elif cutoffs_period is not None:  # by explicit period (inclusive)
        if not (
            isinstance(cutoffs_period, (tuple, list))
            and len(cutoffs_period) == 2
        ):
            raise TypeError(
                "`cutoffs_period` must be a tuple(start, end) of two timestamps."
            )
        start, end = map(pd.Timestamp, cutoffs_period)
        if start > end:
            raise ValueError("cutoffs_period: start date must be before end date.")
        mask_cutoff = (cv_df["cutoff"] >= start) & (cv_df["cutoff"] <= end)
    else:
        mask_cutoff = cv_df["cutoff"].notna()

    cut_df = cv_df.loc[mask_cutoff].copy()
    if cut_df.empty:
        raise ValueError("No cut-offs found in the requested period.")

    # ID filtering 
    all_ids = cut_df["unique_id"].unique()
    if ids is None:
        ids = all_ids
    else:
        unknown = set(ids) - set(all_ids)
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")

    cut_df = cut_df[cut_df["unique_id"].isin(ids)]

    # Columns to keep
    keep_cols = ["ds", "unique_id", "cutoff"] + ordered_models
    for lv in levels:
        keep_cols += [f"{m}-lo-{lv}" for m in ordered_models]
        keep_cols += [f"{m}-hi-{lv}" for m in ordered_models]
    cut_df = cut_df[keep_cols]

    # Plotting limits 
    plot_start = cut_df["ds"].min() - pd.Timedelta(hours=start_offset)
    plot_end   = cut_df["ds"].max() + pd.Timedelta(hours=end_offset)

    # ---------- figure ----------
    n_series = len(ids)
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    model_colors = sns.color_palette("husl", len(ordered_models))

    for ax, uid in zip(axes, ids):
        # historical line
        mask_train = (
            (target_df["unique_id"] == uid) &
            (target_df["ds"] >= plot_start) &
            (target_df["ds"] <= plot_end)
        )
        train_grp = target_df.loc[mask_train]
        ax.plot(train_grp["ds"], train_grp[target_col], color="black", label="Target")

        # weekday scatter
        if highlight_dayofweek:
            weekday_pal = (
                sns.color_palette("YlOrBr", 5) +
                sns.color_palette("YlGn", 2)
            )
            for i, day in enumerate(
                ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
            ):
                mask = train_grp["ds"].dt.day_name() == day
                ax.scatter(
                    train_grp.loc[mask, "ds"],
                    train_grp.loc[mask, target_col],
                    color=weekday_pal[i],
                    label=day,
                )

        # forecasts
        for cutoff_val in cut_df["cutoff"].unique():
            f_grp = cut_df[
                (cut_df["cutoff"] == cutoff_val) &
                (cut_df["unique_id"] == uid)
            ]
            for idx, model in enumerate(ordered_models):
                c = model_colors[idx]
                ax.plot(
                    f_grp["ds"], f_grp[model],
                    color=c, alpha=alpha,
                    label=f"{model}",
                )
                for lv in levels:
                    lo = f_grp[f"{model}-lo-{lv}"]
                    hi = f_grp[f"{model}-hi-{lv}"]
                    if not lo.empty and not hi.empty:
                        alpha_lv = 0.1 + (100 - lv) / 100 * 0.8  # narrower PI darker
                        ax.fill_between(
                            f_grp["ds"], lo, hi,
                            color=c, alpha=alpha_lv, linewidth=0,
                            label=f"{model} {lv}% PI",
                        )

        ax.set_title(f"Series {uid}")
        ax.legend().remove()  # global legend later

    # Configure time axes
    configure_time_axes(
        axes, period=[plot_start, plot_end],
        global_legend=True, legend_fig=fig,
    )

    fig.suptitle("Forecast Results", y=0.98, fontsize=16)
    fig.supxlabel('Date Time [H]')
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])

    return fig

def plot_cutoff_results_with_exog(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    aux_df: Optional[pd.DataFrame] = None,
    exog_vars: Optional[Sequence[str]] = None,
    cutoffs: Optional[Sequence[pd.Timestamp]] = None,
    cutoffs_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    levels: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    id: Optional[str] = None,
    highlight_dayofweek: bool = False,
    order_of_models: Optional[Sequence[str]] = None,
    alpha: float = 0.9,
    add_context: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot forecast results with multiple cutoffs for a single time series,
    along with one panel per exogenous variable.

    This visualization is useful for evaluating model performance across
    cutoffs and understanding how exogenous variables relate to the target.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing the historical target series.
        Must include columns: 'unique_id', 'ds', 'y'.

    cv_df : pd.DataFrame
        Cross-validation results containing forecast values.
        Must include: 'unique_id', 'ds', 'cutoff', one or more model columns,
        and optionally prediction interval columns like 'model-lo-90', 'model-hi-90'.

    start_offset : int
        Hours to extend the plot before the first forecast timestamp.

    end_offset : int
        Hours to extend the plot after the last forecast timestamp.

    aux_df : Optional[pd.DataFrame], default=None
        DataFrame containing the exogenous variables.
        Must include 'unique_id', 'ds', and all variables in `exog_vars`.

    exog_vars : Optional[Sequence[str]], default=None
        List of exogenous variable names to plot in separate panels.

    cutoffs : Optional[Sequence[pd.Timestamp]], default=None
        Exact cutoffs to plot. Takes precedence over `cutoffs_period` if both are provided.

    cutoffs_period : Optional[Tuple[pd.Timestamp, pd.Timestamp]], default=None
        Inclusive period for selecting all cutoffs between `start` and `end`.

    levels : Optional[Sequence[int]], default=None
        List of prediction interval levels (e.g., [80, 95]). If None, no PIs are shown.

    models : Optional[Sequence[str]], default=None
        Subset of model columns in `cv_df` to plot. Defaults to all detected model columns.

    id : Optional[str], default=None
        `unique_id` of the series to plot. If None, the first available ID in `cv_df` is used.

    highlight_dayofweek : bool, default=False
        If True, scatter points in the target series are colored by weekday.

    order_of_models : Optional[Sequence[str]], default=None
        Specifies the order of models in the plot legend. Unlisted models are appended.

    alpha : float, default=0.9
        Opacity for forecast lines.

    add_context : bool, default=False
        If True, an additional panel is included showing the full historical context
        of the target series, and each subplot (including context) will have its own legend.
        If False, a single shared legend is shown above the plot.

    figsize : Optional[Tuple[int, int]], default=None
        Size of the figure to create, in inches. If None, defaults to (12, 4 * (number of panels)).

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure with the main target series forecast panel,
        one panel per exogenous variable, and optionally a context panel.
    """
    target_col = "y"  # Default target column
    if not isinstance(target_df, pd.DataFrame) or not isinstance(cv_df, pd.DataFrame):
        raise TypeError("target_df and cv_df must be pandas DataFrames.")
    if aux_df is not None and not isinstance(aux_df, pd.DataFrame):
        raise TypeError("aux_df must be a pandas DataFrame if provided.")

    if (cutoffs is None) == (cutoffs_period is None):
        if cutoffs is None:
            raise ValueError(
                "At least one of `cutoffs` or `cutoffs_period` must be provided."
            )
        else: # both are provided
            warnings.warn(
                "Both `cutoffs` and `cutoffs_period` are provided. Using `cutoffs` only."
            )

    levels = list(levels or [])

    # Model discovery & validation
    all_models = [
        c for c in cv_df.columns
        if c not in {"ds", "unique_id", "y", "cutoff"}
        and "-lo-" not in c and "-hi-" not in c
    ]

    if models is None:
        models = all_models
    else:
        unknown = set(models) - set(all_models)
        if unknown:
            raise ValueError(f"Unknown model columns: {', '.join(unknown)}")

    if order_of_models:
        extra = [m for m in models if m not in order_of_models]
        ordered_models: List[str] = list(order_of_models) + extra
    else:
        ordered_models = list(models)

    # Choose cut-offs 
    if cutoffs is not None:
        cutoffs = pd.to_datetime(cutoffs).tolist()
        missing = set(cutoffs) - set(cv_df["cutoff"].unique())
        if missing:
            raise ValueError(f"Cut-offs not found: {', '.join(map(str, missing))}")
        mask_cutoff = cv_df["cutoff"].isin(cutoffs)

    else:  # by explicit period (inclusive)
        if not (
            isinstance(cutoffs_period, (tuple, list))
            and len(cutoffs_period) == 2
        ):
            raise TypeError(
                "`cutoffs_period` must be a tuple(start, end) of two timestamps."
            )
        start, end = map(pd.Timestamp, cutoffs_period)
        if start > end:
            raise ValueError("cutoffs_period: start date must be before end date.")
        mask_cutoff = (cv_df["cutoff"] >= start) & (cv_df["cutoff"] <= end)

    cut_df = cv_df.loc[mask_cutoff].copy()
    if cut_df.empty:
        raise ValueError("No cut-offs found in the requested period.")

    # ID filtering 
    all_ids = cut_df["unique_id"].unique()
    if len(all_ids) > 1 and id is None:
        raise ValueError("Multiple unique_id values found, please specify one with `id` parameter.")
    if id is not None:    
        unknown = set([id]) - set(all_ids)
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")
    else:
        id = all_ids[0]

    cut_df = cut_df[cut_df["unique_id"]==id]
    target_id_df = target_df[target_df["unique_id"] == id]

    # Columns to keep 
    keep_cols = ["ds", "unique_id", "cutoff"] + ordered_models
    for lv in levels:
        keep_cols += [f"{m}-lo-{lv}" for m in ordered_models]
        keep_cols += [f"{m}-hi-{lv}" for m in ordered_models]
    cut_df = cut_df[keep_cols]

    # Plotting limits 
    plot_start = cut_df["ds"].min() - pd.Timedelta(hours=start_offset)
    plot_end   = cut_df["ds"].max() + pd.Timedelta(hours=end_offset)

    # ---------- figure ----------
    exog_vars = list(exog_vars or [])
    n_exog = len(exog_vars)
    figsize = figsize or (12, 4 * (1 + n_exog + int(add_context)))
    fig, axes = plt.subplots(1+n_exog+int(add_context), 1, figsize=figsize)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    model_colors = sns.color_palette("husl", len(ordered_models))

    ax = axes[0]
    
    # Historical line
    mask_train = (
        (target_id_df["ds"] >= plot_start) &
        (target_id_df["ds"] <= plot_end)
    )
    train_grp = target_id_df.loc[mask_train]
    ax.plot(train_grp["ds"], train_grp[target_col], color="black", label="Target")

    # Weekday scatter
    if highlight_dayofweek:
        weekday_pal = (
            sns.color_palette("YlOrBr", 5) +
            sns.color_palette("YlGn", 2)
        )
        for i, day in enumerate(
            ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
        ):
            mask = train_grp["ds"].dt.day_name() == day
            ax.scatter(
                train_grp.loc[mask, "ds"],
                train_grp.loc[mask, target_col],
                color=weekday_pal[i],
                label=day,
            )

    # Forecasts
    for cutoff_val in cut_df["cutoff"].unique():
        f_grp = cut_df[
            (cut_df["cutoff"] == cutoff_val) &
            (cut_df["unique_id"] == id)
        ]
        for idx, model in enumerate(ordered_models):
            c = model_colors[idx]
            ax.plot(
                f_grp["ds"], f_grp[model],
                color=c, alpha=alpha,
                label=f"{model}" if cutoff_val == cut_df['cutoff'].unique()[0] else None
            )
            for lv in levels:
                lo = f_grp[f"{model}-lo-{lv}"]
                hi = f_grp[f"{model}-hi-{lv}"]
                if not lo.empty and not hi.empty:
                    alpha_lv = 0.1 + (100 - lv) / 100 * 0.8  # narrower PI darker
                    ax.fill_between(
                        f_grp["ds"], lo, hi,
                        color=c, alpha=alpha_lv, linewidth=0,
                        label=f"{model} {lv}% PI" if cutoff_val == cut_df['cutoff'].unique()[0] else None,
                    )

    ax.set_title(f"Forecasts")
    if add_context: # in this case, the plot is clearer with per-axes legends
        ax.legend(loc="upper left")  # Per-axes legend
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if aux_df is not None:
        # Plot the auxiliary variables
        missing_exog = set(exog_vars) - set(aux_df.columns)
        if missing_exog:
            raise ValueError(f"Missing exogenous variables in aux_df: {', '.join(missing_exog)}")
    
        units = {
            'temperature': '°C',
            'pressure': 'hPa',
            'dew_point': '°C',
            'humidity': '%',
            'wind_speed': 'm/s'
        }
        aux_grp = aux_df[(aux_df["unique_id"] == id) & 
                        (aux_df["ds"] >= plot_start) & 
                        (aux_df["ds"] <= plot_end)]
        for exog_idx, exog_var in enumerate(exog_vars):
            ax = axes[exog_idx + 1]
            ax.plot(aux_grp["ds"], aux_grp[exog_var], color="blue")
            ax.set_title(f"{exog_var} ({units.get(exog_var, '')})")

    if add_context:
        ax = axes[-1]
        ax.set_title("Full Context vs Current Range")

        ds_all_start = cv_df["ds"].min()
        ds_all_end = cv_df["ds"].max()

        ds_cut_start = cut_df["ds"].min()
        ds_cut_end = cut_df["ds"].max()

        # Full context range (gray)
        context_range = target_id_df[
            (target_id_df["ds"] >= ds_all_start) & (target_id_df["ds"] <= ds_all_end)
        ].iloc[::2]  # <- this skips 1 row out of 2 for performance
        ax.plot(
            context_range["ds"], context_range[target_col],
            color="gray", label="Full Context Range"
        )

        # Highlighted range (black)
        current_range = target_id_df[
            (target_id_df["ds"] >= ds_cut_start) & (target_id_df["ds"] <= ds_cut_end)
        ].iloc[::2]  # <- this skips 1 row out of 2 for performance
        ax.plot(
            current_range["ds"], current_range[target_col],
            color="black", label="Current Plot Range"
        )

        ax.legend(loc="upper left")  # Per-axes legend

    # Configure time axes
    configure_time_axes(
        axes, period=[plot_start, plot_end],
        global_legend=not add_context, legend_fig=fig,
    )

    if add_context:
        configure_time_axes(
            [axes[-1]], period=cv_df["ds"],
            global_legend=False,
        )

    fig.suptitle(f"Forecast Results for ID: {id}", y=0.98, fontsize=16)
    fig.supxlabel('Date Time [H]')
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.98])

    return fig

def interactive_plot_cutoff_results_v0(
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    *,
    aux_df: Optional[pd.DataFrame] = None,
    exog_vars: Optional[Sequence[str]] = None,
    n_windows: int = 5,
    models: Optional[Sequence[str]] = None,
    id: Optional[str] = None,
    add_context: bool = False,
    levels: Optional[Sequence[int]] = None,
    alpha: float = 0.9,
    order_of_models: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    only_aligned_to_day: bool = True,
):
    """
    Create an interactive slider to visualize rolling forecast windows
    from cross-validation results.

    Parameters
    ----------
    target_df : pd.DataFrame
        Target time series data.
    cv_df : pd.DataFrame
        Cross-validation forecast results with cutoffs.
    aux_df : Optional[pd.DataFrame], default=None
        Optional DataFrame of exogenous variables.
    exog_vars : Optional[Sequence[str]], default=None
        List of exogenous variables to plot.
    n_windows : int, default=5
        Number of cutoffs to plot at once (centered around slider position).
    models : Optional[Sequence[str]], default=None
        Subset of models to plot.
    id : Optional[str], default=None
        Unique ID of the time series to plot.
    add_context : bool, default=False
        Whether to include a full-history context panel.
    levels : Optional[Sequence[int]], default=None
        List of prediction interval levels to show.
    alpha : float, default=0.9
        Opacity of forecast lines.
    order_of_models : Optional[Sequence[str]], default=None
        Order of models in the plot legend.
    figsize : Optional[Tuple[int, int]], default=None
        Size of the figure to create, in inches. If None, defaults to (12, 4 * (number of panels)).
    
    Returns
    -------
    None
        The function creates an interactive plot with a slider to select
    """
    from IPython.display import clear_output

    if only_aligned_to_day:
        cv_df = cv_df[cv_df['cutoff'].dt.hour == 0].copy()

    cutoffs = sorted(cv_df['cutoff'].unique())
    if len(cutoffs) < n_windows:
        raise ValueError("Not enough cutoffs to create the requested number of windows.")
    
    # Create the slider
    slider = widgets.IntSlider(
        value=len(cutoffs) // 2,
        min=0,
        max=len(cutoffs) - 1,
        step=1,
        description="Cutoff idx:",
        continuous_update=False
    )

    def interactive_plot(center_cutoff_index):
        clear_output(wait=True)

        half_window = (n_windows - 1) // 2 
        start = max(0, center_cutoff_index - half_window)
        end = min(len(cutoffs), center_cutoff_index + half_window + 1 + int(n_windows % 2 == 0))
        selected_cutoffs = cutoffs[start:end]

        mfig = plot_cutoff_results_with_exog(
            target_df=target_df,
            aux_df=aux_df,
            exog_vars=exog_vars,
            cv_df=cv_df,
            start_offset=48,
            end_offset=48,
            cutoffs=selected_cutoffs,
            models=models,
            id=id,
            add_context=add_context,
            levels=levels,
            alpha=alpha,
            order_of_models=order_of_models,  # Use the same models for ordering
            figsize=figsize
        )

        plt.show(mfig)
        #plt.close('all')    # Prevent Jupyter from showing it again
        
        return None

    # Hook up the slider
    widgets.interact(interactive_plot, center_cutoff_index=slider)

    return None

import ipywidgets as widgets
from plotly.tools import mpl_to_plotly

def plotly_forecasts_with_exog(
    *,
    target_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    aux_df=None,
    exog_vars=None,
    n_windows: int = 5,
    models=None,
    id=None,
    add_context: bool = False,
    levels=None,
    alpha: float = 0.9,
    order_of_models=None,
    figsize=None,
    only_aligned_to_day: bool = True,
    start_offset: int = 48,
    end_offset: int = 48,
):
    # --- prep once
    cv_df = cv_df.copy()
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    if only_aligned_to_day:
        cv_df = cv_df[cv_df["cutoff"].dt.hour == 23].copy()

    cutoffs = sorted(cv_df["cutoff"].unique())
    if len(cutoffs) < max(1, n_windows):
        raise ValueError("Not enough cutoffs for the requested window size.")

    slider = widgets.IntSlider(
        value=len(cutoffs)//2, min=0, max=len(cutoffs)-1, step=1,
        description="Cutoff idx:", continuous_update=False
    )
    out = widgets.Output()

    def _strip_legends_inplace(mfig):
        for ax in mfig.get_axes():
            leg = ax.get_legend()
            if leg is not None:
                try: leg.remove()
                except Exception: leg.set_visible(False)
            try: ax.legend_ = None
            except Exception: pass

    def _select(idx: int):
        half = (n_windows-1)//2
        start = max(0, idx - half)
        end   = min(len(cutoffs), idx + half + 1 + int(n_windows % 2 == 0))
        return cutoffs[start:end]

    def _render(idx: int):
        selected_cutoffs = _select(idx)

        mfig = plot_cutoff_results_with_exog(
            target_df=target_df,
            cv_df=cv_df,
            start_offset=start_offset,
            end_offset=end_offset,
            aux_df=aux_df,
            exog_vars=exog_vars,
            cutoffs=selected_cutoffs,
            models=models,
            id=id,
            add_context=add_context,
            levels=levels,
            alpha=alpha,
            order_of_models=order_of_models,
            figsize=figsize,
        )
        _strip_legends_inplace(mfig)
        pfig = mpl_to_plotly(mfig)
        plt.close(mfig)  # prevent MPL from also rendering

        with out:
            out.clear_output(wait=True)  # ← ensures ONLY one figure is shown
            display(pfig)                # ← no fig.show(), no print()

    def _on_change(change):
        if change["name"] == "value":
            _render(change["new"])

    slider.observe(_on_change, names="value")
    _render(slider.value)  # initial draw
    
    ui = widgets.VBox([slider, out])
    display(ui)
    return

def custom_plot_results(
    target_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    start_offset: int,
    end_offset: int,
    *,
    levels: list = [],
    target_col: str = 'y',
    ids: Optional[list] = None,
    highlight_dayofweek: bool = False,
    with_naive: bool = False, # Whether to include the naive model in the plot
    target_train_df: Optional[pd.DataFrame] = None,  # Training data for the naive model
    order_of_models: Optional[List[str]] = None,  # Order of models to plot: the included models will be plotted as first and in this order
    alpha: float = 0.9,  # Transparency of the forecast lines
    return_fig: bool = False
) -> plt.Figure:
    """
    Plot historical data and model forecasts for a set of series.

    The figure contains one subplot per ``unique_id`` in *ids*.  
    Each panel shows:

    * the target series from *target_df* (black line, optionally colour-
      coded by weekday);
    * one or more forecast paths from *forecast_df*;
    * shaded prediction intervals for coverages listed in *levels*;
    * an optional 24-hour seasonal naive forecast computed on the fly
      from *target_train_df* when *with_naive* is True.

    Parameters
    ----------
    target_df : pandas.DataFrame
        Historical observations with columns:
        ``'unique_id'``, ``'ds'`` (timestamp) and *target_col*.
    forecast_df : pandas.DataFrame
        Forecast output containing at minimum the columns
        ``'unique_id'`` and ``'ds'`` plus one column per model.  Prediction
        intervals, if present, must follow the naming pattern  
        ``f"{model}-lo-{level}"`` / ``f"{model}-hi-{level}"``.
    start_offset, end_offset : int
        Time padding on the left and right side of the x-axis, expressed
        in hours.
    levels : list[int], default []
        Coverage levels for prediction intervals, e.g. ``[80, 95]``.
        If empty, no intervals are filled.
    target_col : str, default ``"y"``
        Name of the dependent-variable column in *target_df*.
    ids : list[str], default ``['F1', …, 'F5']``
        The ``unique_id`` values to plot, one subplot per id.
    highlight_dayofweek : bool, default False
        Scatter the target observations using a Monday-Sunday palette.
    with_naive : bool, default False
        Compute and add a 24-hour seasonal naive model (alias
        ``'Naive24h'``) by calling *StatsForecast* on *target_train_df*.
    target_train_df : pandas.DataFrame, optional
        Training data used only when *with_naive* is True.  Must have the
        same structure as *target_df*.
    order_of_models : list[str], optional
        Desired ordering of model columns in the legend.  Any models not
        listed are appended afterwards.
    alpha : float, default 0.9
        Transparency applied to forecast lines.
    return_fig : bool, default False
        If True, the function returns the figure object instead of None.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with ``len(ids)`` stacked subplots.
    """

    if not isinstance(target_df, pd.DataFrame) or not isinstance(forecast_df, pd.DataFrame):
        raise TypeError("target_df and forecast_df must be pandas DataFrames.")
    if not all(col in target_df.columns for col in ['unique_id', 'ds', target_col]):
        raise ValueError(f"target_df must contain 'unique_id', 'ds', and '{target_col}' columns.")
    if not all(col in forecast_df.columns for col in ['unique_id', 'ds']):
        raise ValueError("forecast_df must contain 'unique_id' and 'ds' columns.")
    if not isinstance(levels, list) or not all(isinstance(lv, int) for lv in levels):
        raise TypeError("levels must be a list of integers.")
    
    # Validate ids
    if ids is None or len(ids) == 0:
        ids = target_df['unique_id'].unique().tolist()
    else:
        if not isinstance(ids, Iterable) or not all(isinstance(i, str) for i in ids):
            raise TypeError("ids must be an Iterable of strings.")

        unknown = set(ids) - set(target_df['unique_id'].unique())
        if unknown:
            raise ValueError(f"Unknown unique_id values: {', '.join(unknown)}")

    if with_naive:
        h = len(forecast_df['ds'].unique())  # Number of hours to forecast
        
        if target_train_df is None:
            raise ValueError("target_train_df must be provided when with_naive is True.")
        target_train_df = target_train_df.copy()
        
        # Select only the ids we want to plot
        target_train_df = target_train_df[target_train_df['unique_id'].isin(ids)].reset_index(drop=True)
        
        # Compute forecasts using the naive method
        naive_model24 = SeasonalNaive(season_length=24, alias='Naive24h')
        naive_forecast_df = StatsForecast(
            models=[naive_model24], 
            freq='h'
        ).forecast(h, target_train_df, level=levels)

        # Merge the forecasts into a single DataFrame
        forecast_df = (
            forecast_df
            .merge(naive_forecast_df, on=['unique_id', 'ds'], how='left')
        )
        
    # set the frequency of the validation index (e.g. 'H', 'D', …)
    freq = 'h'   

    # Create a date range for the x-axis
    td_start = pd.Timedelta(start_offset, unit=freq)
    td_end   = pd.Timedelta(end_offset,   unit=freq)
    start_dstemp = forecast_df['ds'].min() - td_start
    end_dstemp   = forecast_df['ds'].max() + td_end
    
    fig, axes = plt.subplots(nrows=len(ids), ncols=1, figsize=(12, 4*len(ids)))
    axes = axes.flatten() if len(ids) > 1 else [axes]
    for ax, uid in zip(axes, ids):
        # Plot training data
        mask_target = (
            (target_df['ds'] >= start_dstemp) &
            (target_df['ds'] <= end_dstemp) &
            (target_df['unique_id'] == uid)
        )
        target_grp = target_df[mask_target]
        ax.plot(target_grp['ds'], target_grp[target_col], label='Target', color='black')

        if highlight_dayofweek:
            pallette_wkdays = sns.color_palette(palette='YlOrBr', n_colors=5)  # Use a color palette for weekdays
            pallette_weekends = sns.color_palette(palette='YlGn', n_colors=2)  # Use a color palette for weekends
            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pal = pallette_wkdays + pallette_weekends  # Combine palettes for weekdays and weekends
            for i, day in enumerate(days):
                mask = target_grp['ds'].dt.day_name() == day
                ax.scatter(
                    target_grp.loc[mask, 'ds'], 
                    target_grp.loc[mask, target_col], 
                    color=pal[i], 
                    label=f'{day}', 
                )
        
        # Plot forecasted data
        models = [c for c in forecast_df.columns
                  if c != 'ds' and c != 'unique_id' and '-lo-' not in c and '-hi-' not in c]
        if order_of_models is not None:
            if not set(order_of_models).issubset(set(models)):
                raise ValueError("order_of_models must be a subset of the models in forecast_df.")
            models = order_of_models + [m for m in models if m not in order_of_models]
        if len(models) == 1:
            colors = ['blue']
        else:
            colors = sns.color_palette("tab10", len(models))  # Use a color palette for models
        forecast_grp = forecast_df.query("unique_id == @uid")
        for i, model in enumerate(models):
            ax.plot(forecast_grp['ds'], forecast_grp[model], label=model, color=colors[i], alpha=alpha)
            for lv in levels:
                low_col = f'{model}-lo-{lv}'
                high_col = f'{model}-hi-{lv}'
                if low_col in forecast_grp and high_col in forecast_grp:
                    min_alpha = 0.1  # fix alpha to avoid transparency issues
                    max_alpha = 0.9
                    alpha_lvl = max_alpha - (float(lv) / 100) * (max_alpha - min_alpha)
                    ax.fill_between(
                        forecast_grp['ds'], 
                        forecast_grp[f'{model}-lo-{lv}'], 
                        forecast_grp[f'{model}-hi-{lv}'], 
                        color=colors[i], alpha=alpha_lvl, label=f'{lv}% PI'
                    )
        ax.set_title(f'Series {uid}')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
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
               bbox_to_anchor=(1, 1))

    fig.suptitle('Forecast Results', y=0.98, fontsize=16)       # push it up a bit
    fig.tight_layout(rect=[0, 0, 1, 0.97])                      # leave the top 3% for the suptitle

    if return_fig:
        return fig

def custom_plot_exog(
    aux_df: pd.DataFrame,
    forecast_ds: pd.Series,
    start_offset: int,
    end_offset: int,
    exog_list: List[str] = ['dew_point', 'pressure', 'temperature', 'humidity', 'wind_speed'],
)-> plt.Figure:
    """
    Plot a window of exogenous variables surrounding a forecast horizon.

    The function draws one subplot per name in *exog_list*.  
    Each panel displays the selected variable for **all** ``unique_id``
    traces contained in *aux_df*.

    Parameters
    ----------
    aux_df : pandas.DataFrame
        Historical exogenous data with columns

        - ``'unique_id'`` — series identifier  
        - ``'ds'``        — timestamp (freq = 1 h assumed)  
        - one column for every exogenous variable to be plotted
          (e.g. *dew_point*, *pressure*, …).
    forecast_ds : pandas.Series
        A series of forecast timestamps (dtype ``datetime64[ns]``) that
        anchor the plotting window.
    start_offset, end_offset : int
        Hours to extend the plot **before** the earliest and **after**
        the latest timestamp in *forecast_ds*.
    exog_list : list[str], optional
        Names of exogenous columns to include.  Defaults to

        ``['dew_point', 'pressure', 'temperature', 'humidity', 'wind_speed']``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with ``len(exog_list)`` stacked subplots.
    """
    # Create a date range for the x-axis
    freq='h'
    td_start = pd.Timedelta(start_offset, unit=freq)
    td_end   = pd.Timedelta(end_offset,   unit=freq)
    start_dstemp = forecast_ds.min() - td_start
    end_dstemp   = forecast_ds.max() + td_end

    # Define units of measurement for each exogenous variable
    exog_units = {
        'dew_point': '°C',
        'pressure': 'hPa',
        'temperature': '°C',
        'humidity': '%',
        'wind_speed': 'm/s'
    }
    fig, axes = plt.subplots(nrows=len(exog_list), ncols=1, figsize=(12, 4*len(exog_list)))
    axes = axes.flatten() if len(exog_list) > 1 else [axes]
    aux_palette = sns.color_palette("tab10", aux_df['unique_id'].nunique())  # Use a color palette for unique_ids
    for ax, exog in zip(axes, exog_list):
        # filter & pivot 
        mask = aux_df['ds'].between(start_dstemp, end_dstemp)
        pivot_df = aux_df.loc[mask, ['ds', 'unique_id', exog]] \
                        .pivot(index='ds', columns='unique_id', values=exog)
        # I avoided using seborn to control more easily the legend
        for i, uid in enumerate(pivot_df.columns):
            ax.plot(
                pivot_df.index, 
                pivot_df[uid], 
                label=uid, 
                alpha=0.7, 
                color=aux_palette[i] 
            )
            ax.set_title(f'{exog} ({exog_units[exog]})')
    
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
               bbox_to_anchor=(1, 1))

    fig.suptitle('Exogenous variables', y=0.98, fontsize=16)    # push it up a bit
    fig.tight_layout(rect=[0, 0, 1, 0.97])                      # leave the top 3% for the suptitle
    return fig


def plot_daily_seasonality(
    target_df: pd.DataFrame,
    *,
    only_cold_months: bool = True,
    make_is_winter: Callable[[str], Callable[[pd.Timestamp], bool]] = make_is_winter,
    transform: str = "none",
    lambda_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ids: Optional[Iterable[str]] = None,
    plot_range: Optional[pd.DatetimeIndex] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Plot daily seasonality of heat demand after an optional transformation.

    What the figure shows
    ---------------------
    **Rows** : One subplot per `unique_id`.  
    **X-axis** : Hour of day (0-23).  
    **Y-axis** : `y` after the selected transformation (`identity`, `log`,
        `boxcox`, or `boxcox_winter`).  
    **Lines** : Each coloured line is *one month* (January-December palette
        is fixed, so January is always the same green, etc.).  
    **Error band** : Semi-transparent ribbon = an 80 % percentile interval
        (±40 %) across all _winter days_ that fall in `plot_range` for that month.  
        In other words, for January the mean at hour 06 is the average of every 06:00 reading on winter days
        in January, and the band spans the 10th-90th percentiles of those hourly readings.  
        Wider = greater day-to-day variability.

    Parameters
    ----------
    target_df : pd.DataFrame
        Must contain columns 'unique_id', 'ds' (datetime64) and
        'y' (numeric).
    only_cold_months : bool, default True
        If True, only months that are considered "cold" by the
        `make_is_winter` function are included in the plot.
    make_is_winter : Callable[[str], Callable[[pd.Timestamp], bool]], default make_is_winter
        Function that returns a callable to determine if a date is in winter.
        The callable should accept a timestamp and return True if it is winter,
        False otherwise. The function is called with the `unique_id` as an argument.
    transform : {'none', 'log', 'boxcox', 'boxcox_winter'}, default 'none'
        Forward transform applied to `y`.  For the two Box-Cox options a
        λ-search is performed. If 'boxcox_winter', the λ is estimated
        only on winter days (as defined by `make_is_winter`).
    lambda_window : tuple(start, end), optional
        Inclusive date window used to estimate λ when *transform*
        starts with “boxcox”.  Ignored otherwise.
    ids : iterable of str, optional
        Subset of `unique_id`s to include. 
    plot_range : pd.DatetimeIndex, optional
        Restrict the plot to this date range. If None, uses the full range of `target_df['ds']`.
    verbose : bool, default False
        Print and display the λ table when estimating Box-Cox parameters.

    Returns
    -------
    dict {unique_id: lambda}.  Empty if no Box-Cox applied.
    """
    # Convert ids argument to list and validate
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(
            f"The following ids are not in target_df: {sorted(missing)}. "
            f"Available ids: {sorted(available_ids)}"
        )
    target_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # Estimate Box-Cox lambdas if requested
    lambdas: Dict[str, float] = {}
    if transform.startswith("boxcox"):
        if lambda_window is None:
            raise ValueError("lambda_window must be provided for Box-Cox transforms.")
        if transform == "boxcox_winter" and not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        start, end = lambda_window
        lambda_df = target_df[(target_df["ds"] >= start) & (target_df["ds"] <= end)]
        if lambda_df.empty: 
            raise ValueError("lambda_window contains no data.")
        lambdas = get_lambdas(
            df=lambda_df,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            make_is_winter=make_is_winter
        )
        transform_name = "boxcox"
        if verbose:
            print("Estimated Box-Cox λ:")
            display(pd.DataFrame(lambdas.items(), columns=["unique_id", "lambda"]))
    else:
        transform_name = transform

    # Apply transform and restrict to plot range
    if plot_range is None: plot_range = target_df["ds"].sort_values().unique()
    plot_df = target_df[target_df["ds"].isin(plot_range)].copy()
    fwd = make_transformer(transform_name, "y", lambdas or None, inv=False)
    plot_df["y_transformed"] = transform_column(plot_df, fwd)

    # Keep only winter rows
    if only_cold_months:
        if not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        is_winter_fn = {uid: make_is_winter(uid) for uid in plot_df["unique_id"].unique()}
        winter_mask = (
            plot_df.groupby("unique_id")["ds"]
            .transform(lambda s: s.map(is_winter_fn[s.name]))    # s.name is the uid
        )
        plot_df = plot_df[winter_mask]

    # Prepare month palette with fixed mapping
    months_sorted = np.arange(1, 13)
    month_palette = dict(zip(months_sorted, sns.color_palette("crest", 12)))
    plot_df["month"] = plot_df["ds"].dt.month
    plot_df["hour"] = plot_df["ds"].dt.hour

    # Create one subplot per id
    n = len(ids)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 3 * n), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (uid, grp) in zip(axes, plot_df.groupby("unique_id")):
        sns.lineplot(
            data=grp,
            x="hour",
            y="y_transformed",
            hue="month",
            palette=month_palette,
            legend=False,
            ax=ax,
            errorbar=("pi", 80),
        )

        # Annotate last point of each month
        for m, subset in grp.groupby("month"):
            ax.text(
                subset['hour'].iloc[-1],
                subset['y_transformed'].iloc[-1],
                str(m),
                fontsize=15,
                color=month_palette[m],  
                ha="left",
                alpha=0.9,
            )

        ax.set_title(uid)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticks(np.arange(0, 24, 1))

    # Shared labels and formatting
    fig.suptitle('Heat Demand for each month in the given period', fontsize=18)
    fig.supxlabel('Hour of Day', fontsize=14)
    fig.supylabel('Heat Demand [kWh]', fontsize=14)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    return lambdas

def plot_weekly_seasonality(
    target_df: pd.DataFrame,
    *,
    only_cold_months: bool = True,
    make_is_winter: Callable[[str], Callable[[pd.Timestamp], bool]] = make_is_winter,
    transform: str = "none",
    lambda_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ids: Optional[Iterable[str]] = None,
    plot_range: Optional[pd.DatetimeIndex] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Plot the **average winter-week load profile** (hour-of-week 0-167) for one
    or more `unique_id`s, colour-coded by calendar month.

    What the figure shows
    ---------------------
    **Rows** : One subplot per `unique_id`.  
    **X-axis** : Hour of week (0 = Mon 00:00, …, 167 = Sun 23:00).  
    **Y-axis** : `y` after the selected transformation (`none`, `log`, `boxcox`, or `boxcox_winter`).  
    **Lines** : Each coloured line is *one month*; palette is fixed so January is always the same green, etc.  
    **Error band** : Semi-transparent ribbon = 80 % percentile interval (10th-90th) across all winter weeks 
        inside `plot_range` for that month.  At `hour_of_week=30` (Tue 06:00) for February, the mean is the
        average of every Tuesday-06:00 winter reading in February, and the band spans the 10th-90th percentiles 
        of those readings. Wide ribbon ⇒ high week-to-week variability.

    Parameters
    ----------
    target_df : pd.DataFrame
        Columns 'unique_id', 'ds' (datetime64) and 'y' (numeric).
    only_cold_months : bool, default True
        If True, only months that are considered "cold" by the
        `make_is_winter` function are included in the plot.
    make_is_winter : Callable[[str], Callable[[pd.Timestamp], bool]], default make_is_winter
        Function that returns a callable to determine if a date is in winter.
        The callable should accept a timestamp and return True if it is winter,
        False otherwise. The function is called with the `unique_id` as an argument.
    transform : {'none', 'log', 'boxcox', 'boxcox_winter'}, default 'none'
        Forward transform applied to `y`.  Box-Cox options estimate λ unless
        ``lambda_window`` is ``None``.
    lambda_window : (start, end), optional
        Inclusive window used to estimate λ when *transform* starts with
        “boxcox”.  Ignored otherwise.
    ids : iterable of str, optional
        Subset of `unique_id`s to include.  Defaults to all.
    plot_range : pd.DatetimeIndex, optional
        Restrict the plot to this date range. If None, uses the full range of `target_df['ds']`.
    verbose : bool, default False
        Print a λ table when estimating Box-Cox parameters.

    Returns
    -------
    dict
        ``{unique_id: lambda}``; empty if no Box-Cox used.
    """
    # Validate ids
    available_ids = set(target_df["unique_id"].unique())
    ids = list(ids) if ids is not None else sorted(available_ids)
    missing = set(ids) - available_ids
    if missing:
        raise ValueError(
            f"The following ids are not in target_df: {sorted(missing)}. "
            f"Available ids: {sorted(available_ids)}"
        )
    target_df = target_df[target_df["unique_id"].isin(ids)].copy()

    # Estimate Box-Cox lambdas if requested
    lambdas: Dict[str, float] = {}
    if transform.startswith("boxcox"):
        if lambda_window is None:
            raise ValueError("lambda_window must be provided for Box-Cox transforms.")
        if transform == "boxcox_winter" and not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        start, end = lambda_window
        lambda_df = target_df[(target_df["ds"] >= start) & (target_df["ds"] <= end)]
        if lambda_df.empty: 
            raise ValueError("lambda_window contains no data.")
        lambdas = get_lambdas(
            df=lambda_df,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            make_is_winter=make_is_winter
        )
        transform_name = "boxcox"
        if verbose:
            print("Estimated Box-Cox λ:")
            display(pd.DataFrame(lambdas.items(), columns=["unique_id", "lambda"]))
    else:
        transform_name = transform

    # Apply transform and restrict to plot range
    if plot_range is None: plot_range = target_df["ds"].sort_values().unique()
    plot_df = target_df[target_df["ds"].isin(plot_range)].copy()
    fwd = make_transformer(transform_name, "y", lambdas or None, inv=False)
    plot_df["y_transformed"] = transform_column(plot_df, fwd)

    # Keep only winter rows
    if only_cold_months:
        if not callable(make_is_winter):
            raise TypeError("make_is_winter must be a callable function.")
        is_winter_fn = {uid: make_is_winter(uid) for uid in plot_df["unique_id"].unique()}
        winter_mask = (
            plot_df.groupby("unique_id")["ds"]
            .transform(lambda s: s.map(is_winter_fn[s.name]))    # s.name is the uid
        )
        plot_df = plot_df[winter_mask]

    # Hour-of-week + month columns
    plot_df["hour_of_week"] = (
        plot_df["ds"].dt.dayofweek * 24 + plot_df["ds"].dt.hour
    )
    base_date = pd.Timestamp('2023-01-02')  # Monday
    plot_df["hour_of_week_datetime"] = base_date + pd.to_timedelta(plot_df["hour_of_week"], unit='h')
    plot_df["month"] = plot_df["ds"].dt.month

    # Fixed month palette
    month_palette = dict(zip(range(1, 13), sns.color_palette("crest", 12)))

    # Plot
    n_ids = len(ids)
    fig, axes = plt.subplots(nrows=n_ids, ncols=1, figsize=(11, 3 * n_ids), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (uid, grp) in zip(axes, plot_df.groupby("unique_id")):
        sns.lineplot(
            data=grp,
            x="hour_of_week",
            y="y_transformed",
            hue="month",
            palette=month_palette,
            legend=False,
            ax=ax,
            estimator="mean",
            errorbar=("pi", 80),
        )

        # Inline month labels at the last point (hour 167)
        for m, sub in grp.groupby("month"):
            last = sub.loc[sub["hour_of_week"].idxmax()]
            ax.text(
                last["hour_of_week"],
                last["y_transformed"],
                str(m),
                ha="left",
                fontsize=15,
                alpha=0.9,
                color=month_palette[m],
            )

        ax.set_title(uid)
        ax.set_ylabel("")
        ax.set_xlabel("")
        day_ticks = np.arange(0, 168 + 24, 24)
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""]
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels)

        # set minor ticks every 2 hours
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(axis="x", which="minor", length=4, width=1)

    # Shared labels and formatting
    fig.suptitle('Heat Demand for each month in the given period', fontsize=18)
    fig.supxlabel('Day of Week', fontsize=14)
    fig.supylabel('Heat Demand [kWh]', fontsize=14)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    return lambdas

def scatter_temp_vs_target_hourly(
    target_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    transform: str = "none",
    is_winter: Optional[Callable[[pd.Timestamp], bool]] = None,
    one_only: Optional[str] = None,  # 'Winter', 'Non-Winter', or None for both
    alphas: dict[str, float] = {"Winter": 0.6, "Non-Winter": 0.6},            
    id: Optional[str] = None,
    interactive: bool = True,
) -> pd.DataFrame:
    """
    Plot transformed heat demand against temperature, bucketed **by hour of day**.

    Parameters
    ----------
    target_df, aux_df : pd.DataFrame
        Must contain 'ds' and the relevant 'y' / 'temperature' columns.
    date_range : pd.DatetimeIndex
        Slice of dates to include in the analysis.
    transform : str, default 'none'
        Name of transformation to apply to 'y'. Options: 'none', 'log', 'boxcox', 
        'boxcox_winter', 'arcsinh', 'arcsinh2', 'arcsinh10'.
    is_winter : Callable[[pd.Timestamp], bool], default None
        Callable that returns True if a date is in winter, False otherwise.
        It's used to filter the data into winter and non-winter buckets and 
        for the Box-Cox transformation if `transform` is 'boxcox_winter'.
        If None, it will be created using 
        `heat_forecast.utils.transforms.make_is_winter(id)`.
    one_only : str, optional
        If 'Winter', only winter buckets are plotted; if 'Non-Winter', only non
        winter buckets are plotted; if None, both are plotted.
    alphas : dict[str, float], default {"Winter": 0.6, "Non-Winter": 0.6}
        Opacity for each bucket in the scatter plot.
    ids : Optional[str], default None
        If provided, filter target_df and aux_df to only that unique_id.
        If None, ensures both DataFrames have the same single unique_id
        and selects it. 
    interactive : bool, default True
        If True, 

    Returns
    -------
    pd.DataFrame
       Dataframe with correlations for each bucket.
       Columns: 0 to 23 (hours).
       Index: 'Winter' or 'Non-Winter' or both depending on `one_only`.
    """
    # ━━ Filter data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if id is not None:
        if not isinstance(id, str):
            raise TypeError("id must be a string representing the unique_id.")
        if id not in target_df["unique_id"].unique():
            raise ValueError(f"Provided id {id} is not present in target_df.")
        if id not in aux_df["unique_id"].unique():
            raise ValueError(f"Provided id {id} is not present in aux_df.")
        target_df = target_df[target_df["unique_id"]==id].copy()
        aux_df  = aux_df [aux_df ["unique_id"]==id].copy()
    else:
        ids = target_df["unique_id"].unique().tolist()
        ids_aux = aux_df["unique_id"].unique().tolist()
        if len(ids) != 1 or len(ids_aux) != 1:
            raise ValueError("No id provided and target_df or aux_df have multiple unique_ids.")
        if set(ids)!=set(ids_aux):
            raise ValueError("target_df and aux_df have different unique_ids.")
        id = ids[0]  # use the single unique_id from both DataFrames
    if id not in ['F1', 'F2', 'F3', 'F4', 'F5']:
        raise ValueError(f"Invalid id: {id}. Expected one of ['F1', 'F2', 'F3', 'F4', 'F5'].")

    heat_train = target_df.loc[target_df["ds"].isin(date_range)].copy()
    aux_train  = aux_df .loc[aux_df ["ds"].isin(date_range)].copy()

    # ━━ Make sure is_winter exists ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if is_winter is None:
        is_winter = make_is_winter(id)
    
    # ━━ Transform ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if transform not in ["none", "log", "boxcox", "boxcox_winter", "arcsinh", "arcsinh2", "arcsinh10"]:
        raise ValueError(
            f"Invalid transform: {transform}. "
            "Valid options are 'none', 'log', 'boxcox', 'boxcox_winter', "
            "'arcsinh', 'arcsinh2', or 'arcsinh10'."
        )

    if transform.startswith("boxcox"):
        lambdas = get_lambdas(
            heat_train_df=heat_train,
            method="loglik",
            winter_focus=(transform == "boxcox_winter"),
            is_winter=is_winter,
        )
        TRANSFORM = "boxcox"
    else:
        TRANSFORM = transform
        lambdas = None
    fwd = make_transformer(TRANSFORM, "y", lambdas, inv=False)
    heat_train["y_transformed"] = transform_column(heat_train, fwd)
    heat_train.drop(columns=["y"], inplace=True)  # drop the old target column

    # ━━ Flags & hour column ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    heat_train["is_winter"] = heat_train["ds"].apply(is_winter)
    aux_train ["is_winter"] = aux_train ["ds"].apply(is_winter)
    heat_train["hour"] = heat_train["ds"].dt.hour
    aux_train ["hour"] = aux_train ["ds"].dt.hour

    # ━━ Bucket definitions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    buckets_heat: Mapping[str, pd.Series] = {}
    buckets_aux : Mapping[str, pd.Series] = {}
    buckets_alphas: Mapping[str, float] = {}

    hours = range(24)
    if one_only=='Winter':
        for h in hours:   # 48 buckets
            key = f"Winter/H{h:02d}"
            mask = (heat_train["is_winter"] == True) & (heat_train["hour"] == h)
            buckets_heat[key] = mask
            buckets_aux [key] = (aux_train["is_winter"] == True) & (aux_train ["hour"] == h)
            buckets_alphas[key] = alphas.get('Winter', 0.6)  # default alpha if not specified
    elif one_only=='Non-Winter':
        for h in hours:
            key = f"Non-Winter/H{h:02d}"
            mask = (heat_train["is_winter"] == False) & (heat_train["hour"] == h)
            buckets_heat[key] = mask
            buckets_aux [key] = (aux_train["is_winter"] == False) & (aux_train ["hour"] == h)
            buckets_alphas[key] = alphas.get('Non-Winter', 0.6)
    elif one_only is None:
        for h in hours:   # 48 buckets
            for season, flag in [("Winter", True), ("Non-Winter", False)]:
                key = f"{season}/H{h:02d}"
                mask = (heat_train["is_winter"] == flag) & (heat_train["hour"] == h)
                buckets_heat[key] = mask
                buckets_aux [key] = (aux_train["is_winter"] == flag) & (aux_train ["hour"] == h)
                buckets_alphas[key] = alphas.get(season, 0.6)  # default alpha if not specified
    else:
        raise ValueError("one_only must be 'Winter', 'Non-Winter', or None.")

    # ━━ Colour palette (one distinct colour per *hour*) ━━━━━━━━━━━━━━━━━━━━━━
    cmap = plt.colormaps.get_cmap("hsv")  # cyclical palette
    hour_colours = {h: cmap(h / 24) for h in hours}  # normalize to [0,1] range

    def colour_for(key: str) -> tuple:
        # extract hour substring and map to colour
        h = int(key.split("H")[1])
        return hour_colours[h]

    # ━━ Plot ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if interactive: 
        fig = go.Figure()
        correlations: dict[str, float] = {}

        for key in buckets_heat.keys():
            x = aux_train .loc[buckets_aux [key], "temperature"]
            y = heat_train.loc[buckets_heat[key], "y_transformed"]
            alpha = buckets_alphas[key]
            ds = heat_train.loc[buckets_heat[key], "ds"]

            # skip empty buckets
            if x.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=key,
                    marker=dict(
                        size=6,
                        color=colors.to_hex(colour_for(key)),         
                        opacity=alpha
                    ),
                    # Pass ds via customdata so we can reference it in hovertemplate
                    customdata=ds.dt.strftime("%Y-%m-%d %H:%M"),
                    hovertemplate=(
                        "Temperature: %{x:.2f}<br>"
                        "Heat (transf.): %{y:.2f}<br>"
                        "ds: %{customdata}<extra></extra>"
                    ),
                )
            )

            correlations[key] = np.corrcoef(x, y)[0, 1]

            if len(x) > 1:
                β1, β0 = np.polyfit(x, y, 1)
                x_grid = np.linspace(x.min(), x.max(), 50)
                y_grid = β1 * x_grid + β0
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=y_grid,
                        mode="lines",
                        line=dict(
                            color=colors.to_hex(colour_for(key)),
                            width=2
                        ),
                        opacity=alpha,
                        showlegend=False                    # don’t add a second item to legend
                    )
                )

        fig.update_layout(
            title="Heat Demand vs Temperature — hourly buckets",
            xaxis_title="Temperature",
            yaxis_title="Transformed Heat Demand",
            legend_title="Season / Hour",
            template="plotly_white",
            width=900,
            height=600,
        )

        fig.show() 
        
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        correlations: dict[str, float] = {}

        for key in buckets_heat.keys():
            x = aux_train.loc[buckets_aux[key], "temperature"]
            y = heat_train.loc[buckets_heat[key], "y_transformed"]
            alpha = buckets_alphas[key]
            ds = heat_train.loc[buckets_heat[key], "ds"]

            if x.empty:
                continue

            color = colors.to_hex(colour_for(key))

            # --- Correlation ---
            correlations[key] = np.corrcoef(x, y)[0, 1]

            # --- Regression line ---
            if len(x) > 1:
                β1, β0 = np.polyfit(x, y, 1)
                x_grid = np.linspace(x.min(), x.max(), 50)
                y_grid = β1 * x_grid + β0
                ax.plot(
                    x_grid,
                    y_grid,
                    color=color,
                    linewidth=2,
                    alpha=alpha
                )

            # --- Scatter points ---
            ax.scatter(
                x,
                y,
                label=key,
                color=color,
                alpha=alpha,
                s=30,
                linewidth=0.3
            )

        # --- Aesthetics ---
        ax.set_title("Heat Demand vs Temperature — Hourly Buckets")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Transformed Heat Demand")

        # Get all legend entries
        handles, labels = ax.get_legend_handles_labels()

        # Select one each two
        selected_handles = handles[::2]
        selected_labels = labels[::2]

        # Add filtered legend
        ax.legend(
            selected_handles,
            selected_labels,
            title="Season / Hour",
            fontsize=10,
            title_fontsize=11,
            loc='upper right'
        )

        plt.tight_layout()
        plt.show()

    # ━━ Create a dataframe with correlations ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    corr_df = pd.DataFrame(correlations, index=[0]).T.rename(columns={0: "Correlation"})
    corr_df = corr_df.reset_index().rename(columns={'index': 'Key'})
    corr_df['Season'] = corr_df['Key'].str.split('/').str[0]
    corr_df['Hour'] = corr_df['Key'].str.extract(r'H(\d+)').astype(int)
    pivot_corr = corr_df.pivot(index='Season', columns='Hour', values='Correlation')

    return pivot_corr
