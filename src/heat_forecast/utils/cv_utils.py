import pandas as pd
import numpy as np
import datetime 
import logging
import warnings
from typing import Optional, Dict, Any
import math
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

# -------------------------------------------------------------------------------
# FUNCTIONS FOR CROSS-VALIDATION 
# -------------------------------------------------------------------------------

def _assert_hour_aligned(ts: pd.Timestamp, label: str) -> None:
    """
    Raise if *ts* is not aligned to the top of the hour.
    Works for any pandas.Timestamp.
    """
    if ts.minute or ts.second or ts.microsecond or ts.nanosecond:
        raise ValueError(
            f"{label} must be aligned to the hour — got {ts.strftime('%Y-%m-%d %H:%M:%S.%f')}"
        )

def get_cv_params(
    start_test_cv: pd.Timestamp,
    end_test_cv: pd.Timestamp,
    n_windows: int,
    horizon_hours: int = 24 * 7,
    start_forecasts_at_midnight: bool = True, # If true, each forecast is forced to start at midnight
    logger: Optional[logging.Logger] = None
) -> tuple[int, int, pd.Timestamp]:
    """
    Compute rolling-window cross-validation parameters.

    The helper returns:
    * step_size          - spacing between successive cut-off times 
                           (expressed in whole hours);  
    * test_hours_actual  - total length of the **evaluated** test span
                           after adjusting the last cut-off;  
    * end_test_actual    - timestamp of the final observation used
                           in cross-validation.

    The function validates the requested span and adjusts *step_size* so
    that exactly *n_windows* cut-offs fit between *start_test_cv* and
    *end_test_cv* while leaving at least *horizon_hours* after the final
    cut-off.  When *start_forecasts_at_midnight* is True, *step_size* is
    rounded down to the nearest multiple of 24 h to keep every forecast
    window aligned on midnight. These parameters are used to
    configure the cross-validation loop in ``statsforecast``.

    Parameters
    ----------
    start_test_cv : pandas.Timestamp
        First timestamp **included** in the test span.  Must be aligned to
        the hour and, when *start_forecasts_at_midnight* is True, equal to
        00:00:00.
    end_test_cv : pandas.Timestamp
        Last timestamp **included** in the test span (hour-aligned).
    n_windows : int
        Number of rolling windows (cut-offs).  Must be at least 1.
    horizon_hours : int, default ``24 * 7``
        Forecast horizon in hours.
    start_forecasts_at_midnight : bool, default ``True``
        If True, force every forecast window to start at 00:00; this makes
        *step_size* a multiple of 24 h.
    logger : logging.Logger, optional
        Logger to use for logging the parameters. If None, uses local logger.

    Returns
    -------
    tuple[int, int, pandas.Timestamp]
        ``(step_size, test_hours_actual, end_test_actual)``

        * step_size          : int, spacing between cut-offs in hours  
        * test_hours_actual  : int, length of the actual test span in hours  
        * end_test_actual    : pandas.Timestamp, final observation used for cross-validation
    """
    _assert_hour_aligned(start_test_cv, "start_test_cv")
    _assert_hour_aligned(end_test_cv, "end_test_cv")

    logger = logger or _LOGGER

    if start_forecasts_at_midnight and start_test_cv.time() != datetime.time(0):
        raise ValueError("start_test_cv must be exactly 00:00:00 when start_forecasts_at_midnight=True.")
    
    test_range = end_test_cv - start_test_cv + pd.Timedelta(hours=1)  # +1 to include the start hour in the test span
    if test_range <= pd.Timedelta(0):
        raise ValueError("end_test_cv must be after start_test_cv.")
    
    test_hours = int(test_range / pd.Timedelta(hours=1))
    if test_hours < horizon_hours:
        raise ValueError("Test span shorter than forecast horizon.")
    
    if n_windows < 1:
        raise ValueError("n_windows must be ≥ 1.")
    elif n_windows == 1:
        step_size = 1 # Else we get a division by zero error when running cv
    else:
        step_size = (test_hours - horizon_hours) // (n_windows - 1) # The -1 is because the first cutoff is before start train, and the last is (close to) the end
        if step_size < 1:
            raise ValueError("Too many windows for the chosen span & horizon.")
        if start_forecasts_at_midnight: 
            step_size = (step_size // 24) * 24  # Ensure step_size is a multiple of 24 hours if starting at midnight
            if step_size < 24:
                raise ValueError("step_size became <24 h after midnight-rounding; relax the midnight constraint or use fewer windows.")

    first_cutoff = start_test_cv - pd.Timedelta(hours=1)
    last_cutoff = first_cutoff + pd.Timedelta(hours=step_size * (n_windows - 1))
    end_test_actual = last_cutoff + pd.Timedelta(hours=horizon_hours)
    test_hours_actual = int((last_cutoff - first_cutoff) / pd.Timedelta(hours=1)) + horizon_hours

    step_size_days = step_size // 24
    step_size_remaining_hours = step_size % 24
    logger.info(
        "CV params: %d window%s, first cutoff %s, last cutoff %s%s.",
        n_windows,
        "" if n_windows == 1 else "s",
        first_cutoff,
        last_cutoff,
        "" if n_windows == 1 else f", step size {step_size_days}d {step_size_remaining_hours}h"
    )
    return step_size, test_hours_actual, end_test_actual

def get_cv_params_v2(
    start_test_cv: pd.Timestamp,
    end_test_cv: pd.Timestamp,
    n_windows: int | None = None,
    step_size: int | None = None,
    max_n_fits: int | None = None,
    horizon_hours: int = 24 * 7,
    start_forecasts_at_midnight: bool = True, # If true, each forecast is forced to start at midnight
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Compute cross-validation windowing and refit schedule for time-series backtesting.

    Given a desired inclusive test span [start_test_cv, end_test_cv], a forecast
    horizon (in hours), and either the number of windows or a stride between cutoffs,
    this function derives the actual step size, realized coverage, and (optionally)
    a model refitting schedule capped by max_n_fits.

    Parameters
    ----------
    start_test_cv : pandas.Timestamp
        Inclusive start of the requested test span. Must be aligned to the hour.
        If start_forecasts_at_midnight=True, it must be exactly 00:00:00.
    end_test_cv : pandas.Timestamp
        Inclusive end of the requested test span. Must be aligned to the hour and
        strictly after start_test_cv.
    n_windows : int or None, default None
        Number of forecast windows (i.e., cutoffs). Exactly one of n_windows or
        step_size must be provided. If n_windows == 1, a dummy step_size
        is chosen only to satisfy validation.
    step_size : int or None, default None
        Stride in hours between consecutive cutoffs. Exactly one of n_windows or
        step_size must be provided. When start_forecasts_at_midnight=True and
        more than one window fits, step_size must be a multiple of 24.
    max_n_fits : int or None, default None
        Optional cap on the total number of model fits across all windows.

        Semantics:
            - 1 → no refitting (refit=False), n_fits=1
            - > n_windows → refit every window (refit=1), n_fits=n_windows
            - otherwise → refit = ceil(n_windows / max_n_fits) and
            n_fits = ceil(n_windows / refit)
            - None → unspecified (output includes refit=None, n_fits=None)
    horizon_hours : int, default 24*7
        Forecast horizon length in hours. Must be ≥ 1.
    start_forecasts_at_midnight : bool, default True
        If True, the first forecast must start at midnight and step_size is
        rounded down to a multiple of  hours. Rounding can reduce the realized
        coverage relative to the requested span.
    logger : logging.Logger or None, default None
        Logger to use for informational messages. Falls back to a module logger.

    Returns
    -------
    dict
        A dictionary with the following keys:

        step_size : int
            Stride between cutoffs in hours (given or inferred; after any midnight rounding).

        test_hours : int
            Realized coverage of the CV test span in hours, computed as
            end_test_actual - start_test_cv + 1 hour. May be less than the requested
            span due to midnight rounding.

        end_test_actual : pandas.Timestamp
            Inclusive timestamp of the final hour covered by the last window
            (i.e., last_cutoff + horizon_hours).

        n_windows : int
            Number of windows actually used.

        refit : int or bool or None
            Refit policy:
                - None → unspecified (default: refit every window)
                - False → fit once (no refitting)
                - int k ≥ 1 → refit every k windows (windows with indices i
                where i % k == 0)

        n_fits : int or None
            Total number of fits implied by refit:
                - 1 when refit is False
                - n_windows when refit == 1
                - ceil(n_windows / refit) when refit is an int ≥ 1
                - n_windows when refit is None
    """
    _assert_hour_aligned(start_test_cv, "start_test_cv")
    _assert_hour_aligned(end_test_cv, "end_test_cv")

    logger = logger or _LOGGER

    if not isinstance(horizon_hours, int) or horizon_hours < 1:
        raise ValueError("horizon_hours must be an integer ≥ 1.")
    if start_forecasts_at_midnight and start_test_cv.time() != datetime.time(0):
        raise ValueError("start_test_cv must be exactly 00:00:00 when start_forecasts_at_midnight=True.")
    
    test_range = end_test_cv - start_test_cv + pd.Timedelta(hours=1)  # +1 to include the start hour in the test span
    if test_range <= pd.Timedelta(0):
        raise ValueError("end_test_cv must be after start_test_cv.")
    
    test_hours = int(test_range / pd.Timedelta(hours=1))
    if test_hours < horizon_hours:
        raise ValueError("Test span shorter than forecast horizon.")

    if (step_size is None) == (n_windows is None):
        raise ValueError("Exactly one of step_size or n_windows must be provided.")

    if n_windows is not None: # n_windows is given
        if (not isinstance(n_windows, int)) or n_windows < 1:
            raise ValueError("n_windows must be an integer ≥ 1.")
        elif n_windows == 1:
            # step_size is irrelevant for placement; pick a dummy that won't trip validations
            step_size = 24 if start_forecasts_at_midnight else 1
        else:
            step_size = (test_hours - horizon_hours) // (n_windows - 1) # The -1 is because the first cutoff is before start train, and the last is (close to) the end
            if step_size < 1:
                raise ValueError("Too many windows for the chosen span & horizon.")
            if start_forecasts_at_midnight: 
                step_size = (step_size // 24) * 24  # Ensure step_size is a multiple of 24 hours if starting at midnight
                if step_size < 24:
                    raise ValueError("step_size became <24 h after midnight-rounding; relax the midnight constraint or use fewer windows.")
    
    else: # step_size is given
        if (not isinstance(step_size, int)) or step_size < 1:
            raise ValueError("step_size must be an integer ≥ 1.")
        n_windows = (test_hours - horizon_hours) // step_size + 1
        if start_forecasts_at_midnight and step_size % 24 != 0 and n_windows > 1:
            raise ValueError("step_size must be a multiple of 24 hours when start_forecasts_at_midnight=True.")
        if n_windows == 1:
            logger.warning("Only one window fits in the chosen span with the given step_size & horizon.")

    first_cutoff = start_test_cv - pd.Timedelta(hours=1)
    last_cutoff = first_cutoff + pd.Timedelta(hours=step_size * (n_windows - 1))
    end_test_actual = last_cutoff + pd.Timedelta(hours=horizon_hours)
    test_hours_actual = int((last_cutoff - first_cutoff) / pd.Timedelta(hours=1)) + horizon_hours

    if max_n_fits is not None:
        if (not isinstance(max_n_fits, int)) or max_n_fits < 1:
            raise ValueError("max_n_fits must be an integer ≥ 1.")
        elif max_n_fits == 1:
            refit = False
            n_fits_actual = 1
        elif max_n_fits > n_windows:
            refit = 1
            n_fits_actual = n_windows
        else:
            refit = math.ceil(n_windows / max_n_fits)
            n_fits_actual = math.ceil(n_windows / refit)
    else:
        refit = None
        n_fits_actual = n_windows

    step_size_days = step_size // 24
    step_size_remaining_hours = step_size % 24
    refit_str = f"" if refit is None else \
        ", no refitting" if not refit else \
        ", refit every window" if refit ==1 else \
        f", refit every {refit} windows" if refit > 1 else ""
    n_fits_str = "" if n_fits_actual is None else \
        f", total number of fits: {n_fits_actual}"
    logger.info(
        "CV params: %d window%s, first cutoff %s, last cutoff %s%s%s%s.",
        n_windows,
        "" if n_windows == 1 else "s",
        first_cutoff,
        last_cutoff,
        "" if n_windows == 1 else f", step size {step_size_days}d {step_size_remaining_hours}h",
        refit_str,
        n_fits_str
    )

    out = {
        "step_size": step_size,
        "test_hours": test_hours_actual,
        "end_test_actual": end_test_actual,
        "n_windows": n_windows,
        "refit": refit,
        "n_fits": n_fits_actual
    }
    return out

def display_info_cv(
        cv_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None
    ) -> None:
    """
    Print a one-screen summary of a rolling-origin cross-validation
    DataFrame and return the forecasting horizon length.

    Parameters
    ----------
    cv_df : pandas.DataFrame
        Rolling-origin CV table that **must** contain the columns

        * ``'unique_id'`` - series identifier
        * ``'ds'``        - forecast timestamps
        * ``'cutoff'``    - training-window end date
        * ``'y'``         - ground-truth values
        * one or more model forecast columns

        Prediction-interval columns (e.g. ``model-lo-95``) are allowed
        but are excluded from the *Models* line of the summary.
    
    logger : logging.Logger, optional
        Logger to use for logging the summary. If None, uses local logger.

    Returns
    -------

    """
    logger = logger or _LOGGER

    # Detect model-forecast columns (exclude PI bounds)
    model_cols = cv_df.columns.difference(['unique_id', 'ds', 'cutoff', 'y'])
    models = [c for c in model_cols if '-lo-' not in c and '-hi-' not in c]

    # Basic CV geometry
    cutoffs = np.sort(cv_df['cutoff'].unique())
    first_cutoff = cutoffs[0]
    last_cutoff = cutoffs[-1]

    if len(cutoffs) > 1:
        step_deltas = np.diff(cutoffs) # array of Timedelta
        step_days = np.array([d / pd.Timedelta(days=1) for d in step_deltas])
        unique_steps = np.unique(step_days)

        if unique_steps.size == 1:                           
            step_size = int(unique_steps[0]) if unique_steps[0].is_integer() else unique_steps[0]
        else:                                                
            step_size = list(step_days)                     
            warnings.warn(
                f"Cut-off windows are not equally spaced: {unique_steps}. Using the full list for reporting.",
                UserWarning, stacklevel=2
            )
    else:
        step_size = None

    horizon_lengths = (
        cv_df.groupby(['cutoff', 'unique_id'], sort=False)['ds'].nunique()
        .to_numpy()
    )
    unique_horizons = np.unique(horizon_lengths)

    if len(unique_horizons) == 1:
        horizon_return = int(unique_horizons[0])
    else:
        horizon_return = horizon_lengths.tolist()
        logger.warning(
            f"Windows have different horizon lengths: {unique_horizons}. Using the full list for reporting.",
            UserWarning, stacklevel=2
        )

    summary_data = {
        'Models': models,
        'Unique IDs': cv_df['unique_id'].unique().tolist(),
        'Horizon length (hours)': horizon_return,
        'Windows': len(cutoffs),
        'First cutoff': first_cutoff,
        'Last cutoff': last_cutoff,
        'Step size (hours)': step_size,
    } 

    def _formatter(value):
        if value is None:
            return "–"
        elif isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value))
        elif isinstance(value, (datetime.datetime, pd.Timestamp)):
            return value.strftime("%Y-%m-%d %H:%M")
        elif isinstance(value, np.datetime64):
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
        return str(value)

    log_lines = []
    for key, value in summary_data.items():
        value_str = _formatter(value)
        log_lines.append(f"{key:<24}: {value_str}")
    max_length = max(len(line) for line in log_lines)
    log_lines = [f"{'━' * max_length}"] + ["Cross-validation summary"] + \
        [f"{'─' * max_length}"] + log_lines + [f"{'━' * max_length}"]

    logger.info("\n" + "\n".join(log_lines))


def get_cv_params_for_test(
        unique_id: str,
        horizon_type: str,  # 'week' or 'day'
        max_n_fits: int | None = None,
        logger: Optional[logging.Logger] = None,
        final: bool = False
    ) -> tuple[int, int, pd.Timestamp, int]:
    """
    Convenience wrapper that selects a pre-defined CV plan for a given series
    and horizon type, then calls :func:`get_cv_params_v2`.

    It injects a preset ``n_windows`` and test-span (``start_test_cv``, ``end_test_cv``)
    based on ``unique_id`` and ``horizon_type``, delegates to ``get_cv_params_v2``,
    and returns its output.

    Parameters
    ----------
    unique_id : str
        Series identifier. Supported IDs: ``'F1'``, ``'F2'``, ``'F3'``, ``'F4'``, ``'F5'``.
    horizon_type : str
        Either ``'week'`` (``horizon_hours=168``) or ``'day'`` (``horizon_hours=24``).
    max_n_fits : int or None, default None
        Optional cap on total number of model fits (forwarded to ``get_cv_params_v2``).
    logger : logging.Logger or None, default None
        Logger to use for informational messages. Falls back to a module logger.
    final : bool, default False
        If True, use the settings for the final test phase; in this case,
        max_n_fits is ignored. If False (default), use the settings for the initial test phase.

    Returns
    -------
    dict
        The dictionary returned by :func:`get_cv_params_v2` (keys: ``'step_size'``,
        ``'test_hours'``, ``'end_test_actual'``, ``'n_windows'``, ``'refit'``,
        ``'n_fits'``), with ``'n_windows'`` explicitly set to the preset value
        for the chosen ``unique_id`` and ``horizon_type``.
    """
    # Validate
    if unique_id not in ['F1', 'F2', 'F3', 'F4', 'F5']:
        raise ValueError(f"Unknown unique_id: {unique_id}. Known IDs are: F1, ..., F5")
    if horizon_type not in ['week', 'day']:
        raise ValueError(f"Unknown horizon_type: {horizon_type}. Known types are: 'week', 'day'.")
    
    if not final:
        for_get_cv_params = {
            'F1': {
                'week': { # step_size = 9d
                    'n_windows': 21,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 37,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F2': {
                'week': { # step_size = 9d
                    'n_windows': 21,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 37,
                    'start_test_cv': pd.to_datetime('2024-10-20'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F3': {
                'week': { # step_size = 9d
                    'n_windows': 19,
                    'start_test_cv': pd.to_datetime('2024-10-28'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 35,
                    'start_test_cv': pd.to_datetime('2024-10-28'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F4': {
                'week': { # step_size = 9d
                    'n_windows': 18,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 32,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
            'F5': {
                'week': { # step_size = 9d
                    'n_windows': 18,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
                'day': { # step_size = 5d
                    'n_windows': 33,
                    'start_test_cv': pd.to_datetime('2024-11-10'),
                    'end_test_cv': pd.to_datetime('2025-04-30'),
                },
            },
        },
        params = for_get_cv_params[unique_id][horizon_type]
    else:
        for_get_cv_params = {
            'day': {
                'start_test_cv': pd.to_datetime('2024-05-20'),
                'end_test_cv': pd.to_datetime('2025-05-20'),
                'step_size': 24,
                'max_n_fits': 53,
            },
            'week': {
                'start_test_cv': pd.to_datetime('2024-05-20'),
                'end_test_cv': pd.to_datetime('2025-05-20'),
                'step_size': 24,
                'max_n_fits': 52,
            }
        }
        params = for_get_cv_params[horizon_type]
    out = get_cv_params_v2(
            start_test_cv=params['start_test_cv'],
            end_test_cv=params['end_test_cv'],
            n_windows=params.get('n_windows', None),
            step_size=params.get('step_size', None),
            max_n_fits=params.get('max_n_fits', max_n_fits),
            horizon_hours=168 if horizon_type == 'week' else 24,
            logger=logger
        )
    return out

