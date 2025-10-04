import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional
from coreforecast.scalers import boxcox, boxcox_lambda, inv_boxcox
import logging
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


# -------------------------------------------------------------------------------
# FOR DATA TRANSFORMATION
# -------------------------------------------------------------------------------

def make_transformer(
    transform: Optional[str] = None,
    target_col: str = 'y',
    lambdas: Optional[Dict[str, float]] = None,
    inv: bool = False
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Return a *group-aware* transformation function.

    The returned closure ``_apply(group)`` is meant to be used with
    ``DataFrameGroupBy.apply`` and acts on a single grouped
    ``pandas.DataFrame`` (the slice produced by ``groupby``).  It performs
    either a forward transform (``inv=False``) or its inverse (``inv=True``)
    on ``group[target_col]``.

    Supported transforms
    --------------------
    - ``"boxcox"``         : Box-Cox power transform (requires a group-specific
                             lambda in *lambdas*).
    - ``"log"``            : Natural logarithm.
    - ``"arcsinh"``        : Inverse hyperbolic sine,               x → asinh(x).
    - ``"arcsinh2"``       : Scaled inverse hyperbolic sine,        x → asinh(x / 2).
    - ``"arcsinh10"``      : Scaled inverse hyperbolic sine,        x → asinh(x / 10).
    - ``"none"`` or *None* : Identity (no change).

    Parameters
    ----------
    target_col : str
        Column in each group that holds the numeric series to transform.
    transform : str, optional
        Name of the transformation to apply; must be one of the
        strings listed above.
    lambdas : dict[str, float], optional
        Mapping ``{group_name: λ}`` used only for the Box-Cox transform.
        Ignored for other transform types.  If *None*, an empty dict is
        assumed and a missing key will raise ``KeyError`` when Box-Cox
        is requested.
    inv : bool, default False
        If True, apply the inverse transform; otherwise apply the forward
        transform.

    Returns
    -------
    Callable[[pd.DataFrame], pd.Series]
        A function ready to be passed to ``DataFrameGroupBy.apply``.
        The returned series keeps the same index as the input group so it
        can be assigned back directly.

    Raises
    ------
    ValueError
        If *transform* is not one of the recognised keywords.

    Examples
    --------
    >>> lambdas = {'A': 0.3, 'B': -0.1}
    >>> fwd = make_transformer('boxcox', 'sales', lambdas)
    >>> df['sales_bc'] = df.groupby('store').apply(fwd)

    >>> inv_f = make_transformer('boxcox', 'sales_bc', lambdas, inv=True)
    >>> df['sales_orig'] = df.groupby('store').apply(inv_f)
    """

    if lambdas is None:
        lambdas = {}
    if transform is None:
        transform = "none"
        
    def _apply(group: pd.DataFrame) -> pd.Series:
        """
        Apply the chosen transformation (or its inverse) to a single group.

        Parameters
        ----------
        group : pd.DataFrame
            Slice of the original DataFrame produced by ``groupby``.
            When using Box-Cox, ``group.name`` must be a key in *lambdas*.

        Returns
        -------
        pd.Series
            Transformed (or back-transformed) series with the same index
            as the input *group*.
        """
        y = group[target_col].to_numpy()
        name = group.name

        # Define the transformation based on the specified type
        if not inv:
            if transform == "boxcox":
                y_t = boxcox(y, lambdas[name])
            elif transform == "log":
                y_t = np.log(y)
            elif transform == "arcsinh":
                y_t = np.arcsinh(y)
            elif transform == "arcsinh2":
                y_t = np.arcsinh(y / 2.0)
            elif transform == "arcsinh10":
                y_t = np.arcsinh(y / 10.0)
            elif transform == "none":
                y_t = y
            else:
                raise ValueError(f"Unknown transform passed to make_transformer: {transform}")
        else:
            if transform == "boxcox":
                y_t = inv_boxcox(y, lambdas[name])
            elif transform == "log":
                y_t = np.exp(y)
            elif transform == "arcsinh":
                y_t = np.sinh(y)
            elif transform == "arcsinh2":
                y_t = np.sinh(y) * 2.0
            elif transform == "arcsinh10":
                y_t = np.sinh(y) * 10.0
            elif transform == "none":
                y_t = y
            else:
                raise ValueError(f"Unknown transform passed to make_transformer: {transform}")

        return pd.Series(y_t, index=group.index)

    return _apply

# Helper to apply the transformer to a full DataFrame
def transform_column(
    df: pd.DataFrame,
    transformer: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    """
    Apply a *group-aware* transformer to each ``unique_id`` group
    and return the concatenated result.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data holding at least the column ``'unique_id'`` and whatever
        additional columns the supplied *transformer* expects.
    transformer : Callable[[pd.DataFrame], pd.Series]
        Function produced by ``make_transformer`` (or another
        transformer with the same signature).  It must accept a single
        grouped data frame and return a series aligned with that frame's
        index.

    Returns
    -------
    pandas.Series
        Transformed column, ready to be assigned back to *df*.

    Examples
    --------
    >>> tr = make_transformer('log', 'sales')
    >>> df['sales_log'] = transform_column(df, tr)
    """
    n_uids = df['unique_id'].nunique()
    result = df.groupby("unique_id", group_keys=False).apply(transformer, include_groups=False)
    if n_uids == 1: # in this case, result is a row of a DataFrame, we have to transform it to a Series
        result = result.iloc[0]  # Get the first (and only) row as a Series
    return result

def make_is_winter(
    unique_id: str,
) -> Callable[[pd.Timestamp], bool]:
    """
    Factory that builds an *extended-winter* predicate for a given series
    identifier.

    The function returns a closure ``is_winter(ts)`` that evaluates to
    ``True`` when the timestamp *ts* falls within the "winter" period
    defined for the selected ``unique_id`` and ``False`` otherwise.

    The extended winter is considered to be:

    * All of **December - March** (inclusive), **plus**  
    * 15 November → 30 November (late-autumn ramp-up)  

    The same definition currently applies to every allowed identifier
    (``'F1'`` … ``'F5'``), but separating them makes future per-ID
    customisation trivial.

    Parameters
    ----------
    unique_id : str
        Identifier of the series/asset (“F1” through “F5”).
        Case sensitive.

    Returns
    -------
    is_winter : Callable[[pandas.Timestamp], bool]
        A one-argument function that returns ``True`` if its timestamp
        argument is in the extended winter season, otherwise ``False``.

    Examples
    --------
    >>> is_winter_F1 = make_is_winter('F1')
    >>> is_winter_F1(pd.Timestamp('2025-02-10'))
    True
    >>> is_winter_F1(pd.Timestamp('2025-07-10'))
    False
    """
    if unique_id not in ['F1', 'F2', 'F3', 'F4', 'F5']:
        raise ValueError(f"Unknown unique_id: {unique_id}. Must be one of 'F1', 'F2', 'F3', 'F4', 'F5'.")
    
    def is_winter(ts: pd.Timestamp) -> bool:
        """
        Predicate to determine if a timestamp is in the (extended) winter season for the given unique_id.
        """
        match unique_id:
            case 'F1':
                return ts.month in (12, 1, 2, 3) or (ts.month == 11 and ts.day >= 15) 
            case 'F2':
                return ts.month in (12, 1, 2, 3) or (ts.month == 11 and ts.day >= 15) 
            case 'F3':
                return ts.month in (12, 1, 2, 3) or (ts.month == 11 and ts.day >= 15) 
            case 'F4':
                return ts.month in (12, 1, 2, 3) or (ts.month == 11 and ts.day >= 15) 
            case 'F5':
                return ts.month in (12, 1, 2, 3) or (ts.month == 11 and ts.day >= 15) 
    
    return is_winter
   
def get_lambdas(
    df: pd.DataFrame,
    *,
    method: str = "loglik", # or "guerrero"
    winter_focus: bool = True,
    make_is_winter: Callable[[str], Callable[[pd.Timestamp], bool]] = make_is_winter,
    season_length: int = 365, # Default season length for Box-Cox transformation with guerrero method
    verbose: bool = False
) -> Dict[str, float]:
    """
    Estimate Box-Cox λ for every series identified by ``unique_id``.

    The function groups *df* by ``'unique_id'`` and calls
    ``statsforecast.utils.boxcox_lambda`` on each target vector ``y``.
    Optionally, it restricts the calculation to winter dates as defined by
    the user-supplied ``is_winter`` predicate.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns
        - ``'unique_id'`` — series identifier
        - ``'ds'``        — timestamp
        - ``'y'``         — numeric values to transform
    method : {'loglik', 'guerrero'}, default 'loglik'
        • ``'loglik'``   maximises the Box-Cox log-likelihood.  
        • ``'guerrero'`` minimises the within-season coefficient of variation
          (Guerrero, 1993) and therefore needs *season_length*.
    winter_focus : bool, default True
        When True, keep only dates for which ``is_winter(ds)`` returns True (where is winter could depend on the unique_id)
        before estimating λ; when False, use the full series.
    make_is_winter : Callable[[str], Callable[[pd.Timestamp], bool]], default make_is_winter
        Predicate that maps each unique_id to a callable that marks a timestamp as winter.  Required if
        *winter_focus* is True; ignored otherwise.
    season_length : int, default 365
        Seasonal period to pass to ``boxcox_lambda`` when *method* is
        ``'guerrero'``.

    Returns
    -------
    dict[str, float]
        Mapping ``{unique_id: λ}``.

    Examples
    --------
    >>> lambdas = get_lambdas(
    ...     train_df,
    ...     method='guerrero',
    ...     winter_focus=True,
    ...     season_length=365,
    ... )
    >>> lambdas['F1']
    0.18
    """
    if method not in ["loglik", "guerrero"]:
        raise ValueError("Method must be either 'loglik' or 'guerrero'")
    df = df.copy()

    # Ensure the DataFrame has the required columns
    if not all(col in df.columns for col in ['unique_id', 'ds', 'y']):
        raise ValueError("df must contain 'unique_id', 'ds', and 'y' columns")
    
    # Apply winter focus if requested
    if winter_focus and make_is_winter is None:
        raise ValueError("winter_focus=True but make_is_winter is None")
    if winter_focus:
        is_winter_fn = {uid: make_is_winter(uid) for uid in df["unique_id"].unique()}
        winter_mask = (
            df.groupby("unique_id")["ds"]
            .transform(lambda s: s.map(is_winter_fn[s.name]))    # s.name is the uid
        )
        df = df[winter_mask]

    def _to_apply(y: pd.Series) -> float:
        if method == 'guerrero':
            return boxcox_lambda(y.to_numpy(), method=method, season_length=season_length)
        return boxcox_lambda(y.to_numpy(), method=method)
    
    lambdas = (  
        df
        .groupby("unique_id")["y"]
        .apply(_to_apply)
        .to_dict()
    )

    if verbose:
        _LOGGER.info("Estimated Box-Cox λ values:")
        for uid, lam in lambdas.items():
            _LOGGER.info(f"λ for {uid}: {lam:.4f}")
    return lambdas