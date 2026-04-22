"""Vectorized digit-extraction primitives.

Every extractor here works from the **logarithmic significand** of
``|x|`` rather than the string representation of the number. This
sidesteps the scale-dependent bugs that string slicing introduces
(e.g. ``str(0.0529)[0:1] == "0"`` when the first significant digit is
really 5), and it runs vectorized on arbitrarily large NumPy arrays.

For any positive finite ``x`` define the significand

    ``m(x) = |x| / 10 ** floor(log10(|x|))``

so that ``m(x) in [1, 10)``. From that single quantity every
positional digit falls out as a simple floor/mod of ``m * 10**k``:

=================== ==============================
Quantity             Expression
=================== ==============================
First digit          ``floor(m)``
Second digit         ``floor(m * 10) mod 10``
Third digit          ``floor(m * 100) mod 10``
First-two digits     ``floor(m * 10)``
First-three digits   ``floor(m * 100)``
Mantissa             ``log10(|x|) - floor(log10(|x|))``
Collapsed (DF)       ``m * 10`` (two digits left of decimal)
=================== ==============================

Invalid inputs — zero, negative-zero, ``NaN``, ``+/-inf`` — are mapped
to ``NaN`` in the returned array so that shape and alignment with the
input are preserved. Negative values are handled by taking the absolute
value first (Benford's Law is sign-invariant). The caller is
responsible for any stratification of positives vs. negatives before
calling these functions, per Nigrini §4.3.

Because integer NumPy dtypes cannot represent ``NaN``, every extractor
returns ``float64`` even when the valid entries are whole numbers. To
recover integer digits, filter with ``~np.isnan(result)`` and cast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "collapse_numbers",
    "extract_first_digit",
    "extract_first_three_digits",
    "extract_first_two_digits",
    "extract_last_two_digits",
    "extract_mantissa",
    "extract_second_digit",
    "extract_third_digit",
]


def _significand(values: ArrayLike) -> NDArray[np.float64]:
    """Return ``|x| / 10**floor(log10(|x|))`` in ``[1, 10)``, ``NaN`` when invalid.

    Private helper — every positional digit extractor below is a
    floor/mod of this quantity.
    """
    abs_arr = np.abs(np.asarray(values, dtype=np.float64))
    valid = np.isfinite(abs_arr) & (abs_arr > 0.0)
    result = np.full(abs_arr.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return result

    v = abs_arr[valid]
    order = np.floor(np.log10(v))
    m = v / (10.0 ** order)
    # Floating-point drift at exact powers of 10 or very large magnitudes
    # can push m just below 1 or at/above 10. Clamp back to [1, 10).
    below = m < 1.0
    m[below] *= 10.0
    above = m >= 10.0
    m[above] /= 10.0

    result[valid] = m
    return result


def extract_first_digit(values: ArrayLike) -> NDArray[np.float64]:
    """Leading significant digit of each value.

    Parameters
    ----------
    values
        Array-like of real numbers. Sign is discarded.

    Returns
    -------
    numpy.ndarray of float64
        Same shape as ``values``. Valid entries are whole numbers in
        ``{1, 2, ..., 9}``; zeros, ``NaN``, and ``+/-inf`` inputs map
        to ``NaN``.

    Examples
    --------
    >>> extract_first_digit([6340, -0.0529, 0, np.nan, 110364])
    array([ 6.,  5., nan, nan,  1.])
    """
    return np.floor(_significand(values))


def extract_second_digit(values: ArrayLike) -> NDArray[np.float64]:
    """Second significant digit of each value.

    Returns a float array whose valid entries lie in ``{0, 1, ..., 9}``.
    Invalid inputs map to ``NaN``.

    Examples
    --------
    >>> extract_second_digit([6340, 0.0529, 110364])
    array([3., 2., 1.])
    """
    m = _significand(values)
    return np.floor(m * 10.0) % 10.0


def extract_third_digit(values: ArrayLike) -> NDArray[np.float64]:
    """Third significant digit of each value.

    Returns a float array whose valid entries lie in ``{0, 1, ..., 9}``.
    Invalid inputs map to ``NaN``. Unlike the notebook's
    ``count_third_digit`` helper, this does *not* silently drop
    numbers that happen to have fewer than three integer-part digits —
    the third *significant* digit is well-defined for any non-zero
    finite value.

    Examples
    --------
    >>> extract_third_digit([6340, 0.0529, 110364])
    array([4., 9., 0.])
    """
    m = _significand(values)
    return np.floor(m * 100.0) % 10.0


def extract_first_two_digits(values: ArrayLike) -> NDArray[np.float64]:
    """Leading two significant digits as a single integer in ``[10, 99]``.

    Valid entries are whole numbers between 10 and 99 inclusive;
    invalid inputs map to ``NaN``.

    Examples
    --------
    >>> extract_first_two_digits([6340, 0.0529, 110364])
    array([63., 52., 11.])
    """
    m = _significand(values)
    return np.floor(m * 10.0)


def extract_first_three_digits(values: ArrayLike) -> NDArray[np.float64]:
    """Leading three significant digits as a single integer in ``[100, 999]``.

    Valid entries are whole numbers between 100 and 999 inclusive;
    invalid inputs map to ``NaN``.

    Examples
    --------
    >>> extract_first_three_digits([6340, 0.0529, 110364])
    array([634., 529., 110.])
    """
    m = _significand(values)
    return np.floor(m * 100.0)


def extract_last_two_digits(values: ArrayLike) -> NDArray[np.float64]:
    """Last two digits of the integer part of ``|x|``.

    ``|x|`` is rounded to the nearest integer before the modulo, so a
    value like ``-1234.7`` yields ``35`` (``round(1234.7) = 1235``).
    Invalid inputs map to ``NaN``.

    Used by the Last-Two Digits test (Nigrini §3.5) to flag rounding,
    fabrication, and threshold effects. Expected distribution is
    uniform at ``1/100`` per bin — see
    :data:`~pybenford.constants.LAST_TWO_DIGITS_EXPECTED`.

    Notes
    -----
    If the data of interest carries sub-unit precision (e.g. currency
    with cents), scale by ``100`` before calling so the cents become
    the "last two digits". This matches the convention in the
    reference notebook.

    Examples
    --------
    >>> extract_last_two_digits([1234, -56789.4, 5, 0, np.nan])
    array([34., 89.,  5., nan, nan])
    """
    abs_arr = np.abs(np.asarray(values, dtype=np.float64))
    valid = np.isfinite(abs_arr) & (abs_arr > 0.0)
    result = np.full(abs_arr.shape, np.nan, dtype=np.float64)
    if valid.any():
        result[valid] = np.rint(abs_arr[valid]) % 100.0
    return result


def extract_mantissa(values: ArrayLike) -> NDArray[np.float64]:
    """Fractional part of ``log10(|x|)`` — the Benford mantissa.

    Defined for any positive finite value as

        ``mantissa(x) = log10(|x|) - floor(log10(|x|))``

    which equals ``log10(significand)``. Valid entries lie in
    ``[0, 1)``; invalid inputs map to ``NaN``.

    Used by the Mantissa Arc Test and the Ordered Mantissa Plot
    (Nigrini §3.6–3.7). A Benford-conforming set has mantissas
    uniformly distributed on ``[0, 1)`` — the ordered plot is a
    straight line, and the Arc Test's mean vector sits near the origin.

    Examples
    --------
    >>> extract_mantissa([1.0, 10.0, 100.0]).round(10)
    array([0., 0., 0.])
    >>> extract_mantissa([2.0]).round(5)
    array([0.30103])
    """
    abs_arr = np.abs(np.asarray(values, dtype=np.float64))
    valid = np.isfinite(abs_arr) & (abs_arr > 0.0)
    result = np.full(abs_arr.shape, np.nan, dtype=np.float64)
    if valid.any():
        log_v = np.log10(abs_arr[valid])
        result[valid] = log_v - np.floor(log_v)
    return result


def collapse_numbers(values: ArrayLike) -> NDArray[np.float64]:
    """Collapse each value to two digits left of the decimal.

    Implements Nigrini Eq. 6.1 of the Distortion Factor model:

        ``collapsed(x) = |x| / 10 ** (floor(log10(|x|)) - 1)``

    Valid outputs lie in ``[10, 100)``; invalid inputs map to ``NaN``.

    The collapse normalizes every value to the same scale so their
    mean can be compared against :data:`~pybenford.constants.DF_EXPECTED_MEAN`
    (``90 / ln(10) ≈ 39.0865``) to gauge systematic over- or
    understatement.

    Notes
    -----
    Nigrini §6.1 instructs callers to **filter out all numbers < 10
    (including negatives) before running the DF model**. The math here
    is well-defined for any non-zero finite value, but the statistical
    interpretation only holds on the filtered stratum — do the
    filtering upstream.

    Examples
    --------
    >>> collapse_numbers([6340, 0.0529, 110364]).round(4)
    array([63.4   , 52.9   , 11.0364])
    """
    return _significand(values) * 10.0
