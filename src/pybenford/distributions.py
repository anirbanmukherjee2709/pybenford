"""Expected Benford distributions computed from first principles.

Every probability in this module derives from the **General Significant
Digit Law** (Nigrini Eq. 1.21, Hill 1995):

.. math::

    P(D_1 = d_1, \\ldots, D_k = d_k) = \\log_{10}\\!\\left(1 +
        \\frac{1}{\\sum_{i=1}^{k} d_i \\cdot 10^{\\,k-i}}\\right)

where :math:`d_1 \\in \\{1, \\ldots, 9\\}` and
:math:`d_j \\in \\{0, \\ldots, 9\\}` for :math:`j \\ge 2`.

The denominator is simply the integer formed by concatenating the
digits, so the joint first-:math:`k`-digits distribution collapses to
``log10(1 + 1/d)`` evaluated over ``d = 10**(k-1) .. 10**k - 1``. The
dedicated functions below (``first_digit_distribution`` etc.) are thin
wrappers that pick the appropriate ``k``; the one exception is
:func:`second_digit_distribution`, which marginalizes the joint
distribution over :math:`d_1` to obtain Nigrini Eq. 1.2.

All functions return ``float64`` NumPy arrays and are fully vectorized.
Outputs are not cached because the arrays are cheap to recompute
(microseconds for ``k <= 3``) and returning fresh arrays prevents
callers from accidentally mutating shared state.

References
----------
- Nigrini, M.J. (2012). *Benford's Law: Applications for Forensic
  Accounting, Auditing, and Fraud Detection*. Wiley. §1.
- Hill, T.P. (1995). "A Statistical Derivation of the Significant-Digit
  Law." *Statistical Science* 10(4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "benford_distribution",
    "digit_range",
    "first_digit_distribution",
    "first_three_digits_distribution",
    "first_two_digits_distribution",
    "second_digit_distribution",
    "third_digit_distribution",
]


def digit_range(k: int) -> NDArray[np.int64]:
    """Return the integer support of the first-:math:`k`-digits distribution.

    Parameters
    ----------
    k
        Number of leading digits. Must be ``>= 1``.

    Returns
    -------
    numpy.ndarray of int64
        The integers ``10**(k-1) .. 10**k - 1`` for ``k >= 2``, or
        ``1 .. 9`` for ``k == 1``. Length is ``9 * 10**(k-1)``.

    Raises
    ------
    ValueError
        If ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    start = 10 ** (k - 1) if k > 1 else 1
    stop = 10**k
    return np.arange(start, stop, dtype=np.int64)


def benford_distribution(k: int) -> NDArray[np.float64]:
    """Joint distribution of the first :math:`k` significant digits.

    Implements the General Significant Digit Law (Nigrini Eq. 1.21):

        ``P(d) = log10(1 + 1/d)`` for ``d = 10**(k-1), ..., 10**k - 1``.

    This is the master formula — every dedicated first-``k``-digit
    helper in this module is a thin wrapper over this function.

    Parameters
    ----------
    k
        Number of leading digits (``k = 1`` for first digit, ``k = 2``
        for first-two, etc.). Must be ``>= 1``.

    Returns
    -------
    numpy.ndarray of float64
        Probabilities aligned with :func:`digit_range` (``k``). The
        array sums to 1.0 up to floating-point error. Length is
        ``9 * 10**(k-1)``: 9, 90, 900, 9000, ...

    Raises
    ------
    ValueError
        If ``k < 1``.

    Notes
    -----
    Memory grows as ``9 * 10**(k-1)`` ``float64`` values, so ``k >= 8``
    is not practical. Realistic forensic use caps at ``k = 3``.

    Examples
    --------
    >>> benford_distribution(1).round(5)
    array([0.30103, 0.17609, 0.12494, 0.09691, 0.07918, 0.06695, 0.05799,
           0.05115, 0.04576])
    >>> float(benford_distribution(2).sum().round(10))
    1.0
    """
    d = digit_range(k).astype(np.float64)
    return np.log10(1.0 + 1.0 / d)


def first_digit_distribution() -> NDArray[np.float64]:
    """First-digit probabilities for ``d = 1..9`` (Nigrini Eq. 1.1).

    ``P(D_1 = d) = log10(1 + 1/d)``.

    Returns
    -------
    numpy.ndarray of float64
        Shape ``(9,)``. Index ``i`` corresponds to digit ``i + 1``;
        there is no leading-zero case.

    Examples
    --------
    >>> first_digit_distribution().round(5)
    array([0.30103, 0.17609, 0.12494, 0.09691, 0.07918, 0.06695, 0.05799,
           0.05115, 0.04576])
    """
    return benford_distribution(1)


def second_digit_distribution() -> NDArray[np.float64]:
    """Unconditional second-digit probabilities for ``d = 0..9``.

    Marginalizes the joint first-two-digits distribution over
    :math:`d_1` (Nigrini Eq. 1.2):

        ``P(D_2 = d_2) = sum_{d_1=1}^{9} log10(1 + 1 / (10*d_1 + d_2))``.

    Returns
    -------
    numpy.ndarray of float64
        Shape ``(10,)``. Index ``i`` is the probability that the
        second significant digit equals ``i``. Much closer to uniform
        than the first-digit distribution, with a residual bias toward
        low digits (Nigrini §4 covers the per-``d_1`` conditional form).

    Examples
    --------
    >>> second_digit_distribution().round(5)
    array([0.11968, 0.11389, 0.10882, 0.10433, 0.10031, 0.09668, 0.09337,
           0.09035, 0.08757, 0.085  ])
    """
    d1 = np.arange(1, 10, dtype=np.float64).reshape(9, 1)
    d2 = np.arange(0, 10, dtype=np.float64).reshape(1, 10)
    return np.log10(1.0 + 1.0 / (10.0 * d1 + d2)).sum(axis=0)


def third_digit_distribution() -> NDArray[np.float64]:
    """Unconditional third-digit probabilities for ``d = 0..9``.

    Marginalizes the joint first-three-digits distribution over
    :math:`d_1` and :math:`d_2` (Nigrini Table 1.2):

        ``P(D_3 = d_3) = sum_{d_1=1}^{9} sum_{d_2=0}^{9}
        log10(1 + 1 / (100*d_1 + 10*d_2 + d_3))``.

    Returns
    -------
    numpy.ndarray of float64
        Shape ``(10,)``. Index ``i`` is the probability that the
        third significant digit equals ``i``. The distribution is
        nearly uniform — values range from ``0.10178`` (digit 0) to
        ``0.09827`` (digit 9) — and from the fourth position onward
        the remaining bias is negligible for practical purposes.

    Examples
    --------
    >>> third_digit_distribution().round(5)
    array([0.10178, 0.10138, 0.10097, 0.10057, 0.10018, 0.09979, 0.0994 ,
           0.09902, 0.09864, 0.09827])
    """
    d1 = np.arange(1, 10, dtype=np.float64).reshape(9, 1, 1)
    d2 = np.arange(0, 10, dtype=np.float64).reshape(1, 10, 1)
    d3 = np.arange(0, 10, dtype=np.float64).reshape(1, 1, 10)
    return np.log10(1.0 + 1.0 / (100.0 * d1 + 10.0 * d2 + d3)).sum(axis=(0, 1))


def first_two_digits_distribution() -> NDArray[np.float64]:
    """First-two-digits probabilities for ``d = 10..99`` (Nigrini Eq. 1.3).

    ``P(D_1 D_2 = d) = log10(1 + 1/d)``.

    Nigrini's primary recommended test — more diagnostic than the
    first-digit test because the 90 bins catch bias patterns that the
    nine-bin first-digit histogram blurs out.

    Returns
    -------
    numpy.ndarray of float64
        Shape ``(90,)``. Index ``i`` is ``P(D_1 D_2 = i + 10)``.
    """
    return benford_distribution(2)


def first_three_digits_distribution() -> NDArray[np.float64]:
    """First-three-digits probabilities for ``d = 100..999``.

    ``P(D_1 D_2 D_3 = d) = log10(1 + 1/d)``.

    Returns
    -------
    numpy.ndarray of float64
        Shape ``(900,)``. Index ``i`` is ``P(D_1 D_2 D_3 = i + 100)``.
    """
    return benford_distribution(3)
