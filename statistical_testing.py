"""
Convenience functions for statistical testing.
"""

import numpy as np


def get_single_FT_surrogate(ts, seed=None):
    """
    Returns single 1D Fourier transform surrogate.

    Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
        (1992). Testing for nonlinearity in time series: the method of
        surrogate data. Physica D: Nonlinear Phenomena, 58(1-4), 77-94.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D FT surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    xf = np.fft.rfft(ts, axis=0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))[:, np.newaxis]
    # set the slowest frequency to zero, i.e. not to be randomised
    angle[0] = 0

    cxf = xf * np.exp(1j * angle)

    return np.fft.irfft(cxf, n=ts.shape[0], axis=0).squeeze()


def get_p_values(data_value, surrogates_value, tailed="upper"):
    """
    Return one-tailed or two-tailed values of percentiles with respect to
    surrogate testing. Two-tailed test assumes that the distribution of the
    test statistic under H0 is symmetric about 0.

    :param data_value: value(s) from data analyses
    :type data_value: np.ndarray
    :param surrogates_value: value(s) from surrogate data analyses, shape must
        be [num_surrogates, ...] where (...) is the shape of the data
    :type surrogates_value: np.ndarray
    :param tailed: which test statistic to compute: `upper`, `lower`, or `both`
    :type tailed: str
    :return: p-value of surrogate testing
    :rtype: float
    """
    assert data_value.shape == surrogates_value.shape[1:], (
        f"Incompatible shapes: data {data_value.shape}; surrogates "
        f"{surrogates_value.shape}"
    )
    num_surrogates = surrogates_value.shape[0]
    if tailed == "upper":
        significance = 1.0 - np.sum(
            np.greater_equal(data_value, surrogates_value), axis=0
        ) / float(num_surrogates)
    elif tailed == "lower":
        significance = np.sum(
            np.less_equal(data_value, surrogates_value), axis=0
        ) / float(num_surrogates)
    elif tailed == "both":
        significance = 2 * (
            1.0
            - np.sum(
                np.greater_equal(np.abs(data_value), surrogates_value), axis=0
            )
            / float(num_surrogates)
        )
    else:
        raise ValueError(f"Unknown tail for testing: {tailed}")

    return significance
