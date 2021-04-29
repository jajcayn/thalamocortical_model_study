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


def get_single_time_shift_surrogate(ts, seed=None):
    """
    Return single 1D time shift surrogate. Timeseries is shifted in time
    (assumed periodic in the sense that is wrapped in the end) by a random
    amount of time. Useful surrogate for testing phase relationships.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D time shift surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    roll = np.random.choice(ts.shape[0], 1)[0]
    # assert roll is not 0 - would give the same timeseries
    while roll == 0:
        roll = np.random.choice(ts.shape[0], 1)[0]
    return np.roll(ts, roll, axis=0)


def get_single_shuffle_surrogate(ts, cut_points=None, seed=None):
    """
    Return single 1D shuffle surrogate. Timeseries is cut into `cut_points`
    pieces at random and then shuffled. If `cut_points` is None, will cut each
    point, hence whole timeseries is shuffled.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param cut_points: number of cutting points, timeseries will be partitioned
        into n+1 partitions with n cut_points; if None, each point is its own
        partition, i.e. classical shuffle surrogate
    :type cut_points: int|None
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D shuffle surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    if cut_points is None:
        cut_points = ts.shape[0]
    assert (
        cut_points <= ts.shape[0]
    ), "Cannot have more cut points than length of the timeseries"
    # generate random partition points without replacement
    partion_points = np.sort(
        np.random.choice(ts.shape[0], cut_points, replace=False)
    )

    def split_permute_concat(x, split_points):
        """
        Helper that splits, permutes and concats the timeseries.
        """
        return np.concatenate(
            np.random.permutation(np.split(x, split_points, axis=0))
        )

    current_permutation = split_permute_concat(ts, partion_points)
    # assert we actually permute the timeseries, useful when using only one
    # cutting point, i.e. two partitions so they are forced to swap
    while np.all(current_permutation == ts):
        current_permutation = split_permute_concat(ts, partion_points)
    return current_permutation


def get_single_AAFT_surrogate(ts, seed=None):
    """
    Returns single 1D amplitude-adjusted Fourier transform surrogate.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D AAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    # create Gaussian data
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    gaussian = np.broadcast_to(
        np.random.randn(ts.shape[0])[:, np.newaxis], ts.shape
    )
    gaussian = np.sort(gaussian, axis=0)
    # rescale data to Gaussian distribution
    ranks = ts.argsort(axis=0).argsort(axis=0)
    rescaled_data = np.zeros_like(ts)
    for i in range(ts.shape[1]):
        rescaled_data[:, i] = gaussian[ranks[:, i], i]
    # do phase randomization
    phase_randomized_data = get_single_FT_surrogate(rescaled_data, seed=seed)
    if phase_randomized_data.ndim == 1:
        phase_randomized_data = phase_randomized_data[:, np.newaxis]
    # rescale back to amplitude distribution of original data
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)
    ranks = phase_randomized_data.argsort(axis=0).argsort(axis=0)

    for i in range(ts.shape[1]):
        rescaled_data[:, i] = sorted_original[ranks[:, i], i]

    return rescaled_data.squeeze()


def get_single_IAAFT_surrogate(ts, n_iterations=1000, seed=None):
    """
    Returns single 1D iteratively refined amplitude-adjusted Fourier transform
    surrogate. A set of AAFT surrogates is iteratively refined to produce a
    closer match of both amplitude distribution and power spectrum of surrogate
    and original data.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param n_iterations: number of iterations of the procedure
    :type n_iterations: int
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D IAAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    # FT of original data
    xf = np.fft.rfft(ts, axis=0)
    # FT amplitudes
    xf_amps = np.abs(xf)
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)

    # starting point of iterative procedure
    R = get_single_AAFT_surrogate(ts, seed=seed)
    if R.ndim == 1:
        R = R[:, np.newaxis]
    # iterate: `R` is the surrogate with "true" amplitudes and `s` is the
    # surrogate with "true" spectrum
    for _ in range(n_iterations):
        # get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=0)
        r_phases = r_fft / np.abs(r_fft)
        # transform back, replacing the actual amplitudes by the desired
        # ones, but keeping the phases exp(i*phase(i))
        s = np.fft.irfft(xf_amps * r_phases, n=ts.shape[0], axis=0)
        #  rescale to desired amplitude distribution
        ranks = s.argsort(axis=0).argsort(axis=0)
        for j in range(R.shape[1]):
            R[:, j] = sorted_original[ranks[:, j], j]

    return R.squeeze()


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
