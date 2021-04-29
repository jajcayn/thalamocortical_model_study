"""
Helper for CFC computation.
"""


from copy import deepcopy
from functools import partial
from multiprocessing import cpu_count

import numpy as np
import xarray as xr
from neurolib.utils.signal import Signal
from pathos.multiprocessing import Pool
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import entropy

from statistical_testing import (
    get_single_AAFT_surrogate,
    get_single_FT_surrogate,
    get_single_IAAFT_surrogate,
    get_single_shuffle_surrogate,
    get_single_time_shift_surrogate,
)

LEAF_SIZE = 15


class SurrogateSignal(Signal):
    """
    Holds signal data and can create surrogates.
    """

    @classmethod
    def from_signal(cls, other_signal):
        """
        Initialize surrogate signal from other signal.
        """
        assert isinstance(other_signal, Signal)
        # __finalize__ copies all attributes
        surr_sig = cls(other_signal.data).__finalize__(other_signal)
        # now alter them as per surrogates
        surr_sig.name = "Surrogates: " + surr_sig.name
        surr_sig.label = "surr: " + surr_sig.label
        surr_sig.signal_type = "surr: " + surr_sig.signal_type
        return surr_sig

    def construct_surrogates(
        self, surrogate_type, univariate=True, inplace=True, **kwargs
    ):
        """
        Construct single 1D surrogate column-wise from the signal.

        :param surrogate_type: type of the surrogate to construct, implemented:
            - shift
            - shuffle
            - FT
            - AAFT
            - IAAFT
        :type surrogate_type: str
        :param univariate: if True, each column will be seeded independetly
            (not preserving any kind of relationship between columns); if False
            will use multivariate construction, i.e. one seed for all
            ealizations in columns, hence will preserve relationships
        :type univariate: bool
        :param inplace: whether to do the operation in place or return
        :type inplace: bool
        :**kwargs: potential arguments for surrogate creation
        """
        seed = (
            None
            if univariate
            else np.random.randint(low=0, high=np.iinfo(np.uint32).max)
        )

        def get_surr(ts, surr_type, seed, **kwargs):
            if surr_type == "shift":
                return get_single_time_shift_surrogate(ts, seed=seed)
            elif surr_type == "shuffle":
                return get_single_shuffle_surrogate(ts, seed=seed, **kwargs)
            elif surr_type == "FT":
                return get_single_FT_surrogate(ts, seed=seed)
            elif surr_type == "AAFT":
                return get_single_AAFT_surrogate(ts, seed=seed)
            elif surr_type == "IAAFT":
                return get_single_IAAFT_surrogate(ts, seed=seed, **kwargs)

        surrogates = xr.apply_ufunc(
            lambda x: get_surr(x, surrogate_type, seed, **kwargs),
            self.data,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
        )
        add_steps = [f"{surrogate_type} surrogate"]
        if inplace:
            self.data = surrogates
            self.process_steps += add_steps
        else:
            return self.__constructor__(surrogates).__finalize__(
                self, add_steps
            )


class BaseXFreqEvaluate:
    name = ""

    def __init__(
        self,
        measure_function,
        slow_timeseries_preprocessing,
        fast_timeseries_preprocessing,
        measure_settings={},
        surrogate_settings={"num_surr": 1000, "surrogate_type": "FT"},
        workers=cpu_count(),
    ):
        """
        :param measure_function: function for computing the x-freq measure
        :type measure_function: callable
        :param slow_timeseries_preprocessing: preprocessing for slow variable,
            is used also for surrogate
        :type slow_timeseries_preprocessing: callable
        :param fast_timeseries_preprocessing: preprocessing for fast variable,
            is used also for surrogate
        :type fast_timeseries_preprocessing: callable
        :param measure_settings: additional kwargs to measure function
        :type measure_settings: dict
        :param surrogate_settings: settings for surrogate: dict with "num_surr"
            and "surrogate_type" keys; can also contain additional keyword
            arguments to surrogate function
        :type surrogate_settings: dict
        :param workers: number of workers for surrogate evaluation
        :type workers: int
        """
        assert callable(measure_function)
        assert callable(slow_timeseries_preprocessing)

        self.num_surr = surrogate_settings.pop("num_surr", 1000)
        self.surr_settings = surrogate_settings

        self.measure_function = partial(measure_function, **measure_settings)
        self.slow_preprocessing = slow_timeseries_preprocessing
        self.fast_preprocessing = fast_timeseries_preprocessing
        self.workers = workers

    @staticmethod
    def validate_input(input_ts):
        raise NotImplementedError

    @staticmethod
    def _get_data(data):
        raise NotImplementedError

    @staticmethod
    def _construct_surrogates(data, **kwargs):
        raise NotImplementedError

    def _compute_surr(self, args):
        to_randomize, fast = args
        surr_ts = self._construct_surrogates(to_randomize, **self.surr_settings)
        slow_surr = self._get_data(self.slow_preprocessing(surr_ts))
        fast_surr = self._get_data(self.fast_preprocessing(fast))
        return self.measure_function(slow_surr, fast_surr)

    def run(self, slow_timeseries, fast_timeseries):
        """
        Run the cross-frequency measure.

        :param slow_timeseries: timeseries for slow phenomenon, usually phase
        :type slow_timeseries: `neuro_signal.Signal`|`mne.io.RawArray`|
            `mne.EpochsArray`
        :param fast_timeseries: timeseries for fast phenomenon, can be phase or
            amplitude, depends on the measure
        :type fast_timeseries: `neuro_signal.Signal`|`mne.io.RawArray`|
            `mne.EpochsArray`
        :return: value of the x-freq measure and values in surrogates
        """
        self.validate_input(slow_timeseries)
        self.validate_input(fast_timeseries)
        slow_data = self._get_data(
            self.slow_preprocessing(deepcopy(slow_timeseries))
        )
        fast_data = self._get_data(
            self.fast_preprocessing(deepcopy(fast_timeseries))
        )
        measure_data = self.measure_function(slow_data, fast_data)

        pool = Pool(self.workers)
        surr_results = np.array(
            pool.map(
                self._compute_surr,
                [
                    (deepcopy(slow_timeseries), deepcopy(fast_timeseries))
                    for _ in range(self.num_surr)
                ],
            )
        )
        pool.close()
        pool.join()

        return measure_data, surr_results


class XFreqEvaluateSignal(BaseXFreqEvaluate):
    """
    Cross-scale measures for Signal class, i.e. model outputs.
    """

    @staticmethod
    def validate_input(input_ts):
        assert isinstance(input_ts, Signal)

    @staticmethod
    def _get_data(data):
        return data.data.values.squeeze()

    @staticmethod
    def _construct_surrogates(data, **kwargs):
        surrogate = SurrogateSignal.from_signal(deepcopy(data))
        surrogate.construct_surrogates(**kwargs, inplace=True)
        return surrogate


def kullback_leibler_modulation_index(
    phase_timeseries, amplitude_timeseries, bins=36, return_for_plotting=False
):
    """
    Compute Kullback-Leibler modulation index, i.e. bin amplitude timeseries as
    per phases and compare that distribution to uniform (no coupling) using KL
    divergence. Relates to phase-amplitude coupling strength. MI of 0 happens
    for uniformly distributed amplitdes w.r.t phases, while maximum value of 1
    means Dirac-like delta distribution for amplitudes w.r.t. phases.

    Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010). Measuring
        phase-amplitude coupling between neuronal oscillations of different
        frequencies. Journal of neurophysiology, 104(2), 1195-1210.

    :param phase_timeseries: timeseries of driving phases
    :type phase_timeseries: np.ndarray
    :param amplitude_timeseries: timeseries of driven amplitudes
    :type amplitude_timeseries: np.ndarray
    :param bins: number of bins to use for the histogram or array of bin edges
    :type bins: int|np.ndarray
    :param return_for_plotting: if True, return also normalized amplitude
        histogram and phase bins for plotting purposes
    :type return_for_plotting: bool
    :return: value of DK-MI and possibly amplitude histogram and phases bin
        edges for plotting purposes
    :rtype: float, (np.ndarray, np.ndarray)
    """
    assert (
        phase_timeseries.shape == amplitude_timeseries.shape
    ), "Unequal length of timeseries"
    # digitize phases
    if isinstance(bins, int):
        phase_bin_edges = np.linspace(
            phase_timeseries.min(), phase_timeseries.max(), bins + 1
        )
    elif isinstance(bins, np.ndarray) and bins.ndim == 1:
        phase_bin_edges = bins.copy()
        bins = bins.shape[0] - 1
    # - 1 to make pythonic counting 0 to bins-1 instead of 1 to bins
    phase_bins = np.digitize(phase_timeseries, phase_bin_edges) - 1
    # compute amplitude histogram conditioned on phase value
    amplitude_histogram = np.array(
        [np.mean(amplitude_timeseries[phase_bins == i]) for i in range(bins)]
    )
    # normalize, i.e. create probability distribution out of it with integral
    # over all phases to be 1
    amplitude_histogram /= np.sum(amplitude_histogram)
    # modulation index is DK(p||U)/log(N) where p is amplitude histogram
    # (distribution) and U is uniform distribution; and DK(p||U) = log(N) - H(p)
    # where N is the number of bins and H(p) is Shannon entropy of amplitude
    # histogram; using natural logarithms
    kl_mi = (np.log(bins) - entropy(amplitude_histogram)) / np.log(bins)
    if return_for_plotting:
        return kl_mi, (amplitude_histogram, phase_bin_edges)
    else:
        return kl_mi


def mean_vector_length(phase_timeseries, amplitude_timeseries):
    """
    Compute mean vector length, where complex timeseries is created using phase
    and amplitude timeseires. The length (real part) of the result represents
    the amount of phase-amplitude coupling, while the phase represents the mean
    phase where amplitude is strongest.

    Canolty, R. T., Edwards, E., Dalal, S. S., Soltani, M., Nagarajan, S. S.,
        Kirsch, H. E., ... & Knight, R. T. (2006). High gamma power is
        phase-locked to theta oscillations in human neocortex. Science,
        313(5793), 1626-1628.

    :param phase_timeseries: timeseries of driving phases
    :type phase_timeseries: np.ndarray
    :param amplitude_timeseries: timeseries of driven amplitudes
    :type amplitude_timeseries: np.ndarray
    :return: value for mean vector length
    :rtype: complex
    """
    assert (
        phase_timeseries.shape == amplitude_timeseries.shape
    ), "Unequal length of timeseries"
    complex_timeseries = amplitude_timeseries * np.exp(
        np.complex(0, 1) * phase_timeseries
    )
    mvl = np.sum(complex_timeseries, axis=0) / complex_timeseries.shape[0]
    return mvl


def phase_locking_value(theta_low_freq, theta_high_freq, n=1, m=1):
    """
    Return complex phase locking value (synchronization index) between two
    phase time series in the n:m mode. The radius of PLV - abs(PLV) - indicates
    the strength of phase locking, while the angle - angle(PLV) - represents a
    phase shift.

    Cohen, M. X. (2008). Assessing transient cross-frequency coupling in EEG
        data. Journal of neuroscience methods, 168(2), 494-499.

    :param theta_low_freq: phase timeseries of low(er) frequency process, in
        radians
    :type theta_low_freq: np.ndarray
    :param theta_high_freq: phase timeseries of high(er) frequency process, in
        radians
    :type theta_high_freq: np.ndarray
    :param n: with m defines the phase-locking mode n:m
    :type n: int
    :param m: with n defines the phase-locking mode n:m
    :type m: int
    :return: complex phase locking value in the n:m mode
    :rtype: complex
    """
    assert isinstance(n, int) and n >= 1, "n must be int >= 1"
    assert isinstance(m, int) and m >= 1, "m must be int >= 1"
    assert (
        theta_low_freq.shape == theta_high_freq.shape
    ), "Unequal length of timeseries"
    complex_phase_diff = np.exp(
        np.complex(0, 1) * (n * theta_low_freq - m * theta_high_freq)
    )
    plv = np.sum(complex_phase_diff, axis=0) / theta_low_freq.shape[0]
    return plv


def _standardize_ts(ts):
    """
    Returns centered time series with zero mean and unit variance.
    """
    assert np.squeeze(ts).ndim == 1, "Only 1D time series can be centered"
    ts -= np.mean(ts)
    ts /= np.std(ts, ddof=1)

    return ts


def _create_naive_eqq_bins(ts, no_bins):
    """
    Create naive EQQ bins given the timeseries.
    """
    assert ts.ndim == 1, "Only 1D timeseries supported"
    ts_sorted = np.sort(ts)
    # bins start with minimum
    ts_bins = [ts.min()]
    for i in range(1, no_bins):
        # add bin edge according to number of bins
        ts_bins.append(ts_sorted[int(i * ts.shape[0] / no_bins)])
    # add last bin - maximum
    ts_bins.append(ts.max())
    return ts_bins


def _create_shifted_eqq_bins(ts, no_bins):
    """
    Create EQQ bins with possible shift if the same values would fall into
    different bins.
    """
    assert ts.ndim == 1, "Only 1D timeseries supported"
    ts_sorted = np.sort(ts)
    # bins start with minimum
    ts_bins = [ts.min()]
    # ideal case
    one_bin_count = ts.shape[0] / no_bins
    for i in range(1, no_bins):
        idx = int(i * one_bin_count)
        if np.all(np.diff(ts_sorted[idx - 1 : idx + 2]) != 0):
            ts_bins.append(ts_sorted[idx])
        elif np.any(np.diff(ts_sorted[idx - 1 : idx + 2]) == 0):
            where = np.where(np.diff(ts_sorted[idx - 1 : idx + 2]) != 0)[0]
            expand_idx = 1
            while where.size == 0:
                where = np.where(
                    np.diff(ts_sorted[idx - expand_idx : idx + 1 + expand_idx])
                    != 0
                )[0]
                expand_idx += 1
            if where[0] == 0:
                ts_bins.append(ts_sorted[idx - expand_idx])
            else:
                ts_bins.append(ts_sorted[idx + expand_idx])
    ts_bins.append(ts.max())
    return ts_bins


def mutual_information(
    x, y, algorithm="EQQ", bins=None, k=None, log2=True, standardize=True
):
    """
    Compute mutual information between two timeseries x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) )
    where p(x), p(y) and p(x, y) are probability distributions.

    :param x: first timeseries, has to be 1D
    :type x: np.ndarray
    :param y: second timeseries, has to be 1D
    :type y: np.ndarray
    :param algorithm: which algorithm to use for probability density estimation:
        - EQD: equidistant binning [1]
        - EQQ_naive: naive equiquantal binning, can happen that samples with
            same value fall into different bins [2]
        - EQQ: equiquantal binning with edge shifting, if same values happen to
            be at the bin edge, the edge is shifted so that all samples of the
            same value will fall into the same bin, can happen that not all the
            bins have necessarily the same number of samples [2]
        - knn: k-nearest neighbours search using k-dimensional tree [3]
        number of bins, at least for EQQ algorithms, should not exceed 3rd root
            of the number of the data samples, in case of I(x,y), i.e. MI of
            two variables [2]
    :param bins: number of bins for binning algorithms
    :type bins: int|None
    :param k: number of neighbours for knn algorithm
    :type k: int|None
    :param log2: whether to use log base 2 for binning algorithms, then the
        units are bits, if False, will use natural log which makes the units
        nats
    :type log2: bool
    :param standardize: whether to standardize timeseries before computing MI,
        i.e. transformation to zero mean and unit variance
    :type standardize: bool
    :return: estimate of the mutual information between x and y
    :rtype: float

    [1] Butte, A. J., & Kohane, I. S. (1999). Mutual information relevance
        networks: functional genomic clustering using pairwise entropy
        measurements. In Biocomputing 2000 (pp. 418-429).
    [2] Paluš, M. (1995). Testing for nonlinearity using redundancies:
        Quantitative and qualitative aspects. Physica D: Nonlinear Phenomena,
        80(1-2), 186-205.
    [3] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
        information. Physical review E, 69(6), 066138.
    """
    assert x.ndim == 1 and y.ndim == 1, "Only 1D timeseries supported"
    if standardize:
        x = _standardize_ts(x)
        y = _standardize_ts(y)

    if algorithm == "knn":
        assert k is not None, "For knn algorithm, `k` must be provided"
        data = np.vstack([x, y]).T
        # build k-d tree
        tree = cKDTree(data, leafsize=LEAF_SIZE)
        # find k-nearest neighbour indices for each point, use the maximum
        # (Chebyshev) norm, which is also limit p -> infinity in Minkowski
        _, ind = tree.query(data, k=k + 1, p=np.inf)
        sum_ = 0
        for n in range(data.shape[0]):
            # find x and y distances between nth point and its k-nearest
            # neighbour
            eps_x = np.abs(data[n, 0] - data[ind[n, -1], 0])
            eps_y = np.abs(data[n, 1] - data[ind[n, -1], 1])
            # use symmetric algorithm with one eps - see the paper
            eps = np.max((eps_x, eps_y))
            # find number of points within eps distance
            n_x = np.sum(np.less(np.abs(x - x[n]), eps)) - 1
            n_y = np.sum(np.less(np.abs(y - y[n]), eps)) - 1
            # add to digamma sum
            sum_ += digamma(n_x + 1) + digamma(n_y + 1)

        sum_ /= data.shape[0]

        mi = digamma(k) - sum_ + digamma(data.shape[0])

    elif algorithm.startswith("E"):
        assert (
            bins is not None
        ), "For binning algorithms, `bins` must be provided"
        log_f = np.log2 if log2 else np.log

        if algorithm == "EQD":
            # bins are simple number of bins - will be divided equally
            x_bins = bins
            y_bins = bins

        elif algorithm == "EQQ_naive":
            x_bins = _create_naive_eqq_bins(x, no_bins=bins)
            y_bins = _create_naive_eqq_bins(y, no_bins=bins)

        elif algorithm == "EQQ":
            x_bins = _create_shifted_eqq_bins(x, no_bins=bins)
            y_bins = _create_shifted_eqq_bins(y, no_bins=bins)

        else:
            raise ValueError(f"Unknown MI algorithm: {algorithm}")

        xy_bins = [x_bins, y_bins]

        # compute histogram counts
        count_x = np.histogramdd([x], bins=[x_bins])[0]
        count_y = np.histogramdd([y], bins=[y_bins])[0]
        count_xy = np.histogramdd([x, y], bins=xy_bins)[0]

        # normalise
        count_xy /= np.float(np.sum(count_xy))
        count_x /= np.float(np.sum(count_x))
        count_y /= np.float(np.sum(count_y))

        # sum
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if count_x[i] != 0 and count_y[j] != 0 and count_xy[i, j] != 0:
                    mi += count_xy[i, j] * log_f(
                        count_xy[i, j] / (count_x[i] * count_y[j])
                    )

    else:
        raise ValueError(f"Unknown MI algorithm: {algorithm}")

    return mi
