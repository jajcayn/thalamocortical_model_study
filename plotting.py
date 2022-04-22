"""
Basic plotting helpers.
"""


from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency.multitaper import psd_array_multitaper
from mne.time_frequency.psd import psd_array_welch
from mne.viz.utils import _convert_psds
from scipy.stats import circmean, circstd, sem

from statistical_testing import get_p_values

POLAR_XTICKS = np.pi / 180.0 * np.array([0, 90, 180, 270])
POLAR_XTICKLABELS = ["0", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"]


def plot_spectrum(
    data,
    method="welch",
    units="Hz",
    title="",
    spectrum_window_size=0.5,
    alpha=1.0,
    cmap=None,
    legend=True,
    logx=False,
    logy=True,
    ax=None,
):
    """
    Plot power spectrum estimated using Welch method with Hamming window or
    multi-taper method.

    :param data: data for PSD estimation
    :type data: pd.DataFrame
    :param method: method for estimating the power spectrum, `welch` or
        `multitaper`
    :type method: str
    :param units: units of data
    :type units: str
    :param title: title
    :type title: str
    :param spectrum_window_size: window size for the Hann window, in seconds
        only used in Welch method
    :type spectrum_window_size: float
    :param cmap: colormap for colors
    :type cmap: str
    :param legend: whether to display a legend
    :type legend: bool
    :param logx: whether to draw x axis in logarithmic scale
    :type logx: bool
    :param logy: whether to draw y axis in logarithmic scale
    :type logy: bool
    :param ax: axis to plot to; if None, will create
    :type ax: `matplotlib.axes._axes.Axes`|None
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    data = pd.DataFrame(data)
    # get sampling frequency
    dt = float(data.index[1] - data.index[0])
    sampling_freq = 1.0 / dt
    if method == "welch":
        spectrum_estimator = partial(
            psd_array_welch,
            n_per_seg=int(spectrum_window_size / dt) - 1,
            window="hann",
        )
    elif method == "multitaper":
        spectrum_estimator = psd_array_multitaper
    else:
        raise ValueError(f"Unknown method: {method}.")
    colors = (
        [plt.get_cmap(cmap)(x) for x in np.linspace(0.0, 1.0, data.shape[1])]
        if cmap is not None
        else [f"C{i}" for i in range(data.shape[1])]
    )
    for i, column in enumerate(data):
        # frequencies above 60Hz are not really interesting for us
        psds, freqs = spectrum_estimator(
            data[column].values, sfreq=sampling_freq, fmin=0.0, fmax=60.0
        )
        ylabel = _convert_psds(
            psds[:, np.newaxis],
            dB=False,
            estimate="power",
            unit=units,
            scaling=1.0,
            ch_names=[],
        )
        ax.plot(freqs, psds, label=column, color=colors[i], alpha=alpha)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_xlabel(f"FREQUENCY [Hz] / {method.upper()} method")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    if legend:
        ax.legend()


def plot_average_events_ts(
    ts,
    events_idx,
    time_before,
    time_after,
    plus_minus_stats="sem",
    color="C0",
    title="",
    ylabel="",
    second_ts=None,
    color_second_ts="C1",
    ax=None,
):
    """
    Plot average time series of events. Optionally plot second time series to
    second y-axis.

    :param ts: time series to plot events from, can be 1D pandas or xarray
        arrays, needs to have `time` index,
    :type ts: `pd.DataFrame`|`pd.Series`|`xr.DataArray`
    :param events_idx: indices for events, as n_events x events time array
    :type events_idx: np.ndarray
    :param time_before: time before center of event, in seconds
    :type time_before: float
    :param time_after: time after center of event, in seconds
    :type time_after: float
    :param plus_minus_stats: what statistics to use to shade the region around
        mean, available are:
        "sem" - standard error of the mean
        "quantile" - 97.5 and 2.5 quantile in the data
        "std" - standard deviation
    :type plus_minus_stats: str
    :param color: color of the `ts`
    :type color: str
    :param title: title for the plot
    :type title: str
    :param ylabel: ylabel for the plot
    :type ylabel: str
    :param second_ts: optional second timeseries, same type etc as `ts`
    :type second_ts: `pd.DataFrame`|`pd.Series`|`xr.DataArray`
    :param color_second_ts: color of the second `ts`
    :type color_second_ts: str
    :param ax: axis to plot to; if None, will create
    :type ax: `matplotlib.axes._axes.Axes`|None
    """

    def stats_up(ts, stat):
        if stat == "sem":
            return np.mean(ts, axis=0) + sem(ts, ddof=1, axis=0)
        elif stat == "std":
            return np.mean(ts, axis=0) + np.std(ts, ddof=1, axis=0)
        elif stat == "quantile":
            return np.quantile(ts, q=0.975, axis=0)
        else:
            raise ValueError(f"Unknown stat: {stat}")

    def stats_down(ts, stat):
        if stat == "sem":
            return np.mean(ts, axis=0) - sem(ts, ddof=1, axis=0)
        elif stat == "std":
            return np.mean(ts, axis=0) - np.std(ts, ddof=1, axis=0)
        elif stat == "quantile":
            return np.quantile(ts, q=0.025, axis=0)
        else:
            raise ValueError(f"Unknown stat: {stat}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    sf = np.around(float(1.0 / (ts.time[1] - ts.time[0])))
    time_before = int(time_before * sf)
    time_after = int(time_after * sf)
    time = np.arange(-time_before, time_after + 1, dtype="int") / sf
    mean_ts = np.mean([ts.values[idxi] for idxi in events_idx], axis=0)
    stat_up = stats_up(
        [ts.values[idxi] for idxi in events_idx], stat=plus_minus_stats
    )
    stat_down = stats_down(
        [ts.values[idxi] for idxi in events_idx], stat=plus_minus_stats
    )
    assert time.shape[0] == mean_ts.shape[0], (time.shape, mean_ts.shape)
    ax.plot(time, mean_ts, color=color)
    ax.fill_between(time, stat_up, stat_down, color=color, alpha=0.3)
    ax.set_ylim([0.0, ts.max()])
    ax.grid()
    if second_ts is not None:
        assert np.all(second_ts.time == ts.time)
        mean_second_ts = np.mean(
            [second_ts.values[idxi] for idxi in events_idx], axis=0
        )
        stat_up_second = stats_up(
            [second_ts.values[idxi] for idxi in events_idx],
            stat=plus_minus_stats,
        )
        stat_down_second = stats_down(
            [second_ts.values[idxi] for idxi in events_idx],
            stat=plus_minus_stats,
        )
        ax2 = ax.twinx()
        ax2.plot(time, mean_second_ts, color=color_second_ts)
        ax2.fill_between(
            time,
            stat_up_second,
            stat_down_second,
            color=color_second_ts,
            alpha=0.3,
        )
        ax2.set_ylim([mean_second_ts.min(), mean_second_ts.max() * 4])
        ax2.set_yticks([mean_second_ts.min(), mean_second_ts.max()])
    ax.set_xlabel("TIME AROUND PEAK [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title + f"; {events_idx.shape[0]} events")


def plot_circular_histogram(
    angles,
    bins=16,
    density=None,
    offset=0.0,
    lab_unit="radians",
    start_zero=False,
    title="",
    ax=None,
):
    """
    Plot circular histogram (rose plot). Expects angles in [-pi, pi].

    :param angles: angles for plotting
    :type angles: np.ndarray|None
    :param bins: number of bins
    :type bins: int
    :param density: if True, area of bin would be proportional to density, if
        False, its height would be proportional, i.e. area would be squared
    :type density: bool
    :param offset: direction of zero angle
    :type offset: float|int
    :param lab_unit: unit of angles: "radians" or "degrees"
    :type lab_unit: str
    :param start_zero: whether to make bins symmetric around 0
    :type start_zero: bool
    :param title: title for the plot
    :type title: str
    :param ax: axis to plot to; if None, will create
    :type ax: `matplotlib.axes._axes.Axes`|None
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    # wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    # set bins symmetrically around zero
    if start_zero:
        # to have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(
        bin[:-1],
        radius,
        zorder=10,
        align="edge",
        width=widths,
        fill=False,
        edgecolor="C0",
    )
    angles_mean = circmean(angles, high=np.pi, low=-np.pi, nan_policy="omit")
    angles_std = circstd(angles, high=np.pi, low=-np.pi, nan_policy="omit")
    # mean
    ax.plot([0, angles_mean], [0, np.max(radius)], color="red", linewidth=3.0)
    # mean +- std
    ax.plot(
        [0, angles_mean - angles_std],
        [0, np.max(radius)],
        "--",
        color="red",
        linewidth=1.5,
    )
    ax.plot(
        [0, angles_mean + angles_std],
        [0, np.max(radius)],
        "--",
        color="red",
        linewidth=1.5,
    )
    ax.fill_between(
        np.linspace(angles_mean - angles_std, angles_mean + angles_std, 100),
        0,
        np.max(radius),
        color="red",
        alpha=0.2,
        zorder=1,
    )
    # set the direction of the zero angle
    ax.set_theta_offset(offset)
    # remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        ax.set_xticks(POLAR_XTICKS)
        ax.set_xticklabels(POLAR_XTICKLABELS)
    ax.set_title(title)


def plot_kullback_leibler_modulation_index(
    dk_mi_result, surrs=None, ax=None, **kwargs
):
    """
    Plot result of Kullback-Leibler modulation index investigation.

    :param dk_mi_result: result of
        `analysis.x_freq.kullback_leibler_modulation_index` function with
        return_for_plottong set to True, hence DK-MI value and tuple with
        amplitude histogram and phase bins
    :type dk_mi_result: float, (np.ndarray, np.ndarray)
    :param surrs: surrogate values for the same as DK-MI result, passed as a
        list of same values as a results
    :type surrs: list[float, (np.ndarray, np.ndarray)]
    :param ax: axis to plot to; if None, will create
    :type ax: `matplotlib.axes._axes.Axes`|None
    :kwargs:
        - data_color - color of the hist for data
        - surr_color - color for the mean of surrogate data
        - perc_color - color for the 95th percentile of surrogate distribution
    """
    dkmi_data, (hist_data, bins) = dk_mi_result
    bin_width = np.diff(bins).mean()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if surrs is not None:
        surr_hist = np.array([surr_result[1][0] for surr_result in surrs])
        surrs_values = np.array([surr_result[0] for surr_result in surrs])
        # plot mean +- SD in surrogate distribution
        ax.bar(
            bins[:-1],
            surr_hist.mean(axis=0),
            yerr=None,  # surr_hist.std(axis=0),
            align="edge",
            width=0.95 * bin_width,
            color=kwargs.pop("surr_color", "#777777"),
            alpha=0.7,
            capsize=5,
        )
        # plot 95th percentile of surrogate distribution
        ax.bar(
            bins[:-1],
            np.percentile(surr_hist, 95.0, axis=0),
            align="edge",
            width=0.95 * bin_width,
            color=kwargs.pop("perc_color", "#222222"),
            alpha=0.7,
        )
        p_val = get_p_values(dkmi_data, surrs_values)

    else:
        p_val = 1.0

    ax.bar(
        bins[:-1],
        hist_data,
        align="edge",
        width=0.95 * bin_width,
        color=kwargs.pop("data_color", "red"),
        alpha=0.8,
    )
    ax.set_ylabel("Amplitude value")
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", 0, r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("Phase bins")
    ax.set_title(
        f"DK-MI value in data: {dkmi_data:.4f}{'*' if p_val < 0.05 else ''}"
    )
