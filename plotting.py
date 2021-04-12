from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency.multitaper import psd_array_multitaper
from mne.time_frequency.psd import psd_array_welch
from mne.viz.utils import _convert_psds


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
