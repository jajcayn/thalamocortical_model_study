"""
Spindle detection helper functions.
"""


import logging
from itertools import tee

import numpy as np
import pandas as pd
import yasa
from neurolib.utils.signal import RatesSignal, Signal
from scipy.signal import welch


def dominant_frequency(
    signal,
    spectrum_window_size=0.5,
    max_freq=60.0,
    threshold_pxx=0.0,
    order=0,
):
    """
    Compute dominant frequency in the signal.

    :param signal: signal to process
    :type signal: `neurolib.utils.signal.Signal`
    :param spectrum_window_size: window size for the hanning window, in seconds
    :type spectrum_window_size: float
    :param max_freq: maximum frequency of interest
    :type max_freq: float
    :param threshold_pxx: only assumes dominant frequency if max power is larger
        than this
    :type threshold_pxx: float
    :param order: which dominant to return, default is dominant, ergo 0
    :type order: int
    :return: dominant frequency and associated power per timeseries
    :rtype: pd.DataFrame
    """
    assert isinstance(signal, Signal)
    # placeholder for the result
    result = pd.DataFrame(
        [], index=["dominant frequency", f"power at dom. fr. [{signal.unit}²]"]
    )
    for col_name, timeseries in signal.iterate(return_as="xr"):
        freqs, psds = welch(
            timeseries.values,
            fs=signal.sampling_frequency,
            window="hann",
            nperseg=int(spectrum_window_size / signal.dt) - 1,
            scaling="spectrum",
        )
        freqs = freqs[freqs < max_freq]
        psds = psds[0 : len(freqs)]
        # find max
        max_psds = np.argsort(psds)[::-1][order]
        dominant_freq = freqs[max_psds] if max(psds) > threshold_pxx else 0.0
        dom_power = psds[max_psds]
        result[col_name] = [dominant_freq, dom_power]

    return result


def scale_to_voltage(dataarray, feature_range=(-80, -20)):
    """
    Scale data array to voltage-like range. Good for spindles detection.
    """
    data_scale = (feature_range[1] - feature_range[0]) / (
        dataarray.max(dim="time") - dataarray.min(dim="time")
    )
    data_min = feature_range[0] - dataarray.min(dim="time") * data_scale

    return data_scale * dataarray + data_min


def spindles_detect_aln(ts, rel_power=0.15, duration=(0.5, 2.0)):
    """
    Detect spindles on ALN model output. First, check variance, maximum firing
    rate, and dominant frequency so that we are not in the down / up state with
    almost no variable activity; and then scale to voltage.

    :param ts: timeseries from ALN model to detect spindles on
    :type ts: xr.DataArray
    :param rel_power: relative power of sigma band to broadband signal (1-30)
        for detection
    :type rel_power: float
    :param duration: duration of spindles - as (min, max) tuple in seconds
    :type duration: tuple(float,float)
    :return: spindle detection using YASA
    :rtype: `yasa.main.SpindlesResults`|None
    """
    assert ts.ndim == 1, "Only works on 1D timeseries"
    dom_freq = float(
        dominant_frequency(RatesSignal(ts), threshold_pxx=1.0).iloc[0]
    )
    if ts.max() < 1.0 or ts.std() < 0.5 or dom_freq == 0.0:
        logging.warning("No spindles found")
        return None
    sf = float(1.0 / (ts.time[1] - ts.time[0]))
    return yasa.spindles_detect(
        scale_to_voltage(ts).values,
        sf=sf,
        duration=duration,
        thresh={"rel_pow": rel_power, "corr": 0.5, "rms": 1.0},
    )


def spindles_detect_thalamus(
    tcr_ts, trn_ts=None, trn_median_thresh=5.0, **kwargs
):
    """
    Detect spindles on TCR model output.

    :param tcr_ts: timeseries from TCR model to detect spindles on
    :type tcr_ts: xr.DataArray
    :param trn_ts: timeseries from TRN model to prevent false positives
        [spindles only when trn is also oscillating]
    :type trn_ts: xr.DataArray
    :param trn_median_thresh: threshold for median of firing rate in TRN
    :type trn_median_thresh: float
    :return: spindle detection using YASA
    :rtype: `yasa.main.SpindlesResults`|None
    """
    assert tcr_ts.ndim == 1, "Only works on 1D timeseries"
    if tcr_ts.max() < 10.0 or tcr_ts.std() < 5.0:
        logging.warning("No spindles found: STD too low")
        return None
    if trn_ts is not None and np.nanmedian(trn_ts) < trn_median_thresh:
        logging.warning("No spindles found: TRN is not oscillating")
        return None
    sf = float(1.0 / (tcr_ts.time[1] - tcr_ts.time[0]))
    return yasa.spindles_detect(
        scale_to_voltage(tcr_ts).values, sf=sf, **kwargs
    )


def so_phase_while_spindle(so_phase, spindle_amp, down_state_centers, **kwargs):
    """
    Compute SO phase on spindle maximum between down states.

    :param so_phase: phase of the SO
    :type so_phase: np.ndarray
    :param spindle_amp: spindle amplitude
    :type spindle_amp: np.ndarray
    :param down_state_centers: indices of down state centers
    :type down_state_centers: np.ndarray
    :return: SO phases on maximum spindle amplitude
    :rtype: np.ndarray
    """
    SOphases = []

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    spindle_amp_threshold = kwargs.get("sp_amp_thresh", 5.0)
    for start_idx, end_idx in pairwise(down_state_centers):
        # get amplitude for each down state
        sp_amp_down_state = spindle_amp[start_idx:end_idx]
        # find max sigma amp
        idx_max = sp_amp_down_state.argmax()
        if sp_amp_down_state[idx_max] >= spindle_amp_threshold:
            SOphases.append(so_phase[start_idx + idx_max])

    return np.array(SOphases)


def event_based_so_phase_while_spindle(
    so_phase, spindle_amp, down_state_centers, sf, event_length=2.0, **kwargs
):
    """
    Compute SO phase on spindle maximum within single DOWN event.

    :param so_phase: phase of the SO
    :type so_phase: np.ndarray
    :param spindle_amp: spindle amplitude
    :type spindle_amp: np.ndarray
    :param down_state_centers: indices of down state centers
    :type down_state_centers: np.ndarray
    :param sf: sampling frequency, in Hz
    :type sf: float
    :param event_length: length of the DOWN event, before and after, in seconds
    :type event_length: float
    :return: SO phases on maximum spindle amplitude
    :rtype: np.ndarray
    """
    SOphases = []
    spindle_amp_threshold = kwargs.get("sp_amp_thresh", 5.0)
    samples_length = int(sf * event_length)

    for ds_center in down_state_centers[1:-1]:
        start_idx = max(ds_center - samples_length, 0)
        end_idx = min(ds_center + samples_length, len(so_phase))
        # get amplitude for each down state
        sp_amp_down_state = spindle_amp[start_idx:end_idx]
        # find max sigma amp
        idx_max = sp_amp_down_state.argmax()
        if sp_amp_down_state[idx_max] >= spindle_amp_threshold:
            SOphases.append(so_phase[start_idx + idx_max])

    return np.array(SOphases)
