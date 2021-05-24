"""
Various helper functions.
"""

from itertools import tee

import numpy as np
import scipy.interpolate as si
import xarray as xr
from neurolib.utils.signal import Signal


def dummy_detect_down_states(timeseries, threshold, min_down_length=0.1):
    """
    Dummy detector of down states.

    :param timeseries: input timeseries
    :type timeseries: `neurolib.utils.signal.Signal`
    :param threshold: threshold for down states (<), in Hz (or other unit of the
        signal)
    :type threshold: float
    :param min_down_length: minimum length of down state for classification, in
        seconds
    :type min_down_length: float
    :return: list of down state indices
    :rtype: list[np.ndarray]
    """

    assert isinstance(timeseries, Signal)
    low_values = timeseries.data.values < threshold
    idx_change = np.nonzero(low_values[1:] != low_values[:-1])[0] + 1
    down_states = np.split(np.arange(len(timeseries.data.values)), idx_change)
    # filter length
    down_states = [
        state
        for state in down_states
        if (
            (len(state) > int(min_down_length * timeseries.sampling_frequency))
            and (low_values[state]).all()
        )
    ]

    return down_states


def get_dummy_so_phase(
    signal,
    threshold=10.0,
    min_down_length=0.1,
    boundary_freq=1.0,
    kind="linear",
):
    """
    Compute dummy phase of the slow oscillation: find down states (values less
    than threshold and duration longer than min_down_length), assign midpoints
    and linearly interpolate in between. First and last segment is computed with
    artificial boundary frequency.

    :param signal: signal to get phase from
    :type signal: `neurolib.utils.signal.Signal`
    :param threshold: threshold for finding down states
    :type threshold: float
    :param min_down_length: minimum duration of the down state, in seconds
    :type min_down_length: float
    :param boundary_freq: first and last segment in the timeseries will be
        treated as if the slow oscillation has this frequency, in Hz
    :type boundary_freq: float
    :param kind: kind of phase interpolation:
        - "linear": put -pi/pi to middle of the down states and linearly
            interpolate in between
        - "pchi": put -pi/pi to middle of the down state, 0 at maximum up
            state and do piecewise cubic Hermite interpolation
    :type kind: str
    :return: wrapped SO phase of the signal
    :rtype: `neurolib.utils.signal.Signal`
    """
    # find down states
    down_states = dummy_detect_down_states(signal, threshold, min_down_length)
    # mid points in down states -> -pi
    phases = np.zeros_like(signal.data.values)
    for state in down_states:
        mid_idx = state[len(state) // 2]
        phases[mid_idx - 1] = np.pi
        phases[mid_idx] = -np.pi
    idx_low_phase = np.where(phases == -np.pi)[0]

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    for idx_low, idx_high in pairwise(idx_low_phase):
        if kind == "linear":
            f_interp = si.interp1d(
                [idx_low, idx_high - 1], phases[[idx_low, idx_high - 1]]
            )
        elif kind == "pchi":
            # find up state maximum
            up_state_idx = (
                signal.data.values[idx_low:idx_high].argmax() + idx_low
            )
            f_interp = si.PchipInterpolator(
                [idx_low, up_state_idx, idx_high - 1],
                [phases[idx_low], 0.0, phases[idx_high - 1]],
            )
        phases[idx_low:idx_high] = f_interp(np.arange(idx_low, idx_high))

    # start and end ~ make artificial oscillation with boundary_freq
    slope = 2 * np.pi / ((1.0 / boundary_freq) * signal.sampling_frequency)
    phases[0 : idx_low_phase[0]] = slope * np.arange(0, idx_low_phase[0])
    phases[idx_low_phase[-1] :] = -np.pi + slope * np.arange(
        0, phases.shape[0] - idx_low_phase[-1]
    )

    return signal.__constructor__(
        xr.DataArray(
            phases,
            dims=signal.data.dims,
            coords=signal.data.coords,
            name=signal.data.name,
        )
    ).__finalize__(signal, ["manual SO wrapped phase"])


def get_phase(signal, filter_args, pad=None):
    """
    Extract phase of the signal. Steps: detrend -> pad -> filter -> Hilbert
    transform -> get phase -> un-pad.

    :param signal: signal to get phase from
    :type signal: `neurolib.utils.signal.Signal`
    :param filter_args: arguments for `Signal`'s filter method (see its
        docstring)
    :type filter_args: dict
    :param pad: how many seconds to pad, if None, won't pad
    :type pad: float|None
    :return: wrapped Hilbert phase of the signal
    :rtype: `neurolib.utils.signal.Signal`
    """
    assert isinstance(signal, Signal)
    phase = signal.detrend(inplace=False)
    if pad:
        phase.pad(
            how_much=pad, in_seconds=True, padding_type="reflect", side="both"
        )
    phase.filter(**filter_args)
    phase.hilbert_transform(return_as="phase_wrapped")
    if pad:
        phase.sel([phase.start_time + pad, phase.end_time - pad])
    return phase


def get_amplitude(signal, filter_args, pad=None):
    """
    Extract amplitude of the signal. Steps: detrend -> pad -> filter -> Hilbert
    transform -> get amplitude -> un-pad.

    :param signal: signal to get amplitude from
    :type signal: `neurolib.utils.signal.Signal`
    :param filter_args: arguments for `Signal`'s filter method (see its
        docstring)
    :type filter_args: dict
    :param pad: how many seconds to pad, if None, won't pad
    :type pad: float|None
    :return: Hilbert amplitude of the signal
    :rtype: `neurolib.utils.signal.Signal`
    """
    assert isinstance(signal, Signal)
    amplitude = signal.detrend(inplace=False)
    if pad:
        amplitude.pad(
            how_much=pad, in_seconds=True, padding_type="reflect", side="both"
        )
    amplitude.filter(**filter_args)
    amplitude.hilbert_transform(return_as="amplitude")
    if pad:
        amplitude.sel([amplitude.start_time + pad, amplitude.end_time - pad])
    return amplitude
