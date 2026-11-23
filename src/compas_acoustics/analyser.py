import warnings
from functools import lru_cache

import numpy as np
import pyfar as pf
import pyrato
from scipy.signal import butter
from scipy.signal import sosfilt

from compas_acoustics.defaults import Defaults

__all__ = ["Analyser", "frequency_response", "time_response"]

# Title, x label and y label of each parameter, in the order of analyse(); the Grasshopper plots read it through the
# worker
PARAMETERS = {
    "EDT": ("Early Decay Time (EDT)", "Frequency (Hz)", "Time (s)"),
    "T20": ("Reverberation Time (T20)", "Frequency (Hz)", "Time (s)"),
    "T30": ("Reverberation Time (T30)", "Frequency (Hz)", "Time (s)"),
    "C50": ("Clarity (C50)", "Frequency (Hz)", "dB"),
    "C80": ("Clarity (C80)", "Frequency (Hz)", "dB"),
    "D50": ("Definition (D50)", "Frequency (Hz)", "Value (%)"),
    "DRR": ("Direct-to-Reverberant Ratio (DRR)", "Frequency (Hz)", "dB"),
    "Ts": ("Center Time (Ts)", "Frequency (Hz)", "Time (s)"),
    "STI": ("Speech Transmission Index (STI)", "Frequency (Hz)", "Modulation transfer index (0-1)"),
    "G": ("Sound Strength (G)", "Frequency (Hz)", "dB"),
    "SPL": ("Sound Pressure Level (SPL), overall A-weighted", "Frequency (Hz)", "dB"),
    "LF": ("Lateral Energy Fraction (LF)", "Frequency (Hz)", "Fraction (0-1)"),
    "LFC": ("Lateral Energy Fraction, cosine (LFC)", "Frequency (Hz)", "Fraction (0-1)"),
}
_PARAM_NAMES = list(PARAMETERS)

# pyrato < 1.0 (the only one on Python 3.9) has no `parameters` module
_PYRATO_PARAMETERS = hasattr(pyrato, "parameters")
_STI_SUPPORTED = _PYRATO_PARAMETERS and hasattr(pyrato.parameters, "speech_transmission_index_indirect")

# A-weighting at the octave centres, in dB (IEC 61672-1:2013 Table 3)
_A_WEIGHTS = {125: -16.1, 250: -8.6, 500: -3.2, 1000: 0.0, 2000: 1.2, 4000: 1.0, 8000: -1.1}


@lru_cache(maxsize=None)
def _octave_band_filter(sampling_rate, center_frequency):
    # One octave band of pyfar's filter bank (Butterworth, order 14; IEC 61260-1:2014). A range of +-10 % of the centre
    # selects that band alone. The filter keeps no state, so one design is reused.
    return pf.dsp.filter.fractional_octave_bands(
        signal=None,
        num_fractions=1,
        sampling_rate=sampling_rate,
        frequency_range=(center_frequency * 0.9, center_frequency * 1.1),
        order=14,
    )


@lru_cache(maxsize=None)
def _octave_bank(sampling_rate):
    # The seven bands of _octave_band_filter in one filter: the same coefficients and samples, one call per segment
    return pf.dsp.filter.fractional_octave_bands(signal=None, num_fractions=1, sampling_rate=sampling_rate, frequency_range=(125, 8000), order=14)


def _level_ratio(first, second):
    """``10 log10(first / second)`` to 2 decimals; NaN when either energy is zero."""
    if first > 0 and second > 0:
        return round(float(10.0 * np.log10(first / second)), 2)
    return np.nan


@lru_cache(maxsize=None)
def _free_field_edcs(sampling_rate):
    """:meth:`Analyser._band_edcs` of the package-scale free field at 10 m, ``1 / (4 pi 10)`` at 0.1 s."""
    reference = np.zeros(sampling_rate)
    reference[int(0.1 * sampling_rate)] = 1.0 / (4.0 * np.pi * 10.0)
    analyser = Analyser(reference)
    analyser.sampling_rate = sampling_rate
    return analyser._band_edcs(reference)


def _channels_first(ir):
    """An impulse response as (channels, samples), from (samples,) or (samples, channels)."""
    return np.atleast_2d(np.asarray(ir, dtype=float).T)


def _decay_end(ir, trim_db, margin=0.05):
    """Samples of ``ir`` (channels, samples) until the Schroeder curve of all channels falls ``trim_db`` below its
    start, lengthened by the fraction ``margin``."""
    remaining = np.cumsum(np.sum(ir**2, axis=0)[::-1])[::-1]
    if remaining[0] == 0:
        return ir.shape[1]
    below = np.flatnonzero(remaining <= remaining[0] * 10 ** (-trim_db / 10))
    if not len(below):
        return ir.shape[1]
    return min(ir.shape[1], int(np.ceil(below[0] * (1 + margin))))


def _level_re_peak(magnitude):
    """``20 log10 |x|`` in dB re the peak of all of ``magnitude``, floored 100 dB below it."""
    magnitude = np.abs(magnitude)
    peak = np.max(magnitude)
    if peak == 0:
        return np.full(magnitude.shape, -100.0)
    return 20 * np.log10(np.maximum(magnitude / peak, 1e-5))


def time_response(ir, sampling_rate, db=True, trim_db=None):
    """The impulse response against time, per channel.

    Parameters
    ----------
    ir : np.ndarray
        (samples,) or (samples, channels).
    sampling_rate : int
    db : bool, optional
        The level in dB re the peak of all channels, floored 100 dB below it; else the amplitude.
    trim_db : float, optional
        End the response 5 % after its Schroeder curve (all channels) falls this far below its start.

    Returns
    -------
    times, values : np.ndarray
        (samples,) in seconds, and (channels, samples).
    """
    ir = _channels_first(ir)
    if trim_db is not None:
        ir = ir[:, : _decay_end(ir, trim_db)]
    times = np.arange(ir.shape[1]) / sampling_rate
    return times, _level_re_peak(ir) if db else ir


def frequency_response(ir, sampling_rate, num_fractions=3, frequency_range=(20.0, 20000.0)):
    """The magnitude response per channel, smoothed over ``1 / num_fractions`` octave.

    Parameters
    ----------
    ir : np.ndarray
        (samples,) or (samples, channels).
    sampling_rate : int
    num_fractions : int, optional
    frequency_range : tuple of float, optional
        The bins kept, in Hz.

    Returns
    -------
    frequencies, values : np.ndarray
        (bins,) in Hz, and (channels, bins) in dB re the peak of all channels, floored 100 dB below it.
    """
    signal = pf.Signal(_channels_first(ir), sampling_rate)
    smoothed, _ = pf.dsp.smooth_fractional_octave(signal, num_fractions=num_fractions)
    keep = (smoothed.frequencies >= frequency_range[0]) & (smoothed.frequencies <= frequency_range[1])
    return smoothed.frequencies[keep], _level_re_peak(smoothed.freq[..., keep])


class Analyser:
    """Room acoustic parameters of an impulse response (:meth:`analyse`).

    Every window starts at ``onset``, the direct sound of channel 0 as ``pyfar.dsp.find_impulse_response_start``
    finds it with the 20 dB criterion of ISO 3382-1:2009, p. 22.

    Parameters
    ----------
    ir : list, np.ndarray or ImpulseResponse
        (samples,) or (samples, channels). LF and LFC take the first-order channels of an ambisonic
        :class:`ImpulseResponse`, every other parameter channel 0 (the omnidirectional component, or the left ear).
        An :class:`ImpulseResponse` is analysed at its raw level and sets :attr:`sampling_rate`.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> room = ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1))
    >>> scene.add_acoustic_objects(room)
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Mono(ray_tracing=True, n_rays=2000, duration=0.5))
    >>> analyser = ca.Analyser(next(iter(responses.values())))
    >>> results = analyser.analyse()
    >>> results["freqs"]
    [125, 250, 500, 1000, 2000, 4000, 8000]
    >>> results["T30_overall"] > 0
    True
    """

    def __init__(self, ir):
        self.ir = ir
        self.analysis_data = None
        # Zero-phase bands of self.ir by centre frequency, kept while analyse() runs
        self._bands_from_onset = None
        sampling_rate = getattr(ir, "sampling_rate", None)
        self.sampling_rate = int(sampling_rate) if sampling_rate is not None else Defaults().SAMPLING_RATE

    @property
    def ir(self):
        return self._ir

    @ir.setter
    def ir(self, ir):
        # G and SPL need the absolute level of a package-scale response
        self._package_scale = (getattr(ir, "generation", None) or {}).get("scale") == "package"
        # the spatial parameters need the response's channels, source and receiver
        self._response = ir
        self._ir = self._as_mono_array(ir)
        # pyfar's start sample is -1 for a response whose first sample is already within 20 dB of its maximum
        start = pf.dsp.find_impulse_response_start(pf.Signal(self._ir, getattr(ir, "sampling_rate", None) or Defaults().SAMPLING_RATE), threshold=20)
        self.onset = max(0, int(np.squeeze(start)))

    @staticmethod
    def _as_mono_array(ir):
        """A 1D array of an impulse response, at the raw level for an ImpulseResponse."""
        if hasattr(ir, "denormalized"):
            ir = ir.denormalized()
        elif hasattr(ir, "ir") and hasattr(ir, "sampling_rate"):
            ir = ir.ir
        if not isinstance(ir, (list, np.ndarray)):
            raise TypeError("Impulse response must be a list, numpy array or ImpulseResponse.")
        ir_array = np.asarray(ir, dtype=float)
        if ir_array.ndim > 2:
            raise ValueError(f"Impulse response must be 1D or 2D, got {ir_array.ndim}D.")
        if ir_array.ndim == 2:
            if ir_array.shape[1] > 1:
                warnings.warn(f"Impulse response has {ir_array.shape[1]} channels; analysing channel 0 only.")
            ir_array = ir_array[:, 0]
        return ir_array

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, int):
            raise TypeError("Sampling rate must be an integer.")
        self._sampling_rate = sampling_rate

    @property
    def frequency_bands(self):
        return Defaults().DEFAULT_FREQUENCY_BANDS

    def _filter_impulse_response(self, impulse_response, center_frequency):
        sig = pf.Signal(np.asarray(impulse_response, dtype=float), self.sampling_rate)
        return _octave_band_filter(self.sampling_rate, center_frequency).process(sig).time.squeeze()

    def _band_signals(self, segment) -> dict:
        """Causal octave bands of a time segment by centre frequency, padded 0.5 s so that the filters ring out."""
        padded = pf.Signal(np.concatenate([np.asarray(segment, dtype=float), np.zeros(int(0.5 * self.sampling_rate))]), self.sampling_rate)
        return dict(zip(self.frequency_bands, _octave_bank(self.sampling_rate).process(padded).time.squeeze()))

    def _band_edcs(self, segment):
        """Schroeder curves of :meth:`_band_signals`, one channel per band."""
        filtered = pf.Signal(np.stack(list(self._band_signals(segment).values())), self.sampling_rate)
        return pyrato.edc.schroeder_integration(filtered)

    def _band_energies(self, segment) -> dict:
        """Energies of :meth:`_band_signals`: filter delay and ringing stay with the segment."""
        return {f: float(np.sum(band**2)) for f, band in self._band_signals(segment).items()}

    def _band_from_onset(self, center_frequency):
        """Zero-phase (forward-backward) octave band of :attr:`ir` from 0.1 s before the direct sound.

        The band's direct sound stays at the broadband onset; the ringing is two-sided, hence the lead. Both are the
        package's choice.

        Returns
        -------
        tuple of (np.ndarray, int)
            The band and the index of the direct sound in it.
        """
        kept = self._bands_from_onset
        if kept is not None and center_frequency in kept:
            return kept[center_frequency]
        fs = self.sampling_rate
        lead = int(round(0.1 * fs))
        onset = self.onset
        padded = np.concatenate([np.zeros(lead), self.ir, np.zeros(int(0.5 * fs))])
        forward = self._filter_impulse_response(padded, center_frequency)
        zero_phase = self._filter_impulse_response(forward[::-1], center_frequency)[::-1]
        band = zero_phase[onset:], lead
        if kept is not None:
            kept[center_frequency] = band
        return band

    def reverberation_times(self) -> dict:
        """EDT, T20 and T30 per band (ISO 3382-1:2009).

        Schroeder (1965, J. Acoust. Soc. Am. 37(3)) backward integration of the zero-phase band
        (:meth:`_band_from_onset`) from the direct sound to the last non-zero sample, normalized to 0 dB at its start,
        then linear regression. No noise compensation: truncate a measured response at its noise floor first.
        """
        ir_array = self.ir
        onset = self.onset
        # Trailing zeros are cut: the filters' ringing into them would steepen the end of the integral
        end = int(np.flatnonzero(ir_array)[-1]) + 1 if np.any(ir_array) else len(ir_array)
        keys = ("EDT", "T20", "T30")
        # in pyrato.parameters from pyrato 1.0, at the top level of pyrato < 1.0
        regression = getattr(pyrato, "parameters", pyrato).reverberation_time_linear_regression

        results = {key: [] for key in keys}
        failed = {}
        for center_freq in self.frequency_bands:
            band, direct = self._band_from_onset(center_freq)
            sig = pf.Signal(band[direct : direct + end - onset], self.sampling_rate)
            edc = pyrato.edc.schroeder_integration(sig)
            with np.errstate(divide="ignore", invalid="ignore"):
                edc = pf.TimeData(edc.time / edc.time[..., :1], edc.times)
                try:
                    values = [round(float(np.squeeze(regression(edc, key))), 2) for key in keys]
                except (ValueError, IndexError, np.linalg.LinAlgError) as error:
                    # a band without energy, or too short to hold the regression range
                    failed[center_freq] = error
                    values = [np.nan] * len(keys)
            for key, value in zip(keys, values):
                results[key].append(value)
        if failed:
            warnings.warn(
                f"pyrato could not evaluate the decay in the {', '.join(f'{f:g}' for f in failed)} Hz band(s) ({next(iter(failed.values()))}); EDT, T20 and T30 are NaN there."
            )
        return results

    def clarity_definition(self) -> dict:
        """C50, C80 and D50 per band (ISO 3382-1:2009).

        Early and late parts are cut from the broadband response before causal band filtering, so filter delay and
        ringing stay with the part they belong to, the package's choice.
        """
        ir_array = self.ir
        onset = self.onset
        bands = list(self.frequency_bands)
        early_50, early_80 = (onset + int(round(t * self.sampling_rate)) for t in (0.05, 0.08))

        energies = {
            name: self._band_energies(segment)
            for name, segment in (
                ("early_50", ir_array[onset:early_50]),
                ("late_50", ir_array[early_50:]),
                ("early_80", ir_array[onset:early_80]),
                ("late_80", ir_array[early_80:]),
            )
        }

        results = {"C50": [], "C80": [], "D50": []}
        for freq in bands:
            e_early_50, e_late_50 = energies["early_50"][freq], energies["late_50"][freq]
            e_early_80, e_late_80 = energies["early_80"][freq], energies["late_80"][freq]
            results["C50"].append(_level_ratio(e_early_50, e_late_50))
            results["C80"].append(_level_ratio(e_early_80, e_late_80))
            results["D50"].append(round(float(e_early_50 / (e_early_50 + e_late_50) * 100.0), 2) if e_early_50 > 0 and e_late_50 > 0 else np.nan)
        return results

    def direct_to_reverberant_ratio(self) -> dict:
        """Direct-to-reverberant ratio per band and for 700-4000 Hz, in dB.

        The energy from 1 ms before to 5 ms after the direct sound over the energy after it. The window and the
        700-4000 Hz band of ``'DRR_overall'`` (a fourth-order Butterworth band-pass, ``butter(2, ..., "bandpass")``) are
        the package's choice. Per band the two parts are cut before filtering; for ``'DRR_overall'`` the response is
        filtered first.

        Returns
        -------
        dict
            ``'DRR'`` per band in :attr:`frequency_bands`, and ``'DRR_overall'``.
        """
        ir_array = self.ir
        onset = self.onset
        start, end = (max(0, onset + int(round(t * self.sampling_rate))) for t in (-0.001, 0.005))

        bands = list(self.frequency_bands)
        direct = self._band_energies(ir_array[start:end])
        reverberant = self._band_energies(ir_array[end:])
        filtered = sosfilt(butter(2, (700.0, 4000.0), "bandpass", fs=self.sampling_rate, output="sos"), ir_array)
        return {
            "DRR": [_level_ratio(direct[freq], reverberant[freq]) for freq in bands],
            "DRR_overall": _level_ratio(np.sum(filtered[start:end] ** 2), np.sum(filtered[end:] ** 2)),
        }

    def center_time(self) -> dict:
        """Centre time Ts of the zero-phase bands, in seconds from the broadband direct sound (ISO 3382-1:2009)."""
        results = {"Ts": []}
        for freq in self.frequency_bands:
            band, direct = self._band_from_onset(freq)
            if not np.any(band):
                results["Ts"].append(np.nan)
                continue
            if _PYRATO_PARAMETERS:
                edc = pyrato.edc.schroeder_integration(pf.Signal(band, self.sampling_rate))
                # measured from the start of the EDC, `direct` samples before the direct sound
                ts = float(np.squeeze(pyrato.parameters.center_time(edc))) - direct / self.sampling_rate
            else:
                energy = band**2
                t = (np.arange(band.size) - direct) / self.sampling_rate
                ts = float(np.sum(t * energy) / np.sum(energy))
            results["Ts"].append(round(ts, 4))
        return results

    def speech_transmission_index(self, snr=np.inf) -> dict:
        """STI and the modulation transfer index per octave band (IEC 60268-16:2020, indirect method).

        Responses shorter than the 1.6 s of Section 6.2 are zero-padded, which assumes a full decay.
        ``snr`` is the signal-to-noise ratio in dB, one value or one per octave band 125-8000 Hz.

        Returns
        -------
        dict
            ``'mti'`` per octave band 125-8000 Hz and ``'sti'``; NaN where pyrato cannot compute them.
        """
        if not _STI_SUPPORTED:
            warnings.warn("STI is not available with pyrato {} (requires pyrato >= 1.0 and Python >= 3.11). STI and MTI values are set to NaN.".format(pyrato.__version__))
            return {"mti": np.full(7, np.nan), "sti": np.nan}

        ir_array = self.ir
        min_samples = int(np.ceil(1.6 * self.sampling_rate))
        if ir_array.size < min_samples:
            ir_array = np.pad(ir_array, (0, min_samples - ir_array.size))
        sig = pf.Signal(ir_array, self.sampling_rate)
        snr_arg = float(snr) if np.ndim(snr) == 0 else snr
        try:
            # (7 octave bands, 14 modulation frequencies)
            mtf = np.squeeze(np.asarray(pyrato.parameters.modulation_transfer_function(sig, snr=snr_arg)))
            snr_app = 10.0 * np.log10(mtf / (1.0 - mtf + 1e-12) + 1e-12)
            mti = np.mean(np.clip((snr_app + 15.0) / 30.0, 0, 1), axis=-1)
            # The public STI function computes the MTF again; pyrato's private reduction (Annex A.2.1) halves the cost
            sti_calc = getattr(pyrato.parameters, "_sti_calc", None)
            if sti_calc is not None and mtf.shape == (7, 14):
                sti_val = round(float(sti_calc(mtf)), 4)
            else:
                sti_val = round(float(np.squeeze(pyrato.parameters.speech_transmission_index_indirect(sig, snr=snr_arg))), 4)
        except Exception as e:
            warnings.warn(f"STI computation failed: {e}")
            mti = np.full(7, np.nan)
            sti_val = np.nan
        return {"mti": mti, "sti": sti_val}

    def stage_support(self) -> dict:
        """ST_early and ST_late per band (Gade; ISO 3382-1:2009 Annex C), for source and receiver 1 m apart.

        The energy 20-100 ms (ST_early) or 100-1000 ms (ST_late) after the direct sound over the energy of the first
        10 ms, in dB; the windows are cut before band filtering. Single numbers average 250-2000 Hz.

        Returns
        -------
        dict
            ``'freqs'``, ``'ST_early'`` and ``'ST_late'`` per band in :attr:`frequency_bands`, and
            ``'ST_early_overall'`` and ``'ST_late_overall'``.
        """
        ir_array = self.ir
        fs = self.sampling_rate
        onset = self.onset

        if onset + fs > len(ir_array):
            warnings.warn(f"Impulse response ends {(len(ir_array) - onset) / fs:.2f} s after the direct sound; the ST_late window (100-1000 ms) is truncated.")

        freqs = list(self.frequency_bands)
        overall_bands = [250, 500, 1000, 2000]

        def segment(window):
            start, stop = (onset + int(round(t * fs)) for t in window)
            return ir_array[start : min(stop, len(ir_array))]

        direct = self._band_energies(segment((0.0, 0.010)))
        result = {"freqs": freqs}
        for name, window in {"ST_early": (0.020, 0.100), "ST_late": (0.100, 1.000)}.items():
            reflected = self._band_energies(segment(window))
            values = {f: 10.0 * np.log10(reflected[f] / direct[f]) if reflected[f] > 0 and direct[f] > 0 else np.nan for f in freqs}
            result[name] = [round(float(values[f]), 2) for f in freqs]
            result[f"{name}_overall"] = round(float(np.nanmean([values[f] for f in overall_bands])), 2)
        return result

    def sound_strength(self, reference_ir=None) -> dict:
        """Strength G per band (ISO 3382-1:2009): the energy over that of the free field at 10 m.

        Both at their raw level. The single number averages 500 and 1000 Hz.

        Parameters
        ----------
        reference_ir : ImpulseResponse or np.ndarray, optional
            The source's free-field response at 10 m, at the same sample rate. Default: the package-scale free field
            ``1 / (4 pi 10)``, for a package-scale response.

        Returns
        -------
        dict
            ``'freqs'``, ``'G'`` per band in :attr:`frequency_bands`, and ``'G_overall'``.
        """
        if reference_ir is None:
            if not self._package_scale:
                raise ValueError("G without a reference_ir needs a package-scale ImpulseResponse; pass the free-field response at 10 m.")
        else:
            reference_rate = getattr(reference_ir, "sampling_rate", None)
            if reference_rate is not None and int(reference_rate) != self.sampling_rate:
                raise ValueError(f"Reference impulse response sample rate {reference_rate} Hz differs from {self.sampling_rate} Hz.")
            reference = self._as_mono_array(reference_ir)

        freqs = list(self.frequency_bands)
        if not _PYRATO_PARAMETERS:
            warnings.warn("Sound strength G needs pyrato >= 1.0 (pyrato {} is installed); G is NaN.".format(pyrato.__version__))
            return {"freqs": freqs, "G": [np.nan] * len(freqs), "G_overall": np.nan}

        reference_edcs = _free_field_edcs(self.sampling_rate) if reference_ir is None else self._band_edcs(reference)
        strength = np.asarray(pyrato.parameters.sound_strength(self._band_edcs(self.ir), reference_edcs)).reshape(-1)
        values = dict(zip(freqs, strength))
        return {
            "freqs": freqs,
            "G": [round(float(values[f]), 2) for f in freqs],
            "G_overall": round(float((values[500] + values[1000]) / 2), 2),
        }

    def sound_pressure_level(self, source_spl=None, strength=None) -> dict:
        """Sound pressure level per band, ``Lp = spl + G - 20 dB``: G's reference at 10 m is 20 dB below 1 m.

        Parameters
        ----------
        source_spl : float or list of float, optional
            The source's on-axis level at 1 m (:attr:`compas_acoustics.Source.spl`), one or one per band. Default: that
            of the response's source.
        strength : dict, optional
            :meth:`sound_strength`, when already computed.

        Returns
        -------
        dict
            ``'freqs'``, ``'SPL'`` per band, ``'SPL_overall'`` the A-weighted total in dB(A) and ``'SPL_total'`` the
            unweighted one. NaN, with a warning, without a level or off the package scale.
        """
        freqs = list(self.frequency_bands)
        nan = {"freqs": freqs, "SPL": [np.nan] * len(freqs), "SPL_overall": np.nan, "SPL_total": np.nan}
        if source_spl is None:
            source_spl = getattr(getattr(self._response, "source", None), "spl", None)
        if source_spl is None:
            warnings.warn("SPL needs the source's level at 1 m: pass source_spl or a response whose source has one.")
            return nan
        if np.size(source_spl) not in (1, len(freqs)):
            raise ValueError(f"source_spl must be one value or one per band ({len(freqs)}), got {np.size(source_spl)}.")
        level = np.broadcast_to(np.asarray(source_spl, dtype=float).reshape(-1), (len(freqs),))
        if strength is None:
            if not self._package_scale:
                warnings.warn("SPL needs a response on the package scale (a simulated one), which keeps its absolute level.")
                return nan
            strength = self.sound_strength()
        spl = level + np.asarray(strength["G"], dtype=float) - 20.0
        weights = np.array([_A_WEIGHTS[f] for f in freqs])
        with np.errstate(divide="ignore"):
            total = 10.0 * np.log10(np.nansum(10.0 ** (spl / 10.0))) if np.any(np.isfinite(spl)) else np.nan
            weighted = 10.0 * np.log10(np.nansum(10.0 ** ((spl + weights) / 10.0))) if np.any(np.isfinite(spl + weights)) else np.nan
        return {"freqs": freqs, "SPL": [round(float(v), 2) for v in spl], "SPL_overall": round(float(weighted), 2), "SPL_total": round(float(total), 2)}

    def lateral_fraction(self) -> dict:
        """Early lateral energy fractions J_LF and J_LFC per band (ISO 3382-1:2009 Annex A).

        J_LF is the figure-of-eight energy 5-80 ms after the direct sound over the omnidirectional energy 0-80 ms; J_LFC
        takes ``|p_L p|`` for ``p_L^2`` (Kleiner 1989; Bradley 1994). The figure-of-eight is the first-order dipole of
        the response, horizontal in the receiver's frame and perpendicular to the source direction (the receiver's y
        axis for a source straight above or below). Windows are cut before band filtering, the package's choice.

        Returns
        -------
        dict
            ``'freqs'``, ``'LF'`` and ``'LFC'`` per band, and ``'LF_overall'`` and ``'LFC_overall'``, the mean of
            125-1000 Hz. NaN for a response that is not an ambisonic :class:`ImpulseResponse` (ACN, SN3D or N3D).
        """
        response = self._response
        freqs = list(self.frequency_bands)
        nan = {"freqs": freqs, "LF": [np.nan] * len(freqs), "LFC": [np.nan] * len(freqs), "LF_overall": np.nan, "LFC_overall": np.nan}
        if getattr(response, "ambisonic_order", None) is None or not hasattr(response, "denormalized"):
            return nan
        channels = _channels_first(response.denormalized())
        if channels.shape[0] < 4:
            return nan
        w, y, z, x = channels[:4]
        if (getattr(response, "normalization", None) or "sn3d").lower() == "n3d":
            x, y, z = (c / np.sqrt(3.0) for c in (x, y, z))

        fs = self.sampling_rate
        onset = self.onset
        source, receiver = response.source, response.receiver
        direction = np.asarray(receiver.rotation).T @ (np.asarray(source.position, dtype=float) - np.asarray(receiver.position, dtype=float))
        axis = np.cross([0.0, 0.0, 1.0], direction)
        norm = np.linalg.norm(axis)
        # a source straight above or below the receiver: the receiver's own lateral (y) axis
        axis = axis / norm if norm > 1e-9 * max(np.linalg.norm(direction), 1e-300) else np.array([0.0, 1.0, 0.0])
        lateral = axis[0] * x + axis[1] * y + axis[2] * z

        overall_bands = [125, 250, 500, 1000]
        start, stop = (onset + int(round(t * fs)) for t in (0.005, 0.080))
        omni_early = self._band_signals(w[onset:stop])
        omni_lateral = self._band_signals(w[start:stop])
        lateral_early = self._band_signals(lateral[start:stop])

        lf, lfc = {}, {}
        for f in freqs:
            total = float(np.sum(omni_early[f] ** 2))
            lf[f] = float(np.sum(lateral_early[f] ** 2)) / total if total > 0 else np.nan
            lfc[f] = float(np.sum(np.abs(lateral_early[f] * omni_lateral[f]))) / total if total > 0 else np.nan
        return {
            "freqs": freqs,
            "LF": [round(lf[f], 3) for f in freqs],
            "LFC": [round(lfc[f], 3) for f in freqs],
            "LF_overall": round(float(np.nanmean([lf[f] for f in overall_bands])), 3),
            "LFC_overall": round(float(np.nanmean([lfc[f] for f in overall_bands])), 3),
        }

    def _sti_and_strength(self) -> dict:
        """``'STI'`` (per band the modulation transfer indices), ``'G'`` and their overall values, as :meth:`analyse`.

        The worker computes only these, for a client whose pyrato cannot (:func:`compas_acoustics.worker.analyse`).
        """
        freqs = list(self.frequency_bands)
        sti_data = self.speech_transmission_index()
        # G needs the absolute level; an array or a normalized-only response has none
        if _PYRATO_PARAMETERS and self._package_scale:
            strength = self.sound_strength()
            g_vector, g_overall = strength["G"], strength["G_overall"]
        else:
            g_vector, g_overall = [np.nan] * len(freqs), np.nan

        sti_vector = [round(mti, 2) for mti in sti_data["mti"]]
        return {
            "STI": np.array(sti_vector, dtype=float).tolist(),
            "STI_overall": round(float(sti_data["sti"]), 4),
            "G": np.array(g_vector, dtype=float).tolist(),
            "G_overall": g_overall,
        }

    def analyse(self) -> dict:
        """Every parameter of :data:`_PARAM_NAMES` per band and as ``'<name>_overall'``.

        The overall values of EDT, T20, T30, C50, C80, D50 and Ts average 500 and 1000 Hz, after the single-number
        averages of ISO 3382-1:2009 Table A.1; the same average for all seven is the package's choice. The others have
        their own. The per-band ``'STI'`` values are the modulation transfer indices, ``'STI_overall'`` the STI.
        """
        # reverberation_times and center_time take the same zero-phase bands
        self._bands_from_onset = {}
        try:
            rt = self.reverberation_times()
            clarity = self.clarity_definition()
            drr = self.direct_to_reverberant_ratio()
            ct = self.center_time()
        finally:
            self._bands_from_onset = None
        sti_and_strength = self._sti_and_strength()
        lateral = self.lateral_fraction()
        level = getattr(getattr(self._response, "source", None), "spl", None)
        if level is not None and not all(np.isnan(np.asarray(sti_and_strength["G"], dtype=float))):
            spl = self.sound_pressure_level(level, strength={"G": sti_and_strength["G"]})
        else:
            spl = {"SPL": [np.nan] * len(self.frequency_bands), "SPL_overall": np.nan}

        freqs = list(self.frequency_bands)
        param_vectors = {
            "EDT": rt["EDT"],
            "T20": rt["T20"],
            "T30": rt["T30"],
            "C50": clarity["C50"],
            "C80": clarity["C80"],
            "D50": clarity["D50"],
            "DRR": drr["DRR"],
            "Ts": ct["Ts"],
            "STI": sti_and_strength["STI"],
            "G": sti_and_strength["G"],
            "SPL": spl["SPL"],
            "LF": lateral["LF"],
            "LFC": lateral["LFC"],
        }
        own_overall = {
            "STI": sti_and_strength["STI_overall"],
            "G": sti_and_strength["G_overall"],
            "DRR": drr["DRR_overall"],
            "LF": lateral["LF_overall"],
            "LFC": lateral["LFC_overall"],
            "SPL": spl["SPL_overall"],
        }

        result = {"freqs": freqs}
        for param in _PARAM_NAMES:
            vals = np.array(param_vectors[param], dtype=float)
            result[param] = vals.tolist()
            if param in own_overall:
                result[f"{param}_overall"] = own_overall[param]
            else:
                result[f"{param}_overall"] = round(float(np.nanmean(vals[[freqs.index(500), freqs.index(1000)]])), 2)

        self.analysis_data = result
        return self.analysis_data

    def print_table(self, parameters=None, show_bands=True, show_overall=False):
        """Print the results of :meth:`analyse`.

        Parameters
        ----------
        parameters : str or list of str, optional
            Default: all.
        show_bands : bool, optional
            Per-band values.
        show_overall : bool, optional
            Overall values.
        """
        params = parameters if parameters else _PARAM_NAMES
        if isinstance(params, str):
            params = [params]
        freqs = self.analysis_data["freqs"]
        header = (freqs if show_bands else []) + (["Overall"] if show_overall else [])
        print()
        print(f"{'':14}" + "".join(f"{str(f):>8}" for f in header))
        print("-" * (14 + 8 * len(header)))
        for p in params:
            row = (self.analysis_data[p] if show_bands else []) + ([self.analysis_data[f"{p}_overall"]] if show_overall else [])
            print(f"{p:14}" + "".join(f"{v:>8}" for v in row))

    def plot(self, parameters=None):
        """Plot the results of :meth:`analyse` against frequency, one figure per parameter (default: all)."""
        import matplotlib.pyplot as plt

        if parameters is not None:
            if isinstance(parameters, str):
                parameters = [parameters]
            for param in parameters:
                if param not in _PARAM_NAMES:
                    raise ValueError(f"Parameter '{param}' not found in results. Available parameters: {_PARAM_NAMES}")
        else:
            parameters = _PARAM_NAMES

        for param in parameters:
            y = np.array(self.analysis_data[param], dtype=float)
            y_clean = y[np.isfinite(y)]

            if len(y_clean) > 0:
                vmin, vmax = np.min(y_clean), np.max(y_clean)
                lower_limit = vmin - (abs(vmin) * 0.4)
                upper_limit = vmax + (abs(vmax) * 0.4)
                if param == "STI":
                    lower_limit, upper_limit = max(0.0, lower_limit), min(1.0, upper_limit)
                if param == "D50":
                    lower_limit, upper_limit = max(0.0, lower_limit), min(100.0, upper_limit)
                ylim = [lower_limit, upper_limit]
            else:
                ylim = None

            plt.figure(figsize=(8, 4), num=f"{param}")
            plt.title(f"{param}")
            ax = plt.gca()
            plt.plot([str(f) for f in self.analysis_data["freqs"]], y, marker="o", linewidth=2, label=param)

            if param in ["EDT", "T20", "T30"]:
                from matplotlib.ticker import FormatStrFormatter
                from matplotlib.ticker import MultipleLocator

                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                if ylim and (ylim[1] - ylim[0]) < 0.4:
                    mid = (ylim[0] + ylim[1]) / 2
                    ylim = [mid - 0.3, mid + 0.3]

            if ylim:
                if ylim[0] == ylim[1]:
                    plt.ylim(ylim[0] - 0.2, ylim[1] + 0.2)
                else:
                    plt.ylim(ylim)

            _, x_label, y_label = PARAMETERS[param]
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    def export_csv(self, filepath: str, parameters=None):
        """Write the results of :meth:`analyse` to a CSV file.

        Parameters
        ----------
        filepath : str
        parameters : str or list of str, optional
            Default: all.
        """
        import csv

        if parameters is None:
            parameters = _PARAM_NAMES
        elif isinstance(parameters, str):
            parameters = [parameters]

        header = ["Parameter"] + [str(f) for f in self.analysis_data["freqs"]] + ["Overall"]
        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for param in parameters:
                if param not in _PARAM_NAMES:
                    warnings.warn(f"Parameter '{param}' not available for CSV export.")
                    continue
                data_row = self.analysis_data[param] + [self.analysis_data[f"{param}_overall"]]
                writer.writerow([param] + [v if not np.isnan(v) else "" for v in data_row])

        print(f"Selected results successfully saved to: {filepath}")

    def plot_rir(self):
        """Plot the impulse response against time."""
        import matplotlib.pyplot as plt

        t, values = time_response(self.ir, self.sampling_rate, db=False)
        plt.figure(figsize=(8, 4), num="Impulse Response")
        plt.plot(t, values[0], label="Impulse Response")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Impulse Response")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_frequency_response(self, smooth=3, raw_alpha=0.3):
        """Plot the magnitude response, with a 1/``smooth`` octave smoothed one over it unless ``smooth`` is None."""
        import matplotlib.pyplot as plt

        sig = pf.Signal(self.ir, self.sampling_rate)
        ref = np.max(np.abs(sig.freq))

        plt.figure(figsize=(8, 4), num="Frequency Response")
        ax = plt.gca()

        if smooth is not None:
            pf.plot.freq(sig, log_reference=ref, ax=ax, linewidth=0.5, alpha=raw_alpha, color="gray", label="Raw")
            sm_signal, _ = pf.dsp.smooth_fractional_octave(sig, num_fractions=smooth)
            pf.plot.freq(sm_signal, log_reference=ref, ax=ax, linewidth=1.5, label=f"1/{smooth} octave smoothing")
            ax.legend(loc="best")
        else:
            pf.plot.freq(sig, log_reference=ref, ax=ax, linewidth=1)

        plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        plt.xlim([20, 20000])
        plt.ylim([-60, 10])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dB")
        plt.title("Frequency Response")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.grid(True, which="minor", ls=":", alpha=0.2)
        plt.tight_layout()
        plt.show()
