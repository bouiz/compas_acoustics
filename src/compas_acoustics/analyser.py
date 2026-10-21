import warnings
from functools import lru_cache

import numpy as np
import pyfar as pf
import pyrato
from scipy.signal import butter
from scipy.signal import sosfilt

from compas_acoustics.defaults import Defaults

__all__ = ["AcousticAnalyser", "frequency_response", "time_response"]

# Row order used throughout the class
_PARAM_NAMES = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'DRR', 'Ts', 'STI', 'G']

# Single-number values of EDT, T20, T30, C50, C80, D50 and Ts average the 500 and 1000 Hz octaves
# (ISO 3382-1:2009 Table A.1; C50 is a function of D50 and takes its bands)
_SINGLE_NUMBER_BANDS = [500, 1000]

# pyrato < 1.0 (the only one on Python 3.9, Rhino 8) has no `parameters` module
_PYRATO_PARAMETERS = hasattr(pyrato, 'parameters')
_STI_SUPPORTED = _PYRATO_PARAMETERS and hasattr(pyrato.parameters, 'speech_transmission_index_indirect')

# Minimum IR length for STI in seconds (IEC 60268-16:2020, Section 6.2)
_STI_MIN_DURATION = 1.6

# Stage support (ISO 3382-1:2009 Annex C), measured 1 m from the source. Windows in
# seconds after the direct sound; single-number values average the 250-2000 Hz octaves.
_STAGE_DIRECT_WINDOW = (0.0, 0.010)
_STAGE_SUPPORT_WINDOWS = {'ST_early': (0.020, 0.100), 'ST_late': (0.100, 1.000)}
_STAGE_SUPPORT_BANDS = [250, 500, 1000, 2000]

# Direct-to-reverberant ratio (Haapaniemi and Lokki 2018, eq. 1): the direct sound is the first 5 ms
# from its arrival, in seconds; the window opens 1 ms early so that a band-limited arrival is held
# whole. Their single-number value is taken in 700-4000 Hz with a second-order Butterworth band-pass.
_DRR_DIRECT_WINDOW = (-0.001, 0.005)
_DRR_BAND = (700.0, 4000.0)
_DRR_BAND_ORDER = 2

# Single-number strength G averages the 500 and 1000 Hz octaves (ISO 3382-1:2009 Table A.1)
_STRENGTH_BANDS = [500, 1000]

# Distance of the free-field reference of G, in metres (ISO 3382-1:2009 Section 4.3)
_STRENGTH_DISTANCE = 10.0

# Zero padding after a windowed segment so the octave filters can ring out, in seconds
_FILTER_TAIL = 0.5

# Zero-phase band filtering rings symmetrically around the direct sound; band energies are
# integrated from this long before the direct sound, in seconds
_ZERO_PHASE_LEAD = 0.1

# pyrato's ISO 3382-1 regression: in pyrato.parameters, at the top level before pyrato 1.0 (Python 3.9)
_REVERBERATION_TIME = getattr(pyrato, 'parameters', pyrato).reverberation_time_linear_regression

# The direct sound is found as ita_start_IR (ITA Toolbox, ISO 3382-1) finds it, this far below the
# maximum in dB. On the package scale the search starts this long before the geometric arrival, in seconds.
_START_THRESHOLD_DB = 20.0
_GEOMETRIC_SEARCH_LEAD = 0.001

# Floor of the levels of time_response and frequency_response, in dB below the peak
_RESPONSE_FLOOR_DB = 100.0


@lru_cache(maxsize=None)
def _octave_band_filter(sampling_rate, center_frequency):
    # A filter without an initialised state keeps none between calls, so one design serves every call
    return pf.dsp.filter.fractional_octave_bands(
        signal=None,
        num_fractions=1,
        sampling_rate=sampling_rate,
        frequency_range=(center_frequency * 0.9, center_frequency * 1.1),
        order=14,
    )


def _iso3382_start(ir_array, threshold_db=_START_THRESHOLD_DB):
    """Start sample of an impulse response as ``ita_start_IR`` (ITA Toolbox) finds it, after ISO 3382-1.

    The last sample more than ``threshold_db`` below the maximum before the maximum. If an earlier
    sample comes within ``threshold_db - 6`` dB of the maximum, the last sample below the threshold
    before that one. 0 when the maximum is less than 20 dB above the noise (the mean energy of the
    last 10 per cent) or lies in it.
    """
    energy = np.asarray(ir_array, dtype=float) ** 2
    n = energy.size
    peak = int(np.argmax(energy))
    tail = int(round(0.9 * n))
    if peak == 0 or energy[peak] < 100.0 * np.mean(energy[max(tail - 1, 0):]) or peak + 1 > tail:
        return 0
    with np.errstate(divide='ignore'):
        level = 10.0 * np.log10(energy[:peak + 1] / energy[peak])
    below = np.flatnonzero(level < -threshold_db)
    start = int(below[-1]) if below.size else 0
    earlier = np.flatnonzero(level[:start + 1] > -threshold_db + 6.0)
    if earlier.size:
        before = np.flatnonzero(level[:earlier[0] + 1] < -threshold_db)
        start = int(before[-1]) if before.size else 0
    return start


def _channels_first(ir):
    """An impulse response as (channels, samples), from (samples,) or (samples, channels)."""
    return np.atleast_2d(np.asarray(ir, dtype=float).T)


def _decay_end(ir, trim_db, margin=0.05):
    """Samples of ``ir`` (channels, samples) until the Schroeder curve of all channels falls ``trim_db`` below its start, plus ``margin``."""
    remaining = np.cumsum(np.sum(ir ** 2, axis=0)[::-1])[::-1]
    if remaining[0] == 0:
        return ir.shape[1]
    below = np.flatnonzero(remaining <= remaining[0] * 10 ** (-trim_db / 10))
    if not len(below):
        return ir.shape[1]
    return min(ir.shape[1], int(np.ceil(below[0] * (1 + margin))))


def time_response(ir, sampling_rate, db=True, trim_db=None):
    """The impulse response against time, per channel.

    Parameters
    ----------
    ir : np.ndarray
        Shaped (samples,) or (samples, channels).
    sampling_rate : int
    db : bool, optional
        ``True``: the level ``20 log10 |h|`` in dB re the peak of all channels, so that level
        differences between channels are kept, floored 100 dB below it.
        ``False``: the amplitude.
    trim_db : float, optional
        Ends the response 5 % after its Schroeder curve (the energy of all channels) falls this
        far below its start. The Schroeder curve, unlike single samples, is not lengthened by a
        stray late bin.

    Returns
    -------
    times : np.ndarray
        (samples,), in seconds.
    values : np.ndarray
        (channels, samples).
    """
    ir = _channels_first(ir)
    if trim_db is not None:
        ir = ir[:, :_decay_end(ir, trim_db)]
    times = np.arange(ir.shape[1]) / sampling_rate
    if not db:
        return times, ir
    peak = np.max(np.abs(ir))
    if peak == 0:
        return times, np.full_like(ir, -_RESPONSE_FLOOR_DB)
    return times, 20 * np.log10(np.maximum(np.abs(ir) / peak, 10 ** (-_RESPONSE_FLOOR_DB / 20)))


def frequency_response(ir, sampling_rate, num_fractions=3, frequency_range=(20.0, 20000.0)):
    """The magnitude response, fractional-octave smoothed, per channel.

    Parameters
    ----------
    ir : np.ndarray
        Shaped (samples,) or (samples, channels).
    sampling_rate : int
    num_fractions : int, optional
        Smoothing width in fractions of an octave (``pyfar.dsp.smooth_fractional_octave``).
    frequency_range : tuple of float, optional
        Bins kept, in Hz.

    Returns
    -------
    frequencies : np.ndarray
        (bins,), in Hz.
    values : np.ndarray
        (channels, bins), in dB re the peak of all channels, floored 100 dB below it.
    """
    signal = pf.Signal(_channels_first(ir), sampling_rate)
    smoothed, _ = pf.dsp.smooth_fractional_octave(signal, num_fractions=num_fractions)
    keep = (smoothed.frequencies >= frequency_range[0]) & (smoothed.frequencies <= frequency_range[1])
    magnitude = np.abs(smoothed.freq[..., keep])
    peak = np.max(magnitude)
    if peak == 0:
        return smoothed.frequencies[keep], np.full(magnitude.shape, -_RESPONSE_FLOOR_DB)
    return smoothed.frequencies[keep], 20 * np.log10(np.maximum(magnitude / peak, 10 ** (-_RESPONSE_FLOOR_DB / 20)))


class AcousticAnalyser:
    """Class for analyzing acoustic parameters from an impulse response (IR).
    
    Parameters
    ----------
    ir : list, np.ndarray or ImpulseResponse
        The impulse response to analyze, shaped (samples,) or (samples, channels).
        Multi-channel input is reduced to channel 0, which is the omnidirectional
        component of an ambisonic impulse response. An :class:`ImpulseResponse` is
        analysed at its raw (denormalized) level and sets :attr:`sampling_rate`.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Pyroomacoustics(ray_tracing=True, n_rays=2000, duration=0.5))
    >>> analyser = ca.AcousticAnalyser(next(iter(responses.values())))
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
        sampling_rate = getattr(ir, 'sampling_rate', None)
        self.sampling_rate = int(sampling_rate) if sampling_rate is not None else Defaults().SAMPLING_RATE

    @property
    def ir(self):
        return self._ir
    
    @ir.setter
    def ir(self, ir):
        # Geometric direct-sound sample of a package-scale response (see ImpulseResponse.direct_sound_index):
        # the start search begins there, so that an early ray-traced bin is not taken for the direct sound.
        generation = getattr(ir, 'generation', None) or {}
        self._package_scale = generation.get('scale') == 'package'
        self.onset = ir.direct_sound_index() if self._package_scale else None
        self._ir = self._as_mono_array(ir)

    def _on_package_scale(self, ir=None):
        if ir is None:
            return self._package_scale
        return (getattr(ir, 'generation', None) or {}).get('scale') == 'package'

    @staticmethod
    def _as_mono_array(ir):
        """A 1D array of an impulse response, at the raw level for an ImpulseResponse."""
        if hasattr(ir, 'denormalized'):
            ir = ir.denormalized()
        elif hasattr(ir, 'ir') and hasattr(ir, 'sampling_rate'):
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
    
    @property
    def T30(self):
        return self.analysis_data['T30'] if self.analysis_data else None
    
    @property
    def T20(self):
        return self.analysis_data['T20'] if self.analysis_data else None
    
    @property
    def C50(self):
        return self.analysis_data['C50'] if self.analysis_data else None
    
    @property
    def C80(self):
        return self.analysis_data['C80'] if self.analysis_data else None
    
    @property
    def D50(self):
        return self.analysis_data['D50'] if self.analysis_data else None
    
    @property
    def DRR(self):
        return self.analysis_data['DRR'] if self.analysis_data else None
    
    @property
    def STI(self):
        return self.analysis_data['STI'] if self.analysis_data else None
    
    @property
    def Ts(self):
        return self.analysis_data['Ts'] if self.analysis_data else None
    
    @property
    def G(self):
        return self.analysis_data['G'] if self.analysis_data else None

    @property
    def EDT(self):
        return self.analysis_data['EDT'] if self.analysis_data else None
    
    @property
    def T20_overall(self):
        return self.analysis_data['T20_overall'] if self.analysis_data else None
    
    @property
    def T30_overall(self):
        return self.analysis_data['T30_overall'] if self.analysis_data else None
    
    @property
    def C50_overall(self):
        return self.analysis_data['C50_overall'] if self.analysis_data else None
    
    @property
    def C80_overall(self):
        return self.analysis_data['C80_overall'] if self.analysis_data else None
    
    @property
    def D50_overall(self):
        return self.analysis_data['D50_overall'] if self.analysis_data else None
    
    @property
    def DRR_overall(self):
        return self.analysis_data['DRR_overall'] if self.analysis_data else None
    
    @property
    def STI_overall(self):
        return self.analysis_data['STI_overall'] if self.analysis_data else None
    
    @property
    def Ts_overall(self):
        return self.analysis_data['Ts_overall'] if self.analysis_data else None
    
    @property
    def EDT_overall(self):
        return self.analysis_data['EDT_overall'] if self.analysis_data else None

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, (int)):
            raise TypeError("Sampling rate must be a number.")
        self._sampling_rate = sampling_rate

    @property
    def frequency_bands(self):
        # Return custom bands if set, otherwise return defaults
        if hasattr(self, '_frequency_bands'):
            return self._frequency_bands
        return Defaults().DEFAULT_FREQUENCY_BANDS

    @frequency_bands.setter
    def frequency_bands(self, bands):
        if not isinstance(bands, (list, np.ndarray)):
            raise TypeError("Frequency bands must be a list or numpy array.")
        self._frequency_bands = list(bands)


    # Helper functions

    def _filter_impulse_response(self, impulse_response: np.ndarray, center_frequency: float) -> np.ndarray:
        """IEC 61260-1:2014 compliant octave-band filter via pyfar (order-14 SOS).

        frequency_range uses ±10% of the center frequency rather than ±1/2 octave.
        The ±1/2 octave range coincides exactly with adjacent IEC band edges, causing
        pyfar 0.8.0 to include 3 bands instead of 1.
        """
        sig = pf.Signal(np.asarray(impulse_response, dtype=float), self.sampling_rate)
        return _octave_band_filter(self.sampling_rate, center_frequency).process(sig).time.squeeze()


    # Calculate acoustic parameters

    def reverberation_times(self, impulse_response=None) -> dict:
        """Compute reverberation times (EDT, T20, T30) per frequency band.

        Schroeder's backward integration (pyrato ``schroeder_integration``), normalized to
        0 dB at the direct sound, and the ISO 3382-1 linear regression of pyrato's
        ``reverberation_time_linear_regression``. There is no noise compensation: a
        simulated response has no noise floor, and pyrato's Lundeby search
        (``energy_decay_curve_chu_lundeby``) finds no intersection in a ray-traced
        response that decays into numerical zero, which left most bands NaN. A measured
        response with a noise floor should be truncated before it is analysed.

        The decay starts at the direct sound of a zero-phase octave-band IR (see
        :meth:`_band_from_onset`): with the causal filter, its slow build-up at low
        frequencies flattened the first 10 dB and lengthened EDT. It ends at the response's
        last non-zero sample, without the zero padding the filters ring into.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        onset = self._direct_sound_onset(ir_array)
        # Trailing zeros are cut: the filters' ringing into them would steepen the end of the integral
        end = int(np.flatnonzero(ir_array)[-1]) + 1 if np.any(ir_array) else len(ir_array)
        keys = ('EDT', 'T20', 'T30')

        results = {key: [] for key in keys}
        failed = {}
        for center_freq in self.frequency_bands:
            band, direct = self._band_from_onset(ir_array, center_freq)
            # the band already starts at the direct sound
            sig = pf.Signal(band[direct:direct + end - onset], self.sampling_rate)
            edc = pyrato.edc.schroeder_integration(sig)
            with np.errstate(divide='ignore', invalid='ignore'):
                edc = pf.TimeData(edc.time / edc.time[..., :1], edc.times)
                try:
                    values = [round(float(np.squeeze(_REVERBERATION_TIME(edc, key))), 2) for key in keys]
                except (ValueError, IndexError, np.linalg.LinAlgError) as error:
                    # a band without energy, or too short to hold the regression range
                    failed[center_freq] = error
                    values = [np.nan] * len(keys)
            for key, value in zip(keys, values):
                results[key].append(value)
        if failed:
            warnings.warn(f"pyrato could not evaluate the decay in the {', '.join(f'{f:g}' for f in failed)} Hz band(s) "
                          f"({next(iter(failed.values()))}); EDT, T20 and T30 are NaN there.")
        return results

    def _band_from_onset(self, ir_array, center_frequency):
        """Zero-phase octave-band IR starting :data:`_ZERO_PHASE_LEAD` before the direct sound.

        The causal octave filter delays the band-filtered direct sound (by about 18 ms at
        250 Hz and 35 ms at 125 Hz), which moves direct energy past fixed time limits
        measured from the broadband direct sound. Filtering forward and backward removes
        that delay; its ringing is symmetric, so integration starts before the direct sound.

        Returns
        -------
        tuple of (np.ndarray, int)
            The band-filtered IR and the index of the direct sound in it.
        """
        kept = self._bands_from_onset if ir_array is self.ir else None
        if kept is not None and center_frequency in kept:
            return kept[center_frequency]
        fs = self.sampling_rate
        lead = int(round(_ZERO_PHASE_LEAD * fs))
        onset = self._direct_sound_onset(ir_array)
        padded = np.concatenate([np.zeros(lead), ir_array, np.zeros(int(_FILTER_TAIL * fs))])
        forward = self._filter_impulse_response(padded, center_frequency)
        zero_phase = self._filter_impulse_response(forward[::-1], center_frequency)[::-1]
        # The direct sound is at onset + lead in the padded signal; start lead before it
        band = zero_phase[onset:], lead
        if kept is not None:
            kept[center_frequency] = band
        return band

    def clarity_definition(self, impulse_response=None) -> dict:
        """Compute clarity and definition parameters (C50, C80, D50) per frequency band.

        The early (up to 50 or 80 ms after the direct sound, ISO 3382-1) and late parts are
        cut from the broadband impulse response before octave filtering, so filter delay
        and ringing at low frequencies stay with the part they belong to.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        onset = self._direct_sound_onset(ir_array)
        bands = list(self.frequency_bands)
        early_50, early_80 = (onset + int(round(t * self.sampling_rate)) for t in (0.05, 0.08))

        energies = {name: self._band_energies(segment, bands) for name, segment in (
            ('early_50', ir_array[onset:early_50]), ('late_50', ir_array[early_50:]),
            ('early_80', ir_array[onset:early_80]), ('late_80', ir_array[early_80:]),
        )}

        results = {'C50': [], 'C80': [], 'D50': []}
        for freq in bands:
            e_early_50, e_late_50 = energies['early_50'][freq], energies['late_50'][freq]
            e_early_80, e_late_80 = energies['early_80'][freq], energies['late_80'][freq]
            results['C50'].append(round(float(10.0 * np.log10((e_early_50 + 1e-12) / (e_late_50 + 1e-12))), 2))
            results['C80'].append(round(float(10.0 * np.log10((e_early_80 + 1e-12) / (e_late_80 + 1e-12))), 2))
            results['D50'].append(round(float(e_early_50 / (e_early_50 + e_late_50 + 1e-12) * 100.0), 2))
        return results

    def direct_to_reverberant_ratio(self, impulse_response=None) -> dict:
        """Compute the direct-to-reverberant ratio (DRR) per octave band and in the 700-4000 Hz band.

        After Haapaniemi and Lokki (2018), eq. 1: the energy of the first 5 ms from the direct
        sound over the energy after it, in dB. The window opens 1 ms before the direct sound,
        so that a band-limited arrival (a fractional delay spreads over neighbouring samples)
        is held whole; nothing is reflected before the direct sound arrives.

        The single-number value, ``'DRR_overall'``, is theirs: the impulse response filtered
        with a second-order Butterworth band-pass of 700-4000 Hz, the band of LOC (Griesinger)
        and of their localisation threshold in concert halls (-2.7 to 1.7 dB). A 5 ms window
        does not resolve the low octaves, which is why they use this band.

        Per octave band the two parts are cut before filtering, as for C50: the octave filters
        ring for tens of milliseconds at low frequencies and, applied first, would move the
        direct sound past 5 ms. The low octaves therefore see only what arrives in the window;
        a source's own response longer than 5 ms (a high-pass below the band) counts as
        reverberant there, as it does in their definition.

        Returns
        -------
        dict
            ``'DRR'`` per band in :attr:`frequency_bands`, and ``'DRR_overall'`` in 700-4000 Hz.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        onset = self._direct_sound_onset(ir_array)
        start, end = (max(0, onset + int(round(t * self.sampling_rate))) for t in _DRR_DIRECT_WINDOW)

        def ratio(direct, reverberant):
            if direct > 0 and reverberant > 0:
                return round(float(10.0 * np.log10(direct / reverberant)), 2)
            return np.nan

        bands = list(self.frequency_bands)
        direct = self._band_energies(ir_array[start:end], bands)
        reverberant = self._band_energies(ir_array[end:], bands)
        filtered = sosfilt(butter(_DRR_BAND_ORDER, _DRR_BAND, 'bandpass', fs=self.sampling_rate, output='sos'), ir_array)
        return {
            'DRR': [ratio(direct[freq], reverberant[freq]) for freq in bands],
            'DRR_overall': ratio(np.sum(filtered[start:end] ** 2), np.sum(filtered[end:] ** 2)),
        }

    def center_time(self, impulse_response=None) -> dict:
        """Compute the center time Ts per frequency band with pyrato, in seconds.

        Uses ``pyrato.parameters.center_time`` on the energy decay curve of zero-phase
        octave-band IRs (see :meth:`_band_from_onset`), with time measured from the
        broadband direct sound (ISO 3382-1).
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        results = {'Ts': []}
        for freq in self.frequency_bands:
            band, direct = self._band_from_onset(ir_array, freq)
            if not np.any(band):
                results['Ts'].append(np.nan)
                continue
            if _PYRATO_PARAMETERS:
                edc = pyrato.edc.schroeder_integration(pf.Signal(band, self.sampling_rate))
                # pyrato measures from the start of the EDC, which is `direct` samples before the direct sound
                ts = float(np.squeeze(pyrato.parameters.center_time(edc))) - direct / self.sampling_rate
            else:
                energy = band ** 2
                t = (np.arange(band.size) - direct) / self.sampling_rate
                ts = float(np.sum(t * energy) / np.sum(energy))
            results['Ts'].append(round(ts, 4))
        return results

    def speech_transmission_index(self, impulse_response=None, snr=np.inf) -> dict:
        """Compute the Speech Transmission Index (STI) and MTI per octave band.

        Uses pyrato's modulation_transfer_function and speech_transmission_index_indirect
        per IEC 60268-16:2020. Overall STI is computed by pyrato; per-band MTI is derived
        from the MTF via the standard TI conversion.

        IRs shorter than the 1.6 s required by IEC 60268-16:2020 (Section 6.2) are
        zero-padded to that length, which assumes the IR has fully decayed.
        """
        if not _STI_SUPPORTED:
            warnings.warn(
                "STI is not available with pyrato {} (requires pyrato >= 1.0 and Python >= 3.11). "
                "STI and MTI values are set to NaN.".format(pyrato.__version__)
            )
            return {'mti': np.full(7, np.nan), 'sti': np.nan}

        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        min_samples = int(np.ceil(_STI_MIN_DURATION * self.sampling_rate))
        if ir_array.size < min_samples:
            ir_array = np.pad(ir_array, (0, min_samples - ir_array.size))
        sig = pf.Signal(ir_array, self.sampling_rate)
        snr_arg = float(snr) if np.ndim(snr) == 0 else snr
        try:
            # MTF shape: (7, 14) — 7 octave bands x 14 modulation frequencies, values in [0, 1]
            mtf = np.squeeze(np.asarray(pyrato.parameters.modulation_transfer_function(sig, snr=snr_arg)))
            snr_app = 10.0 * np.log10(mtf / (1.0 - mtf + 1e-12) + 1e-12)
            mti = np.mean(np.clip((snr_app + 15.0) / 30.0, 0, 1), axis=-1)  # (7,)
            sti_val = round(float(np.squeeze(pyrato.parameters.speech_transmission_index_indirect(sig, snr=snr_arg))), 4)
        except Exception as e:
            warnings.warn(f"STI computation failed: {e}")
            mti = np.full(7, np.nan)
            sti_val = np.nan
        return {'mti': mti, 'sti': sti_val}

    def _direct_sound_onset(self, ir_array) -> int:
        """Index of the direct sound: where ``ita_start_IR`` (ITA Toolbox) puts t = 0, after ISO 3382-1.

        On the package scale the search starts 1 ms before the geometric arrival: nothing arrives
        earlier, and an early ray-traced bin could otherwise be taken for the direct sound. A blocked
        direct path is treated as the ITA Toolbox treats any response: the start is the first
        arrival the search finds.
        """
        lower = 0
        # The stored geometric onset belongs to the analyser's own response only
        if self.onset is not None and ir_array is self.ir:
            lower = max(0, self.onset - int(round(_GEOMETRIC_SEARCH_LEAD * self.sampling_rate)))
        return lower + _iso3382_start(ir_array[lower:])

    def _band_energies(self, segment, bands) -> dict:
        """Octave-band energies of a time segment, filtered after windowing so filter delay and ringing stay with the segment."""
        padded = np.concatenate([segment, np.zeros(int(_FILTER_TAIL * self.sampling_rate))])
        return {f: float(np.sum(self._filter_impulse_response(padded, f) ** 2)) for f in bands}

    def stage_support(self, impulse_response=None) -> dict:
        """Compute early and late stage support (ST_early, ST_late) per octave band.

        Defined in ISO 3382-1:2009 Annex C after Gade, for a source and receiver 1 m apart
        on stage: the reflected energy 20-100 ms (ST_early) or 100-1000 ms (ST_late) after
        the direct sound, relative to the energy in the first 10 ms (direct sound and floor
        reflection), in dB. The single-number values are the arithmetic mean of the 250,
        500, 1000 and 2000 Hz octaves.

        Each time window is cut from the broadband impulse response before octave
        filtering, so that the delay and ringing of the low-frequency filters do not move
        direct sound energy into the reflection windows.

        Returns
        -------
        dict
            ``'freqs'``, ``'ST_early'`` and ``'ST_late'`` per band in :attr:`frequency_bands`,
            and ``'ST_early_overall'`` and ``'ST_late_overall'``.
        """
        ir_array = self.ir if impulse_response is None else self._as_mono_array(impulse_response)
        fs = self.sampling_rate
        onset = self._direct_sound_onset(ir_array)

        late_end = onset + int(round(_STAGE_SUPPORT_WINDOWS['ST_late'][1] * fs))
        if late_end > len(ir_array):
            warnings.warn(f"Impulse response ends {(len(ir_array) - onset) / fs:.2f} s after the direct sound; "
                          "the ST_late window (100-1000 ms) is truncated.")

        freqs = list(self.frequency_bands)
        bands = list(dict.fromkeys(freqs + _STAGE_SUPPORT_BANDS))

        def segment(window):
            start, stop = (onset + int(round(t * fs)) for t in window)
            return ir_array[start:min(stop, len(ir_array))]

        direct = self._band_energies(segment(_STAGE_DIRECT_WINDOW), bands)
        result = {'freqs': freqs}
        for name, window in _STAGE_SUPPORT_WINDOWS.items():
            reflected = self._band_energies(segment(window), bands)
            values = {f: 10.0 * np.log10(reflected[f] / direct[f]) if reflected[f] > 0 and direct[f] > 0 else np.nan for f in bands}
            result[name] = [round(float(values[f]), 2) for f in freqs]
            result[f'{name}_overall'] = round(float(np.nanmean([values[f] for f in _STAGE_SUPPORT_BANDS])), 2)
        return result

    def sound_strength(self, reference_ir=None, impulse_response=None) -> dict:
        """Compute the sound strength G per octave band with pyrato.

        G is the total energy of the impulse response relative to the energy of a
        free-field impulse response of the same source at 10 m (ISO 3382-1). Both must be
        at their raw level: pass :class:`ImpulseResponse` objects, which are used
        denormalized, or raw arrays. The single-number value averages the 500 and
        1000 Hz octaves.

        Parameters
        ----------
        reference_ir : ImpulseResponse or np.ndarray, optional
            Free-field impulse response of the same source at 10 m, at the same sample rate.
            Without one, the package-scale free field ``1 / (4 pi 10)`` is used, which needs
            the impulse response to be package-scale (as every simulation method returns).
        impulse_response : ImpulseResponse or np.ndarray, optional
            The impulse response to evaluate. Default is :attr:`ir`.

        Returns
        -------
        dict
            ``'freqs'``, ``'G'`` per band in :attr:`frequency_bands`, and ``'G_overall'``.
        """
        ir_array = self.ir if impulse_response is None else self._as_mono_array(impulse_response)
        if reference_ir is None:
            if not self._on_package_scale(impulse_response):
                raise ValueError("G without a reference_ir needs a package-scale ImpulseResponse; pass the free-field response at 10 m.")
            reference = np.zeros(self.sampling_rate)
            reference[int(_ZERO_PHASE_LEAD * self.sampling_rate)] = 1.0 / (4.0 * np.pi * _STRENGTH_DISTANCE)
        else:
            reference_rate = getattr(reference_ir, 'sampling_rate', None)
            if reference_rate is not None and int(reference_rate) != self.sampling_rate:
                raise ValueError(f"Reference impulse response sample rate {reference_rate} Hz differs from {self.sampling_rate} Hz.")
            reference = self._as_mono_array(reference_ir)

        if not _PYRATO_PARAMETERS:
            warnings.warn("Sound strength G needs pyrato >= 1.0 (pyrato {} is installed); G is NaN.".format(pyrato.__version__))
            freqs = list(self.frequency_bands)
            return {'freqs': freqs, 'G': [np.nan] * len(freqs), 'G_overall': np.nan}

        freqs = list(self.frequency_bands)
        bands = list(dict.fromkeys(freqs + _STRENGTH_BANDS))
        tail = np.zeros(int(_FILTER_TAIL * self.sampling_rate))

        def band_edcs(x):
            padded = np.concatenate([x, tail])
            filtered = pf.Signal(np.stack([self._filter_impulse_response(padded, f) for f in bands]), self.sampling_rate)
            return pyrato.edc.schroeder_integration(filtered)

        strength = np.asarray(pyrato.parameters.sound_strength(band_edcs(ir_array), band_edcs(reference))).reshape(-1)
        values = dict(zip(bands, strength))
        return {
            'freqs': freqs,
            'G': [round(float(values[f]), 2) for f in freqs],
            'G_overall': round(float(np.mean([values[f] for f in _STRENGTH_BANDS])), 2),
        }

    # Methods

    def _sti_and_strength(self, ir=None) -> dict:
        """STI and G as :meth:`analyse` gives them: ``'STI'`` and ``'G'`` per band in :attr:`frequency_bands`, and their overall values.

        The worker computes only these, for a client whose pyrato cannot (:func:`compas_acoustics.worker.analyse`).
        """
        if ir is None:
            ir = self.ir
        freqs = list(self.frequency_bands)
        sti_data = self.speech_transmission_index(ir)
        # G needs the absolute level; an array or a normalized-only response has none
        own = None if ir is self.ir else ir
        if _PYRATO_PARAMETERS and self._on_package_scale(own):
            strength = self.sound_strength(impulse_response=own)
            g_vector, g_overall = strength['G'], strength['G_overall']
        else:
            g_vector, g_overall = [np.nan] * len(freqs), np.nan

        sti_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
        sti_vector = [round(sti_data['mti'][sti_bands.index(f)], 2) if f in sti_bands else np.nan for f in freqs]
        return {
            'STI': np.array(sti_vector, dtype=float).tolist(), 'STI_overall': round(float(sti_data['sti']), 4),
            'G': np.array(g_vector, dtype=float).tolist(), 'G_overall': g_overall,
        }

    def analyse(self, ir=None) -> dict:
        if ir is None:
            ir = self.ir
        # reverberation_times and center_time take the same zero-phase bands
        self._bands_from_onset = {}
        try:
            rt = self.reverberation_times(ir)
            clarity = self.clarity_definition(ir)
            drr = self.direct_to_reverberant_ratio(ir)
            ct = self.center_time(ir)
        finally:
            self._bands_from_onset = None
        sti_and_strength = self._sti_and_strength(ir)

        freqs = list(self.frequency_bands)

        param_vectors = {
            'EDT': rt['EDT'], 'T20': rt['T20'], 'T30': rt['T30'],
            'C50': clarity['C50'], 'C80': clarity['C80'], 'D50': clarity['D50'],
            'DRR': drr['DRR'], 'Ts': ct['Ts'], 'STI': sti_and_strength['STI'], 'G': sti_and_strength['G']
        }

        result = {'freqs': freqs}
        for param in _PARAM_NAMES:
            vals = np.array(param_vectors[param], dtype=float)
            result[param] = vals.tolist()
            if param in ('STI', 'G'):
                overall = sti_and_strength[f'{param}_overall']
            elif param == 'DRR':
                overall = drr['DRR_overall']
            else:
                idx = [freqs.index(f) for f in _SINGLE_NUMBER_BANDS]
                overall = round(float(np.nanmean(vals[idx])), 2)
            result[f'{param}_overall'] = overall

        self.analysis_data = result
        return self.analysis_data


    def print_table(self, parameters=None, show_bands=True, show_overall=False):
        """
        Print a table of analysis results.
        Parameters
        ----------
        parameters : str or list of str, optional
            Parameter name(s) to include in the table. If None, all parameters are printed.
        show_bands : bool, optional
            Whether to show per-band values. Default is True.
        show_overall : bool, optional
            Whether to show overall values. Default is False.

        Notes
        -----
        Overall values for EDT, T20, T30, C50, C80, D50 and Ts are averaged over the 500 Hz and 1000 Hz
        bands (ISO 3382-1:2009 Table A.1). STI, DRR and G have their own overall values.

        """
        params = parameters if parameters else _PARAM_NAMES
        if isinstance(params, str):
            params = [params]
        freqs = self.analysis_data['freqs']
        header = (freqs if show_bands else []) + (['Overall'] if show_overall else [])
        print()
        print(f"{'':14}" + "".join(f"{str(f):>8}" for f in header))
        print("-" * (14 + 8 * len(header)))
        for p in params:
            row = (self.analysis_data[p] if show_bands else []) + ([self.analysis_data[f'{p}_overall']] if show_overall else [])
            print(f"{p:14}" + "".join(f"{v:>8.2f}" for v in row))


    def plot(self, parameters=None):
        import matplotlib.pyplot as plt

        if parameters is not None:
            if isinstance(parameters, str):
                parameters = [parameters]
            for param in parameters:
                if param not in _PARAM_NAMES:
                    raise ValueError(f"Parameter '{param}' not found in results. Available parameters: {_PARAM_NAMES}")
        else:
            parameters = _PARAM_NAMES
            
        # Updated units for plot labels only
        units = {
            'EDT': 'Time (s)',
            'T20': 'Time (s)',
            'T30': 'Time (s)',
            'STI': 'Percentage',
            'C50': 'dB',
            'C80': 'dB',
            'DRR': 'dB',
            'D50': 'Percentage',
            'Ts': 'Time (s)',
            'G': 'dB',
        }
        
        for param in parameters:

            y = np.array(self.analysis_data[param], dtype=float)
            y_clean = y[np.isfinite(y)]
            
            if len(y_clean) > 0:
                vmin, vmax = np.min(y_clean), np.max(y_clean)
                
                # Apply 40% padding based on absolute values
                lower_limit = vmin - (abs(vmin) * 0.4)
                upper_limit = vmax + (abs(vmax) * 0.4)
                
                # Apply boundary constraints
                if param == 'STI':
                    lower_limit, upper_limit = max(0.0, lower_limit), min(1.0, upper_limit)
                if param == 'D50':
                    lower_limit, upper_limit = max(0.0, lower_limit), min(100.0, upper_limit)
                
                ylim = [lower_limit, upper_limit]
            else:
                ylim = None

            plt.figure(figsize=(8, 4), num=f"{param}")
            plt.title(f"{param}")
            ax = plt.gca()
            plt.plot([str(f) for f in self.analysis_data['freqs']], y, marker='o', linewidth=2, label=param)
            
            # Formatting logic for RT parameters
            if param in ['EDT', 'T20', 'T30']:
                from matplotlib.ticker import FormatStrFormatter
                from matplotlib.ticker import MultipleLocator
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                if ylim and (ylim[1] - ylim[0]) < 0.4:
                    mid = (ylim[0] + ylim[1]) / 2
                    ylim = [mid - 0.3, mid + 0.3]

            # Y-limit application
            if ylim:
                if ylim[0] == ylim[1]:
                    plt.ylim(ylim[0] - 0.2, ylim[1] + 0.2)
                else:
                    plt.ylim(ylim)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(units.get(param, ''))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    def export_csv(self, filepath: str, parameters=None):
        """Export analysis results to a CSV file.

        Parameters
        ----------
        result : dict
            The dict returned by :meth:`analyzer`.
        filepath : str
            The path where the CSV should be saved.
        parameters : str or list of str, optional
            Parameter name(s) to include in the CSV. 
            If None, all parameters are exported.
        """
        import csv

        if parameters is None:
            parameters = _PARAM_NAMES
        elif isinstance(parameters, str):
            parameters = [parameters]

        freqs_header = [str(f) for f in self.analysis_data['freqs']] + ['Overall']
        header = ['Parameter'] + freqs_header

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for param in parameters:
                if param not in _PARAM_NAMES:
                    warnings.warn(f"Parameter '{param}' not available for CSV export.")
                    continue
                data_row = self.analysis_data[param] + [self.analysis_data[f'{param}_overall']]
                formatted_row = [param] + [round(v, 2) if not np.isnan(v) else "" for v in data_row]
                writer.writerow(formatted_row)
        
        print(f"Selected results successfully saved to: {filepath}")


    def plot_rir(self):
        """Plot the impulse response in the time domain."""
        import matplotlib.pyplot as plt

        t = np.arange(len(self.ir)) / self.sampling_rate
        plt.figure(figsize=(8, 4), num="Impulse Response")
        plt.plot(t, self.ir, label='Impulse Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Impulse Response')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_frequency_response(self, smooth=3, raw_alpha=0.3):
        """Plot the frequency response of a signal with optional octave smoothing."""
        import matplotlib.pyplot as plt

        sig = pf.Signal(self.ir, self.sampling_rate)
        ref = np.max(np.abs(sig.freq))

        plt.figure(figsize=(8, 4), num="Frequency Response")
        ax = plt.gca()

        if smooth is not None:
            pf.plot.freq(sig, log_reference=ref, ax=ax, linewidth=0.5,
                         alpha=raw_alpha, color="gray", label="Raw")
            sm_signal, _ = pf.dsp.smooth_fractional_octave(sig, num_fractions=smooth)
            pf.plot.freq(sm_signal, log_reference=ref, ax=ax, linewidth=1.5,
                         label=f"1/{smooth} octave smoothing")
            ax.legend(loc="best")
        else:
            pf.plot.freq(sig, log_reference=ref, ax=ax, linewidth=1)

        xtick_positions = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        xtick_labels = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
        plt.xticks(xtick_positions, xtick_labels)

        plt.xlim([20, 20000])
        plt.ylim([-60, 10])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dB")
        plt.title("Frequency Response")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.grid(True, which="minor", ls=":", alpha=0.2)
        plt.tight_layout()
        plt.show()

