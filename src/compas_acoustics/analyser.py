import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from compas.data import Data
from scipy import signal
from scipy.fft import next_fast_len
from scipy.fft import rfft
from scipy.fft import rfftfreq

from compas_acoustics.defaults import Defaults

__all__ = ["AcousticAnalyser"]

# Row order used throughout the class
_PARAM_NAMES = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'DRR', 'Ts', 'STI']

# Parameters whose "Overall" is averaged over 500 & 1000 Hz (ISO 3382-1:2009)
_RT_PARAMS = {'EDT', 'T20', 'T30'}


class AcousticAnalyser(Data):

    def __init__(self, ir):
        self.ir = ir
        self.analysis_data = None
        self.sampling_rate = Defaults().SAMPLING_RATE

    @property
    def ir(self):
        return self._ir
    
    @ir.setter
    def ir(self, ir):
        if not isinstance(ir, (list, np.ndarray)):
            raise TypeError("Impulse response must be a list or numpy array.")
        self._ir = np.asarray(ir, dtype=float)

    @property
    def sampling_rate(self):
        return self._sampling_rate

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
        nyquist = 0.5 * self.sampling_rate
        lower_cutoff = center_frequency / math.sqrt(2)
        upper_cutoff = center_frequency * math.sqrt(2)
        low = lower_cutoff / nyquist
        high = upper_cutoff / nyquist
        if high >= 1.0:
            high = 0.99
        sos = signal.butter(6, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, impulse_response)


    def _schroeder_integration(self, impulse_response: list):
        ir = np.asarray(impulse_response)
        edc = np.cumsum(ir[::-1] ** 2)[::-1]
        edc_db = 10.0 * np.log10(edc / np.max(edc) + 1e-12)
        t = np.arange(len(edc_db)) / self.sampling_rate
        return edc_db, t


    def _smooth_octave(self, mag_db, freqs, fraction=3):
        f = 2.0 ** (1.0 / (2.0 * fraction))
        i_low = np.searchsorted(freqs, freqs / f, 'left')
        i_high = np.searchsorted(freqs, freqs * f, 'right')
        
        cmag = np.r_[0.0, np.cumsum(mag_db)]
        counts = i_high - i_low
        
        smoothed = np.where(counts > 0, (cmag[i_high] - cmag[i_low]) / np.maximum(counts, 1), mag_db)
        smoothed[0] = mag_db[0]
        
        return smoothed


    # Calculate acoustic parameters

    def compute_rt(self, impulse_response=None) -> dict:
        """Compute reverberation times (EDT, T20, T30) per frequency band."""
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        if ir_array.ndim > 1:
            ir_array = ir_array[:, 0]
        
        freqs = self.frequency_bands
        results = {'EDT': [], 'T20': [], 'T30': []}

        for center_freq in freqs:
            filtered_ir = self._filter_impulse_response(ir_array, center_freq)
            decay_curve, t = self._schroeder_integration(filtered_ir)

            for key, (low, high) in zip(['EDT', 'T20', 'T30'], [(0, -10), (-5, -25), (-5, -35)]):
                mask = (decay_curve <= low) & (decay_curve >= high)
                if np.sum(mask) > 1:
                    p = np.polyfit(t[mask], decay_curve[mask], 1)
                    results[key].append(round(-60.0 / p[0], 2))
                else:
                    results[key].append(np.nan)
        return results

    def compute_clarity_definition(self, impulse_response=None) -> dict:
        """Compute clarity and definitions parameters (C50, C80, D50) per frequency band."""
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        freqs = self.frequency_bands
        results = {'C50': [], 'C80': [], 'D50': []}

        broadband_er = ir_array ** 2
        peak_idx = int(np.argmax(broadband_er))
        s50 = min(peak_idx + int(0.05 * self.sampling_rate), len(ir_array))
        s80 = min(peak_idx + int(0.08 * self.sampling_rate), len(ir_array))

        for center_freq in freqs:
            filtered_ir = self._filter_impulse_response(ir_array, center_freq)
            er = filtered_ir ** 2
            e50, l50 = np.sum(er[peak_idx:s50]), np.sum(er[s50:])
            e80, l80 = np.sum(er[peak_idx:s80]), np.sum(er[s80:])

            results['C50'].append(round(10.0 * np.log10((e50 + 1e-12) / (l50 + 1e-12)), 2))
            results['C80'].append(round(10.0 * np.log10((e80 + 1e-12) / (l80 + 1e-12)), 2))
            results['D50'].append(round((e50 / (e50 + l50 + 1e-12)) * 100.0, 2))
        return results

    def compute_drr(self, impulse_response=None) -> dict:
        """Compute Direct-to-Reverberant Ratio (DRR) per frequency band."""

        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        peak_idx = int(np.argmax(ir_array**2))
        d_win = int(0.0025 * self.sampling_rate)
        l_start = int(0.05 * self.sampling_rate)
        
        s_dir, e_dir = max(0, peak_idx - d_win//2), min(len(ir_array), peak_idx + d_win//2)
        s_late = min(len(ir_array), peak_idx + l_start)

        results = {'DRR': []}
        for freq in self.frequency_bands:
            filtered = self._filter_impulse_response(ir_array, freq)
            er = filtered**2
            results['DRR'].append(round(10.0 * np.log10(np.sum(er[s_dir:e_dir]) / (np.sum(er[s_late:]) + 1e-12)), 2))
        return results

    def center_time(self, impulse_response=None) -> dict:

        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        t = np.arange(len(ir_array)) / self.sampling_rate
        results = {'Ts': []}
        for freq in self.frequency_bands:
            filtered = self._filter_impulse_response(ir_array, freq)
            er = filtered**2
            results['Ts'].append(round(np.sum(t * er) / (np.sum(er) + 1e-12), 4))
        return results

    def compute_sti(self, impulse_response=None, snr=np.inf) -> dict:
        """Compute the Speech Transmission Index (STI) and Modulation Transfer Index (MTI) per frequency band."""

        if impulse_response is None:
            impulse_response = self.ir
        
        ir_array = np.asarray(impulse_response, dtype=float)
        octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
        mod_freqs = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5])
        alpha = np.array([0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143])

        if np.ndim(snr) == 0:
            snr = np.full(7, float(snr))
        t = np.arange(len(ir_array)) / self.sampling_rate
        mti_values = []

        for i, center_freq in enumerate(octave_bands):
            filtered = self._filter_impulse_response(ir_array, center_freq)
            h2 = filtered ** 2
            total_e = np.sum(h2)
            if total_e == 0: 
                mti_values.append(0.0)
                continue
                
            m_band = []
            for fm in mod_freqs:
                m_val = np.abs(np.sum(h2 * np.exp(-1j * 2 * np.pi * fm * t))) / total_e
                if not np.isinf(snr[i]):
                    m_val *= (1.0 / (1.0 + 10 ** (-snr[i] / 10.0)))
                m_band.append(m_val)
            
            snr_app = 10.0 * np.log10(np.array(m_band) / (1.0 - np.array(m_band) + 1e-12) + 1e-12)
            ti = np.clip((snr_app + 15.0) / 30.0, 0, 1)
            mti_values.append(np.mean(ti))

        mti_values = np.array(mti_values)
        return {'mti': mti_values, 'sti': round(np.sum(alpha * mti_values), 4)}

    # Methods

    def analyse(self, ir=None) -> dict:
        if ir is None:
            ir = self.ir
        rt = self.compute_rt(ir)
        clarity = self.compute_clarity_definition(ir)
        drr = self.compute_drr(ir)
        ct = self.center_time(ir)
        sti_data = self.compute_sti(ir)

        freqs = list(self.frequency_bands)
        
        # STI uses specific bands [125-8000]. We map those to our current frequency_bands.
        sti_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
        sti_freq_mapped = []
        for f in freqs:
            if f in sti_bands:
                sti_freq_mapped.append(round(sti_data['mti'][sti_bands.index(f)], 2))
            else:
                sti_freq_mapped.append(np.nan)

        param_vectors = {
            'EDT': rt['EDT'], 'T20': rt['T20'], 'T30': rt['T30'],
            'C50': clarity['C50'], 'C80': clarity['C80'], 'D50': clarity['D50'],
            'DRR': drr['DRR'], 'Ts': ct['Ts'], 'STI': sti_freq_mapped
        }

        raw = np.array([param_vectors[p] for p in _PARAM_NAMES], dtype=float)
        overall = np.empty(len(_PARAM_NAMES))
        for i, param in enumerate(_PARAM_NAMES):
            if param == 'STI':
                overall[i] = sti_data['sti']
            else:
                idx = [freqs.index(500), freqs.index(1000)] if param in _RT_PARAMS else [freqs.index(500), freqs.index(1000), freqs.index(2000)]
                overall[i] = round(float(np.nanmean(raw[i, idx])), 2)

        self.analysis_data = {'data': np.hstack([raw, overall.reshape(-1, 1)]), 'params': _PARAM_NAMES, 'freqs': freqs}
        return self.analysis_data


    def print_table(self, parameters=None):
        params = parameters if parameters else self.analysis_data['params']
        if isinstance(params, str): params = [params]
        header = self.analysis_data['freqs'] + ['Overall']
        print(f"{'':14}" + "".join(f"{str(f):>8}" for f in header))
        print("-" * (14 + 8 * len(header)))
        for p in params:
            row = self.analysis_data['data'][self.analysis_data['params'].index(p)]
            print(f"{p:14}" + "".join(f"{v:>8.2f}" for v in row))


    def plot(self, parameters=None):

        if parameters is not None:
            if isinstance(parameters, str):
                parameters = [parameters]
            for param in parameters:
                if param not in self.analysis_data['params']:
                    raise ValueError(f"Parameter '{param}' not found in results. Available parameters: {self.analysis_data['params']}")
        else:
            parameters = self.analysis_data['params']
            
        # Updated units for plot labels only
        units = {
            'EDT': 'EDC (s)', 
            'T20': 'T20 (s)',
            'T30': 'T30 (s)', 
            'STI': 'Speech Transmission Index', 
            'C50': 'Clarity (dB)', 
            'C80': 'Clarity (dB)', 
            'DRR': 'DRR (dB)', 
            'D50': 'D50 (%)', 
            'Ts': 'Time (s)'
        }
        
        for param in parameters:

            row_idx = self.analysis_data['params'].index(param)
            y = self.analysis_data['data'][row_idx, :-1]
            y_clean = y[~np.isnan(y)]
            
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

            plt.figure(figsize=(8, 4))
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

        # Parameter selection
        if parameters is None:
            parameters = self.analysis_data['params']
        elif isinstance(parameters, str):
            parameters = [parameters]

        freqs_header = [str(f) for f in self.analysis_data['freqs']] + ['Overall']
        header = ['Parameter'] + freqs_header

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for param in parameters:
                if param not in self.analysis_data['params']:
                    warnings.warn(f"Parameter '{param}' not available for CSV export.")
                    continue
                
                # Get the row index and the corresponding data
                idx = self.analysis_data['params'].index(param)
                data_row = self.analysis_data['data'][idx].tolist()
                
                # Format row: Label + rounded values (or empty string if NaN)
                formatted_row = [param] + [
                    round(v, 2) if not np.isnan(v) else "" for v in data_row
                ]
                writer.writerow(formatted_row)
        
        print(f"Selected results successfully saved to: {filepath}")


    def plot_rir(self):
        """Plot the impulse response in the time domain."""

        t = np.arange(len(self.ir)) / self.sampling_rate
        plt.figure(figsize=(8, 4))
        plt.plot(t, self.ir, label='Impulse Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Impulse Response in Time Domain')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_frequency_response(self, smooth=3, raw_alpha=0.3):
        """Plot the frequency response of a signal with optional octave smoothing."""

        Nfft = next_fast_len(len(self.ir))
        IR_fft = rfft(self.ir, n=Nfft)
        freqs = rfftfreq(Nfft, 1 / self.sampling_rate)

        mag_db = 20 * np.log10(np.abs(IR_fft) + 1e-12)
        mag_db = mag_db - np.max(mag_db)

        plt.figure(figsize=(8, 4))

        if smooth is not None:
            plt.semilogx(freqs, mag_db, linewidth=0.5, alpha=raw_alpha, color="gray", label="Raw")
            mag_db_smooth = self._smooth_octave(mag_db, freqs, fraction=smooth)
            plt.semilogx(freqs, mag_db_smooth, linewidth=1.5, label=f"1/{smooth} octave smoothing")
            plt.legend(loc="best")
        else:
            plt.semilogx(freqs, mag_db, linewidth=1)

        xtick_positions = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        xtick_labels = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]
        plt.xticks(xtick_positions, xtick_labels)

        plt.xlim([20, 20000])
        plt.ylim([-60, 10])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Frequency Response")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.grid(True, which="minor", ls=":", alpha=0.2)
        plt.tight_layout()
        plt.show()

