import warnings
import numpy as np
import pyfar as pf
import pyrato
import matplotlib.pyplot as plt
from compas.data import Data
from compas_acoustics.defaults import Defaults

__all__ = ["AcousticAnalyser"]

# Row order used throughout the class
_PARAM_NAMES = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'DRR', 'Ts', 'STI']

# Parameters whose "Overall" is averaged over 500 & 1000 Hz (ISO 3382-1:2009)
_RT_PARAMS = {'EDT', 'T20', 'T30'}


class AcousticAnalyser(Data):
    """Class for analyzing acoustic parameters from an impulse response (IR).
    
    Parameters
    ----------
    ir : list or np.ndarray
        The impulse response to analyze. Can be a 1D or 2D array (mono or multi-channel).
    """

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
    
    @property
    def frequency_bands(self):
        # Return custom bands if set, otherwise return defaults
        if hasattr(self, '_frequency_bands'):
            return self._frequency_bands
        return Defaults().DEFAULT_FREQUENCY_BANDS

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
        filt = pf.dsp.filter.fractional_octave_bands(
            signal=None,
            num_fractions=1,
            sampling_rate=self.sampling_rate,
            frequency_range=(center_frequency * 0.9, center_frequency * 1.1),
            order=14,
        )
        return filt.process(sig).time.squeeze()


    # Calculate acoustic parameters

    def compute_rt(self, impulse_response=None) -> dict:
        """Compute reverberation times (EDT, T20, T30) per frequency band.

        Uses pyrato's plain Schroeder backward integration (no noise truncation;
        assumes noise-free RIRs) and manual ISO 3382-1 linear regression. pyrato's
        own regression is bypassed because it does not guard against log10(0) on
        zero-tail EDCs.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        if ir_array.ndim > 1:
            ir_array = ir_array[:, 0]

        results = {'EDT': [], 'T20': [], 'T30': []}
        for center_freq in self.frequency_bands:
            filtered_ir = self._filter_impulse_response(ir_array, center_freq)
            sig = pf.Signal(filtered_ir, self.sampling_rate)
            edc_obj = pyrato.edc.schroeder_integration(sig)
            edc_lin = edc_obj.time.squeeze()
            t = edc_obj.times
            edc_db = 10.0 * np.log10(np.abs(edc_lin) / (np.max(np.abs(edc_lin)) + 1e-30) + 1e-30)

            for key, (low, high) in zip(['EDT', 'T20', 'T30'], [(0, -10), (-5, -25), (-5, -35)]):
                mask = (edc_db <= low) & (edc_db >= high)
                if np.sum(mask) > 1:
                    p = np.polyfit(t[mask], edc_db[mask], 1)
                    results[key].append(round(-60.0 / p[0], 2))
                else:
                    results[key].append(np.nan)
        return results

    def compute_clarity_definition(self, impulse_response=None) -> dict:
        """Compute clarity and definition parameters (C50, C80, D50) per frequency band.

        Uses pyrato's Schroeder EDC trimmed to the direct sound onset, then computes
        energy ratios manually. pyrato's own clarity/definition are bypassed because
        they do not guard against log10(0) on zero-tail EDCs.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        # Use broadband peak as onset reference across all bands (ISO 3382-1).
        peak_idx = int(np.argmax(ir_array ** 2))
        results = {'C50': [], 'C80': [], 'D50': []}
        for center_freq in self.frequency_bands:
            filtered_ir = self._filter_impulse_response(ir_array, center_freq)
            sig = pf.Signal(filtered_ir[peak_idx:], self.sampling_rate)
            try:
                edc_obj = pyrato.edc.schroeder_integration(sig)
                edc_lin = edc_obj.time.squeeze()   # backward cumsum; edc_lin[0] = total energy
                t = edc_obj.times                   # seconds from direct sound onset

                total_e = edc_lin[0]
                idx_50 = min(np.searchsorted(t, 0.05), len(edc_lin) - 1)
                idx_80 = min(np.searchsorted(t, 0.08), len(edc_lin) - 1)

                e_late_50, e_late_80 = edc_lin[idx_50], edc_lin[idx_80]
                e_early_50 = total_e - e_late_50
                e_early_80 = total_e - e_late_80

                results['C50'].append(round(10.0 * np.log10((e_early_50 + 1e-12) / (e_late_50 + 1e-12)), 2))
                results['C80'].append(round(10.0 * np.log10((e_early_80 + 1e-12) / (e_late_80 + 1e-12)), 2))
                results['D50'].append(round((e_early_50 / (total_e + 1e-12)) * 100.0, 2))
            except Exception:
                results['C50'].append(np.nan)
                results['C80'].append(np.nan)
                results['D50'].append(np.nan)
        return results

    def compute_drr(self, impulse_response=None) -> dict:
        """Compute Direct-to-Reverberant Ratio (DRR) per frequency band."""
        
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        if ir_array.ndim > 1:
            ir_array = ir_array[:, 0]

        d_half = int(0.0025 * self.sampling_rate)

        results = {'DRR': []}
        for freq in self.frequency_bands:
            filtered = self._filter_impulse_response(ir_array, freq)
            er = filtered ** 2
            peak_idx = int(np.argmax(er))
            i_dir_start = max(0, peak_idx - d_half)
            i_dir_end = min(len(er), peak_idx + d_half)

            e_dir = np.sum(er[i_dir_start:i_dir_end])
            e_rev = np.sum(er[i_dir_end:])
            if e_dir > 0 and e_rev > 0:
                results['DRR'].append(round(10.0 * np.log10(e_dir / e_rev), 2))
            else:
                results['DRR'].append(np.nan)
        return results

    def center_time(self, impulse_response=None) -> dict:

        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        # t=0 at direct-sound onset (ISO 3382-1), consistent with clarity/definition.
        peak_idx = int(np.argmax(ir_array ** 2))
        results = {'Ts': []}
        for freq in self.frequency_bands:
            filtered = self._filter_impulse_response(ir_array, freq)[peak_idx:]
            er = filtered ** 2
            t = np.arange(len(er)) / self.sampling_rate
            results['Ts'].append(round(np.sum(t * er) / (np.sum(er) + 1e-12), 4))
        return results

    def compute_sti(self, impulse_response=None, snr=np.inf) -> dict:
        """Compute the Speech Transmission Index (STI) and MTI per octave band.

        Uses pyrato's modulation_transfer_function and speech_transmission_index_indirect
        per IEC 60268-16:2020. Overall STI is computed by pyrato; per-band MTI is derived
        from the MTF via the standard TI conversion.
        """
        if impulse_response is None:
            impulse_response = self.ir

        ir_array = np.asarray(impulse_response, dtype=float)
        sig = pf.Signal(ir_array, self.sampling_rate)
        snr_arg = float(snr) if np.ndim(snr) == 0 else snr
        try:
            # MTF shape: (7, 14) — 7 octave bands x 14 modulation frequencies, values in [0, 1]
            mtf = np.squeeze(np.asarray(pyrato.parameters.modulation_transfer_function(sig, snr=snr_arg)))
            snr_app = 10.0 * np.log10(mtf / (1.0 - mtf + 1e-12) + 1e-12)
            mti = np.mean(np.clip((snr_app + 15.0) / 30.0, 0, 1), axis=-1)  # (7,)
            sti_val = round(float(np.squeeze(pyrato.parameters.speech_transmission_index_indirect(sig, snr=snr_arg))), 4)
        except Exception:
            mti = np.full(7, np.nan)
            sti_val = np.nan
        return {'mti': mti, 'sti': sti_val}

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

        result = {'freqs': freqs}
        for param in _PARAM_NAMES:
            vals = np.array(param_vectors[param], dtype=float)
            result[param] = vals.tolist()
            if param == 'STI':
                overall = round(float(sti_data['sti']), 4)
            else:
                idx = [freqs.index(500), freqs.index(1000)] if param in _RT_PARAMS else [freqs.index(500), freqs.index(1000), freqs.index(2000)]
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
        Overall values for EDT, T20, and T30 are averaged over 500 Hz and 1000 Hz bands (ISO 3382-1:2009).
        Overall values for other parameters are averaged over 500 Hz, 1000 Hz, and 2000 Hz bands.

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
            'Ts': 'Time (s)'
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

