import warnings

import Grasshopper
from Grasshopper import DataTree
import numpy as np
import os

from compas_acoustics import Defaults
from compas_acoustics import worker
from compas_acoustics.auraliser import Auraliser
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
OUTPUT_FORMATS = ["stereo", "binaural", "mono", "ambisonics"]
LEVELS = ["relative", "individual", "none"]
HEADROOM_DB = -1.0
HRTFS = {
    "kemar": "KEMAR_FreeFieldComp_48kHz.sofa",
}


class Auralise(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, irs, signals, output_format, hrtf, level, multiconvolve, save_file_path, save_file_name, save):
        ensure_inputs(ghenv.Component, {"output_format": OUTPUT_FORMATS, "hrtf": list(HRTFS), "level": LEVELS})

        if irs is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'irs' failed to collect data."
            )
            return None, None

        output_format = output_format or OUTPUT_FORMATS[0]
        level = level or LEVELS[0]
        if level not in LEVELS:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, f"Input parameter 'level' must be one of {LEVELS}."
            )
            return None, None
        hrtf_path = self.resolve_hrtf_path(hrtf)
        if hrtf_path is None:
            return None, None

        has_signals = signals is not None and signals.DataCount > 0
        if has_signals:
            if irs.BranchCount != signals.BranchCount:
                raise ValueError("DataTree branch count mismatch.")
            if any(len(irs.Branch(i)) != len(signals.Branch(i)) for i in range(irs.BranchCount)):
                raise ValueError("DataTree branch length mismatch.")

        # Renders of this run: (path, auraliser, signal, suffix). Their output level is set together.
        renders = []
        # 1. Data conversions
        for i in range(irs.BranchCount):
            path = irs.Paths[i]

            br = irs.Branch(i)
            sound_files = signals.Branch(i) if has_signals else [ir.source.sound_file for ir in br]

            ir_groups = {}
            sound_groups = {}

            for idx, (IR, sound_file) in enumerate(zip(br, sound_files)):
                receiver_id = str(IR.receiver.guid)

                ir_groups.setdefault(receiver_id, {})
                sound_groups.setdefault(receiver_id, {})

                sub_key = f"source_{idx}"

                # Auralise at the raw level so level differences between impulse responses are kept
                raw_ir = IR.denormalized()
                ir_data = raw_ir[:, np.newaxis] if raw_ir.ndim == 1 else raw_ir

                order = getattr(IR, 'ambisonic_order', None)
                ir_groups[receiver_id][sub_key] = {
                    'ir': ir_data,
                    'sampling_rate': IR.sampling_rate,
                    'ambisonic_order': int(round(np.sqrt(ir_data.shape[1]))) - 1 if order is None else int(order),
                    # The response's own convention, e.g. a hybrid response in N3D
                    'channel_order': getattr(IR, 'channel_order', None) or 'acn',
                    'normalization': getattr(IR, 'normalization', None) or 'sn3d',
                    'binaural': (IR.generation or {}).get('output') == 'binaural',
                }

                sound_groups[receiver_id][sub_key] = {
                    'filepath': sound_file
                }

            # 2. Construct auraliser(s), convolve and decode signals
            for r, receiver_id in enumerate(ir_groups.keys()):
                if multiconvolve:
                    tasks = [(receiver_id, ir_groups[receiver_id], sound_groups[receiver_id], f"{r}")]
                else:
                    tasks = [
                        (
                            sub_key,
                            {sub_key: ir_groups[receiver_id][sub_key]},
                            {sub_key: sound_groups[receiver_id][sub_key]},
                            f"{r}_{j}",
                        )
                        for j, sub_key in enumerate(ir_groups[receiver_id].keys())
                    ]

                for name, ir_data, sound_data, suffix in tasks:
                    # Building the auraliser warns too: impulse responses resampled to a common rate, sounds mixed down to mono
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            # Sample rate defaults to that of the impulse responses
                            auraliser = Auraliser(
                                name=name,
                                impulse_responses=ir_data,
                                anechoic_sounds=sound_data,
                                output_format=output_format,
                            )
                        except ValueError as e:
                            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                            return None, None
                        auraliser.hrtf_path = hrtf_path
                        try:
                            signal = worker.convolve(auraliser)
                        except ValueError as e:
                            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                            return None, None
                    for text in dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning)):
                        ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, text)
                    renders.append((path, auraliser, signal, suffix))

        # 3. Output level: 'relative' applies one gain to all renders of this run
        gains = Auraliser.render_gains([signal for _, _, signal, _ in renders], level, HEADROOM_DB)

        signals_out = DataTree[object]()
        sampling_rates_out = DataTree[object]()
        peak_dbfs = -np.inf
        for (path, auraliser, signal, suffix), gain in zip(renders, gains):
            leveled = signal * gain
            peak_dbfs = max(peak_dbfs, Auraliser.peak_dbfs(leveled))
            signals_out.Add(leveled, path)
            sampling_rates_out.Add(auraliser.sampling_rate, path)

            if save:
                filepath = self.set_save_file_path(save_file_path, save_file_name, suffix)
                auraliser.export_convolved_sound(filepath, signal, gain=gain)

        ghenv.Component.Message = "{}, {}, peak {:.1f} dBFS".format(output_format, level, peak_dbfs)
        if peak_dbfs > 0:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Output peaks at {:+.1f} dBFS and will clip on playback. Use level 'relative' or 'individual'.".format(peak_dbfs)
            )

        if output_format == "ambisonics":
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                "Ambisonic signals cannot be played by the Audio Player. Save them as a multichannel file, or use binaural for headphone playback."
            )

        return signals_out, sampling_rates_out

    @staticmethod
    def resolve_hrtf_path(hrtf):
        """Return the SOFA file for a bundled HRTF name or a file path. Default is the KEMAR set."""
        if not hrtf:
            hrtf = list(HRTFS)[0]
        if hrtf in HRTFS:
            return os.path.join(Defaults().HRTF_LIBRARY_PATH, HRTFS[hrtf])
        if os.path.isfile(hrtf) and hrtf.lower().endswith(".sofa"):
            return hrtf
        ghenv.Component.AddRuntimeMessage(
            Grasshopper.Kernel.GH_RuntimeMessageLevel.Error,
            f"Input parameter 'hrtf' must be one of {list(HRTFS)} or the path to a .sofa file."
        )
        return None

    @staticmethod
    def set_save_file_path(save_file_path=None, save_file_name=None, suffix=0):
        if save_file_path:
            base_dir = save_file_path
        else:
            base_dir = os.path.dirname(ghenv.Component.OnPingDocument().FilePath)

        if save_file_name:
            base_filename = save_file_name
        else:
            base_filename = "convolved_sound"
        return os.path.join(base_dir, f"{base_filename}_mic_{suffix}.wav")
