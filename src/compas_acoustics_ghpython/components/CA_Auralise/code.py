import warnings

import Grasshopper
from Grasshopper import DataTree
import numpy as np
import os

from compas_acoustics.auraliser import Auraliser
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
LEVELS = ["relative", "individual", "none"]
HEADROOM_DB = -1.0


class Auralise(Grasshopper.Kernel.GH_ScriptInstance):

    @staticmethod
    def ir_format(ir):
        """'binaural', 'ambisonic' or 'mono', as the impulse response was simulated."""
        if (ir.generation or {}).get('output') == 'binaural':
            return 'binaural'
        raw = ir.denormalized()
        return 'ambisonic' if raw.ndim == 2 and raw.shape[1] > 1 else 'mono'

    def RunScript(self, irs, anechoic_sounds, level, mix_sources, save_file_path, save_file_name, save):
        ensure_inputs(ghenv.Component, {"level": LEVELS})

        if irs is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'irs' failed to collect data."
            )
            return None, None

        level = level or LEVELS[0]
        if level not in LEVELS:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, f"Input parameter 'level' must be one of {LEVELS}."
            )
            return None, None

        has_sounds = anechoic_sounds is not None and anechoic_sounds.DataCount > 0
        if has_sounds:
            if irs.BranchCount != anechoic_sounds.BranchCount:
                raise ValueError("DataTree branch count mismatch.")
            if any(len(irs.Branch(i)) != len(anechoic_sounds.Branch(i)) for i in range(irs.BranchCount)):
                raise ValueError("DataTree branch length mismatch.")

        # Renders of this run: (path, auraliser, signal, suffix). Their output level is set together.
        renders = []
        # 1. Data conversions
        for i in range(irs.BranchCount):
            path = irs.Paths[i]

            br = irs.Branch(i)
            sound_files = anechoic_sounds.Branch(i) if has_sounds else [ir.source.sound_file for ir in br]

            ir_groups = {}
            sound_groups = {}
            formats = {}

            for idx, (IR, sound_file) in enumerate(zip(br, sound_files)):
                receiver_id = str(IR.receiver.guid)

                ir_groups.setdefault(receiver_id, {})
                sound_groups.setdefault(receiver_id, {})

                sub_key = f"source_{idx}"
                formats[sub_key] = self.ir_format(IR)

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

            # 2. Construct auraliser(s) and convolve
            for r, receiver_id in enumerate(ir_groups.keys()):
                if mix_sources:
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
                    task_formats = sorted({formats[sub_key] for sub_key in ir_data})
                    if len(task_formats) > 1:
                        ghenv.Component.AddRuntimeMessage(
                            Grasshopper.Kernel.GH_RuntimeMessageLevel.Error,
                            "The impulse responses at receiver {} are {}: mix responses of one format, or turn mix_sources off.".format(
                                r, " and ".join(task_formats)
                            ),
                        )
                        return None, None
                    (ir_format,) = task_formats
                    # Building the auraliser warns too: impulse responses resampled to a common rate, sounds mixed down to mono
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            # Sample rate defaults to that of the impulse responses
                            auraliser = Auraliser(name=name, impulse_responses=ir_data, anechoic_sounds=sound_data)
                            signal = auraliser.convolve()
                        except ValueError as e:
                            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                            return None, None
                    for text in dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning)):
                        ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, text)
                    renders.append((path, auraliser, signal, suffix, ir_format))

        # 3. Output level: 'relative' applies one gain to all renders of this run
        gains = Auraliser.render_gains([render[2] for render in renders], level, HEADROOM_DB)

        signals_out = DataTree[object]()
        sampling_rates_out = DataTree[object]()
        peak_dbfs = -np.inf
        for (path, auraliser, signal, suffix, _), gain in zip(renders, gains):
            leveled = signal * gain
            peak_dbfs = max(peak_dbfs, Auraliser.peak_dbfs(leveled))
            signals_out.Add(leveled, path)
            sampling_rates_out.Add(auraliser.sampling_rate, path)

            if save:
                filepath = self.set_save_file_path(save_file_path, save_file_name, suffix)
                auraliser.export_convolved_sound(filepath, signal, gain=gain)

        out_formats = sorted({render[4] for render in renders})
        ghenv.Component.Message = "{}, {}, peak {:.1f} dBFS".format(" + ".join(out_formats), level, peak_dbfs)
        if peak_dbfs > 0:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Output peaks at {:+.1f} dBFS and will clip on playback. Use level 'relative' or 'individual'.".format(peak_dbfs)
            )

        if "ambisonic" in out_formats:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                "Ambisonic signals (ACN/SN3D) cannot be played by the Audio Player. Save them as a multichannel file for an ambisonic decoder, "
                "or simulate with Simulate Hybrid output binaural for headphone playback."
            )

        return signals_out, sampling_rates_out

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
