# flake8: noqa
import Grasshopper
import System
from Grasshopper import DataTree
import numpy as np
import os
import scriptcontext as sc

from compas_acoustics import Defaults
from compas_acoustics.auraliser import Auraliser

# First item is the default
OUTPUT_FORMATS = ["stereo", "binaural", "mono", "ambisonics"]
HRTFS = {
    "kemar_normal_pinna": "mit_kemar_normal_pinna.sofa",
    "kemar_large_pinna": "mit_kemar_large_pinna.sofa",
}


class Auralise(Grasshopper.Kernel.GH_ScriptInstance):

    def offer_value_lists(self):
        component = ghenv.Component
        offered_key = "auralise_value_lists_offered_{}".format(component.InstanceGuid)
        if sc.sticky.get(offered_key):
            return
        sc.sticky[offered_key] = True

        doc = component.OnPingDocument()
        if doc is None:
            return

        value_lists = {"output_format": OUTPUT_FORMATS, "hrtf": list(HRTFS)}
        targets = [
            (index, param.Name, value_lists[param.Name])
            for index, param in enumerate(component.Params.Input)
            if param.Name in value_lists and param.SourceCount == 0
        ]
        if not targets:
            return

        def deferred(gh_doc):
            base = component.Attributes.Pivot
            for index, name, items in targets:
                input_param = component.Params.Input[index]
                if input_param.SourceCount > 0:
                    continue

                value_list = Grasshopper.Kernel.Special.GH_ValueList()
                value_list.CreateAttributes()
                value_list.NickName = name
                value_list.ListItems.Clear()
                for item in items:
                    value_list.ListItems.Add(Grasshopper.Kernel.Special.GH_ValueListItem(item, '"{}"'.format(item)))
                value_list.Attributes.Pivot = System.Drawing.PointF(base.X - 220, input_param.Attributes.Pivot.Y - 10)
                gh_doc.AddObject(value_list, False)
                input_param.AddSource(value_list)
                value_list.ExpireSolution(True)

        doc.ScheduleSolution(1, deferred)

    def RunScript(self, irs, signals, multiconvolve, save_file_path, save_file_name, save, output_format, hrtf):
        self.offer_value_lists()

        if irs is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'irs' failed to collect data."
            )
            return None, None

        output_format = output_format or OUTPUT_FORMATS[0]
        hrtf_path = self.resolve_hrtf_path(hrtf)
        if hrtf_path is None:
            return None, None

        has_signals = signals is not None and signals.DataCount > 0
        if has_signals:
            if irs.BranchCount != signals.BranchCount:
                raise ValueError("DataTree branch count mismatch.")
            if any(len(irs.Branch(i)) != len(signals.Branch(i)) for i in range(irs.BranchCount)):
                raise ValueError("DataTree branch length mismatch.")

        signals_out = DataTree[object]()
        sample_rates_out = DataTree[object]()
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

                ir_data = IR.ir[:, np.newaxis] if IR.ir.ndim == 1 else IR.ir

                ir_groups[receiver_id][sub_key] = {
                    'ir': ir_data,
                    'sample_rate': IR.sample_rate,
                    'ambisonic_order': int(round(np.sqrt(ir_data.shape[1]))) - 1,
                    'channel_order': 'acn',
                    'normalisation': 'sn3d'
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

                    convolved = auraliser.convolve()

                    signals_out.Add(convolved, path)
                    sample_rates_out.Add(auraliser.sample_rate, path)

                    if save:
                        filepath = self.set_save_file_path(save_file_path, save_file_name, suffix)
                        auraliser.export_convolved_sound(filepath, convolved)

        if output_format == "ambisonics":
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                "Ambisonic signals cannot be played by the Audio Player. Save them as a multichannel file, or use binaural for headphone playback."
            )

        return signals_out, sample_rates_out

    @staticmethod
    def resolve_hrtf_path(hrtf):
        """Return the SOFA file for a bundled HRTF name or a file path. Default is the KEMAR normal pinna."""
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
