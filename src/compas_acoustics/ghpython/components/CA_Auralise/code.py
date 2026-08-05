# flake8: noqa
import Grasshopper
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import System
import scriptcontext as sc
import numpy as np
import os

from compas_acoustics.auraliser import Auraliser


class Auralise(Grasshopper.Kernel.GH_ScriptInstance):

    def offer_buttons(self):
        component = ghenv.Component
        offered_key = "Auraliser_buttons_offered_{}".format(component.InstanceGuid)
        if sc.sticky.get(offered_key):
            return
        sc.sticky[offered_key] = True

        doc = component.OnPingDocument()
        if doc is None:
            return

        targets = [
            (index, nickname, comp) for index, nickname, comp in ((2, "Multiconvolve", "tog"), (5, "Save", "but"))
            if component.Params.Input[index].SourceCount == 0
        ]
        if not targets:
            return

        def deferred(gh_doc):
            base = component.Attributes.Pivot
            for row, (index, nickname, comp) in enumerate(targets):
                input_param = component.Params.Input[index]
                if input_param.SourceCount > 0:
                    continue

                if comp == "tog":
                    popout = Grasshopper.Kernel.Special.GH_BooleanToggle()
                elif comp == "but":
                    popout = Grasshopper.Kernel.Special.GH_ButtonObject()
                else:
                    continue
                popout.CreateAttributes()
                popout.NickName = nickname
                popout.Attributes.Pivot = System.Drawing.PointF(base.X - 280, base.Y - 20 + row * 60)
                gh_doc.AddObject(popout, False)
                input_param.AddSource(popout)
                popout.ExpireSolution(True)

        doc.ScheduleSolution(1, deferred)

    def RunScript(self, irs, signals, multiconvolve, save_file_path, save_file_name, save):

        self.offer_buttons()

        if irs is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'irs' failed to collect data."
            )
            return
        
        has_signals = signals is not None and signals.DataCount > 0
        if has_signals:
            if irs.BranchCount != signals.BranchCount:
                raise ValueError("DataTree branch count mismatch.")
            if any(len(irs.Branch(i)) != len(signals.Branch(i)) for i in range(irs.BranchCount)):
                raise ValueError("DataTree branch length mismatch.")

        signals_out = DataTree[object]()
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

                # ir_data = IR.ir
                ir_data = IR.ir[:, np.newaxis] if IR.ir.ndim == 1 else IR.ir

                ir_groups[receiver_id][sub_key] = {
                    'ir': ir_data,
                    'sample_rate': IR.sample_rate,
                    'ambisonic_order': 0,
                    'channel_order': 'acn',
                    'normalisation': 'sn3d'
                }

                sound_groups[receiver_id][sub_key] = {
                    'filepath': sound_file
                }

            # 2. Construct auraliser(s) and convolve signals
            for i, receiver_id in enumerate(ir_groups.keys()):
                if multiconvolve:
                    tasks = [(receiver_id, ir_groups[receiver_id], sound_groups[receiver_id], f"{i}")]
                else:
                    tasks = [
                        (
                            sub_key,
                            {sub_key: ir_groups[receiver_id][sub_key]},
                            {sub_key: sound_groups[receiver_id][sub_key]},
                            f"{i}_{j}",
                        )
                        for j, sub_key in enumerate(ir_groups[receiver_id].keys())
                    ]
                
                for name, ir_data, sound_data, suffix in tasks:
                    auraliser = Auraliser(
                        name=name,
                        impulse_responses=ir_data,
                        anechoic_sounds=sound_data,
                        output_format="mono",
                        # sample_rate=ir_groups[receiver_id][f"source_{0}"]['sample_rate']
                        sample_rate=48000
                    )
                    convolved_matrix = auraliser.multiconvolution()
                    
                    signals_out.Add(convolved_matrix, path)

                    if save:
                        filepath = self.set_save_file_path(save_file_path, save_file_name, suffix)
                        auraliser.export_convolved_sound(filepath, convolved_matrix)
        return signals_out
    
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