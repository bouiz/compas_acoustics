import numpy as np
from compas_acoustics.ir import ImpulseResponse



def _direct_sound_index(ir: np.ndarray) -> int:
    """Index of the direct-sound onset (broadband peak)."""
    return int(np.argmax(ir ** 2))


def compute_ST1(impulse_response: ImpulseResponse) -> float:
    """Early stage support (ST1 / STearly), ISO 3382-1 Annex A.

    Ratio of reflected energy in the 20-100 ms window to the direct-sound
    energy in the first 10 ms, both measured from the direct-sound onset.
    Source and receiver are assumed to already be positioned on stage
    (e.g. 1 m or 4 m apart) when the impulse response was simulated.
    """
    ir = np.asarray(impulse_response.ir, dtype=float)
    fs = impulse_response.sample_rate
    onset = _direct_sound_index(ir)

    i_dir_end = onset + int(round(0.010 * fs))
    i_early_start = onset + int(round(0.020 * fs))
    i_early_end = onset + int(round(0.100 * fs))

    e_dir = np.sum(ir[onset:i_dir_end] ** 2)
    e_early = np.sum(ir[i_early_start:min(i_early_end, len(ir))] ** 2)

    if e_dir <= 0 or e_early <= 0:
        return float('nan')
    return round(10.0 * np.log10(e_early / e_dir), 2)


def compute_ST2(impulse_response: ImpulseResponse) -> float:
    """Late stage support (ST2 / STlate / STtotal), ISO 3382-1 Annex A.

    Ratio of reflected energy in the 100-1000 ms window to the direct-sound
    energy in the first 10 ms.
    """
    ir = np.asarray(impulse_response.ir, dtype=float)
    fs = impulse_response.sample_rate
    onset = _direct_sound_index(ir)

    i_dir_end = onset + int(round(0.010 * fs))
    i_late_start = onset + int(round(0.100 * fs))
    i_late_end = onset + int(round(1.000 * fs))

    e_dir = np.sum(ir[onset:i_dir_end] ** 2)
    e_late = np.sum(ir[i_late_start:min(i_late_end, len(ir))] ** 2)

    if e_dir <= 0 or e_late <= 0:
        return float('nan')
    return round(10.0 * np.log10(e_late / e_dir), 2)


def compute_G_stage(impulse_response: ImpulseResponse, reference_ir: ImpulseResponse) -> float:
    """Stage strength (G_stage), ISO 3382-1 Annex A.

    Sound strength on stage relative to a free-field reference impulse
    response, simulated with the same source at 10 m and no shell/scene
    geometry present.

    Parameters
    ----------
    impulse_response : ImpulseResponse
        The on-stage impulse response to evaluate.
    reference_ir : ImpulseResponse
        Free-field, 10 m reference impulse response for the same source.
    """
    ir = np.asarray(impulse_response.ir, dtype=float)
    ref = np.asarray(reference_ir.ir, dtype=float)

    e_total = np.sum(ir ** 2)
    e_ref = np.sum(ref ** 2)

    if e_total <= 0 or e_ref <= 0:
        return float('nan')
    return round(10.0 * np.log10(e_total / e_ref), 2)


def compute_ITDG(impulse_response: ImpulseResponse, threshold_db: float = -20.0) -> float:
    """Initial time-delay gap (ITDG), in milliseconds.

    Time between the direct-sound onset and the first reflection whose
    magnitude exceeds `threshold_db` relative to the direct-sound peak.
    """
    ir = np.asarray(impulse_response.ir, dtype=float)
    fs = impulse_response.sample_rate
    onset = _direct_sound_index(ir)

    direct_level = np.abs(ir[onset])
    if direct_level <= 0:
        return float('nan')

    # Skip past the direct-sound sample itself before searching for the first reflection.
    i_dir_end = onset + int(round(0.001 * fs))
    tail = ir[i_dir_end:]
    if len(tail) == 0:
        return float('nan')

    threshold = direct_level * 10 ** (threshold_db / 20.0)
    above_threshold = np.where(np.abs(tail) >= threshold)[0]
    if len(above_threshold) == 0:
        return float('nan')

    first_reflection_idx = i_dir_end + above_threshold[0]
    return round((first_reflection_idx - onset) / fs * 1000.0, 3)


def stage_support_parameters(impulse_response: ImpulseResponse, reference_ir: ImpulseResponse = None) -> dict:
    """Compute ST1, ST2, ITDG and (optionally) G_stage directly from an
    ImpulseResponse instance.

    Parameters
    ----------
    impulse_response : ImpulseResponse
        The on-stage impulse response to evaluate.
    reference_ir : ImpulseResponse, optional
        Free-field, 10 m reference impulse response for the same source,
        required to compute G_stage. If omitted, G_stage is left out.

    Returns
    -------
    dict
        Keys: 'ST1', 'ST2', 'ITDG', and 'G_stage' (if `reference_ir` given).
    """
    result = {
        'ST1': compute_ST1(impulse_response),
        'ST2': compute_ST2(impulse_response),
        'ITDG': compute_ITDG(impulse_response),
    }
    if reference_ir is not None:
        result['G_stage'] = compute_G_stage(impulse_response, reference_ir)
    return result