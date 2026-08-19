from compas.geometry import Frame


# -------------------------------
# String orchestra layouts
# -------------------------------

def string_quartet(base_frame=None):
    """Four named points (violin_1, violin_2, viola, cello) on base_frame.

    Each offset is [u, v, w] in the frame's local coordinates -- w carries
    the instrument's playing height (e.g. 1.15 m for violin/viola, 0.75 m
    for cello).
    """
    base_frame = base_frame or Frame.worldXY()

    positions = {
        "violin_1": [-1.3, 0.4, 1.15],
        "violin_2": [-0.6, 1.4, 1.15],
        "viola":    [ 0.6, 1.4, 1.15],
        "cello":    [ 1.3, 0.4, 0.75],
    }

    names = list(positions.keys())
    points = [base_frame.to_world_coordinates(xyz) for xyz in positions.values()]
    return points, names


def chamber_strings(base_frame=None):
    """Sixteen named points for a chamber string section on base_frame.

    Same [u, v, w] convention as :func:`string_quartet`.
    """
    base_frame = base_frame or Frame.worldXY()

    positions = {
        # Violin I (5 players)
        "violin_1_1": [-1.8, 0.8, 1.15],
        "violin_1_2": [-2.4, 1.4, 1.15],
        "violin_1_3": [-1.4, 1.6, 1.15],
        "violin_1_4": [-3.0, 2.0, 1.15],
        "violin_1_5": [-2.0, 2.2, 1.15],

        # Violin II (4 players)
        "violin_2_1": [-0.9, 1.8, 1.15],
        "violin_2_2": [-1.5, 2.6, 1.15],
        "violin_2_3": [-0.6, 2.8, 1.15],
        "violin_2_4": [-1.2, 3.4, 1.15],

        # Viola (3 players)
        "viola_1":    [ 0.6, 1.8, 1.15],
        "viola_2":    [ 1.2, 2.6, 1.15],
        "viola_3":    [ 0.4, 3.0, 1.15],

        # Cello (3 players)
        "cello_1":    [ 1.6, 0.8, 0.75],
        "cello_2":    [ 2.2, 1.5, 0.75],
        "cello_3":    [ 1.8, 2.3, 0.75],

        # Double Bass (1 player)
        "double_bass_1": [2.8, 2.8, 1.40],
    }

    names = list(positions.keys())
    points = [base_frame.to_world_coordinates(xyz) for xyz in positions.values()]
    return points, names


def medium_chamber_strings(base_frame=None):
    """Twenty-four named points for a larger chamber string section on base_frame.

    Same [u, v, w] convention as :func:`string_quartet`.
    """
    base_frame = base_frame or Frame.worldXY()

    positions = {
        # Violin I (8 players)
        "violin_1_1": [-2.0, 0.8, 1.15], "violin_1_2": [-2.6, 1.3, 1.15],
        "violin_1_3": [-1.6, 1.6, 1.15], "violin_1_4": [-3.2, 1.9, 1.15],
        "violin_1_5": [-2.2, 2.2, 1.15], "violin_1_6": [-3.8, 2.5, 1.15],
        "violin_1_7": [-2.8, 2.8, 1.15], "violin_1_8": [-3.4, 3.4, 1.15],

        # Violin II (6 players)
        "violin_2_1": [-1.0, 1.8, 1.15], "violin_2_2": [-1.6, 2.5, 1.15],
        "violin_2_3": [-0.6, 2.8, 1.15], "violin_2_4": [-1.2, 3.5, 1.15],
        "violin_2_5": [-0.3, 3.8, 1.15], "violin_2_6": [-0.9, 4.4, 1.15],

        # Viola (4 players)
        "viola_1": [0.6, 1.8, 1.15], "viola_2": [1.2, 2.5, 1.15],
        "viola_3": [0.6, 3.2, 1.15], "viola_4": [1.2, 3.8, 1.15],

        # Cello (4 players)
        "cello_1": [1.8, 0.8, 0.75], "cello_2": [2.5, 1.4, 0.75],
        "cello_3": [2.0, 2.2, 0.75], "cello_4": [2.7, 2.8, 0.75],

        # Double Bass (2 players)
        "double_bass_1": [3.2, 3.2, 1.40], "double_bass_2": [3.8, 4.0, 1.40],
    }

    names = list(positions.keys())
    points = [base_frame.to_world_coordinates(xyz) for xyz in positions.values()]
    return points, names


def symphonic_strings(base_frame=None):
    """Fifty named points for a full symphonic string section on base_frame.

    Same [u, v, w] convention as :func:`string_quartet`.
    """
    base_frame = base_frame or Frame.worldXY()

    positions = {
        # Violin I (14 players)
        "violin_1_1":  [-2.2, 0.8, 1.15], "violin_1_2":  [-3.0, 1.2, 1.15],
        "violin_1_3":  [-2.0, 1.7, 1.15], "violin_1_4":  [-3.8, 1.8, 1.15],
        "violin_1_5":  [-2.8, 2.3, 1.15], "violin_1_6":  [-4.6, 2.4, 1.15],
        "violin_1_7":  [-3.6, 2.9, 1.15], "violin_1_8":  [-5.4, 3.0, 1.15],
        "violin_1_9":  [-4.4, 3.5, 1.15], "violin_1_10": [-6.0, 3.8, 1.15],
        "violin_1_11": [-5.0, 4.3, 1.15], "violin_1_12": [-6.6, 4.6, 1.15],
        "violin_1_13": [-5.6, 5.1, 1.15], "violin_1_14": [-6.2, 5.8, 1.15],

        # Violin II (12 players)
        "violin_2_1":  [-1.2, 2.0, 1.15], "violin_2_2":  [-1.9, 2.6, 1.15],
        "violin_2_3":  [-0.9, 3.0, 1.15], "violin_2_4":  [-1.6, 3.6, 1.15],
        "violin_2_5":  [-2.3, 4.0, 1.15], "violin_2_6":  [-1.2, 4.4, 1.15],
        "violin_2_7":  [-2.0, 4.8, 1.15], "violin_2_8":  [-2.7, 5.2, 1.15],
        "violin_2_9":  [-1.6, 5.6, 1.15], "violin_2_10": [-2.4, 6.0, 1.15],
        "violin_2_11": [-3.1, 6.3, 1.15], "violin_2_12": [-2.0, 6.7, 1.15],

        # Viola (10 players)
        "viola_1": [0.8, 2.0, 1.15], "viola_2": [1.5, 2.6, 1.15],
        "viola_3": [0.6, 3.0, 1.15], "viola_4": [1.3, 3.6, 1.15],
        "viola_5": [2.0, 4.0, 1.15], "viola_6": [0.9, 4.4, 1.15],
        "viola_7": [1.7, 4.8, 1.15], "viola_8": [2.4, 5.2, 1.15],
        "viola_9": [1.3, 5.6, 1.15], "viola_10": [2.0, 6.0, 1.15],

        # Cello (8 players)
        "cello_1": [2.0, 0.8, 0.75], "cello_2": [2.8, 1.2, 0.75],
        "cello_3": [2.4, 1.9, 0.75], "cello_4": [3.3, 2.1, 0.75],
        "cello_5": [3.0, 2.8, 0.75], "cello_6": [3.9, 3.0, 0.75],
        "cello_7": [3.6, 3.8, 0.75], "cello_8": [4.5, 4.0, 0.75],

        # Double Bass (6 players)
        "double_bass_1": [4.5, 4.8, 1.40], "double_bass_2": [5.3, 5.2, 1.40],
        "double_bass_3": [4.2, 5.8, 1.40], "double_bass_4": [5.0, 6.2, 1.40],
        "double_bass_5": [3.9, 6.8, 1.40], "double_bass_6": [4.7, 7.2, 1.40],
    }

    names = list(positions.keys())
    points = [base_frame.to_world_coordinates(xyz) for xyz in positions.values()]
    return points, names
