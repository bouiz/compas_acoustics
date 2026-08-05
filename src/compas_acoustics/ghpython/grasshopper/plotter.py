from ...analyser import AcousticAnalyser
from ...analyser import _PARAM_NAMES
from ...ir import ImpulseResponse
import Rhino.Geometry as rg  # type: ignore
import math
from typing import Optional


__all__ = ["AcousticParameter", "Plotter2D", "Plotter2DSettings",]


# ----------------------------------------
# Mathematical helper functions
# ----------------------------------------
def order_of_magnitude(value):
    return 10 ** math.floor(math.log10(value))

def remap_number(value, from_min, from_max, to_min, to_max):
    return to_min + (to_max - to_min) * ((value - from_min) / (from_max - from_min))

def calculate_ticks(values, max_value=None):
    if not values:
        return [0.0]
    
    raw_min = min(values)
    raw_max = max(values)

    if max_value is not None and max_value < raw_max:
        raise ValueError(f"Provide max_value {max_value} is less than the data maximum value {raw_max}.")
    
    if max_value is not None and math.isclose(max_value, 1.0) and raw_min >= 0.0:
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    if max_value is None:
        val_range = raw_max - raw_min
        if math.isclose(val_range, 0.0):
            val_range = abs(raw_max) if not math.isclose(raw_max, 0.0) else 1.0
        # find the order of magnitude
        magnitude = order_of_magnitude(val_range)

        approx_max = math.ceil(raw_max / magnitude) * magnitude
        approx_min = math.floor(raw_min / magnitude) * magnitude
    else:
        approx_max = max_value
        val_range = approx_max - raw_min
        magnitude = order_of_magnitude(val_range)
        approx_min = math.floor(raw_min / magnitude) * magnitude

    graph_range = approx_max - approx_min
    if math.isclose(graph_range, 0.0):
        return [approx_min]

    # Heckbert's Label Placement Algorithm to determine a "nice" step size for ticks
    segment = 5
    raw_step = graph_range / segment
    step_magnitude = order_of_magnitude(raw_step)
    significand = raw_step / step_magnitude

    if significand < 1.5:
        nice_step = 1.0 * step_magnitude
    elif significand < 3.0:
        nice_step = 2.0 * step_magnitude
    elif significand < 7.0:
        nice_step = 5.0 * step_magnitude
    else:
        nice_step = 10.0 * step_magnitude

    # Snap the range to whole multiples of the chosen step so the generated
    # ticks always reach (or exceed) raw_max instead of stopping short of it
    # (approx_min/approx_max above are rounded to the value's own order of
    # magnitude, not to nice_step, so they don't always land on a tick).
    nice_min = math.floor(raw_min / nice_step) * nice_step
    if math.isclose(nice_min, raw_min):
        # Data's lowest point would land exactly on the axis minimum, i.e. the
        # plotted curve would sit right on top of the x-axis line. Drop the
        # axis one more step below so the curve stays visually clear of it.
        nice_min -= nice_step
    nice_max = approx_max if max_value is not None else math.ceil(raw_max / nice_step) * nice_step

    if nice_step < 1.0:
        decimal = max(0, -math.floor(math.log10(nice_step)))
    else:
        decimal = 0

    ticks = []
    current_tick = nice_min
    while current_tick <= nice_max + (nice_step * 0.001):
        rounded = round(current_tick, decimal)
        if decimal == 0:
            rounded = int(rounded)

        ticks.append(rounded)
        current_tick += nice_step
    return ticks

# ----------------------------------------
# Helper classes for Rhino plotting
# ----------------------------------------
class AcousticParameter:
    """Class representing an acoustic parameter with its values and overall metric."""

    _TITLE_NAMES = [
        'Early Decay Time (EDT)', 'Reverberation Time (T20)', 'Reverberation Time (T30)',
        'Clarity (C50)', 'Clarity (C80)', 'Definition (D50)',
        'Direct-to-Reverberant Ratio (DRR)', 'Center Time (Ts)', 'Speech Transmission Index (STI)',
    ]
    _X_UNITS = [
        'Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)',
        'Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)',
        'Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)',
    ]

    _Y_UNITS = [
        'Time (s)', 'Time (s)', 'Time (s)',
        'dB', 'dB', 'Value (%)',
        'dB', 'Time (s)', 'Index (0-1)',
    ]

    def __init__(
        self,
        name=None,
        frequencies=None,
        values=None, 
        overall=None,
        title=None,
        x_unit=None,
        y_unit=None,
        source_name=None,
        receiver_name=None
    ):
        self.name = name
        self.frequencies = frequencies
        self.values = values
        self.overall = overall
        
        self.title = title
        self.x_unit = x_unit
        self.y_unit = y_unit

        self.source_name = source_name
        self.receiver_name = receiver_name

    def __str__(self):
        return f"{self.title}, {self.source_name} -> {self.receiver_name}"
    
    @classmethod
    def from_analyser(cls, analyser: AcousticAnalyser, name: str):
        if name not in _PARAM_NAMES:
            raise ValueError(f"Parameter '{name}' not found in analyser results. Available parameters: {_PARAM_NAMES}")

        frequencies = analyser.frequency_bands
        values = analyser.analysis_data[name]
        overall = analyser.analysis_data[f"{name}_overall"]
        title = cls._TITLE_NAMES[_PARAM_NAMES.index(name)]
        x_unit = cls._X_UNITS[_PARAM_NAMES.index(name)]
        y_unit = cls._Y_UNITS[_PARAM_NAMES.index(name)]
        # TODO: Add source and receiver names, but it needs to modify the AcousticAnalyser class
        return cls(name, frequencies, values, overall, title, x_unit, y_unit)

    @classmethod
    def from_ir(cls, ir: ImpulseResponse):
        """A duck-typed constructor to create an AcousticParameter from an ImpulseResponse object."""
        return cls(
            name="IR", values=ir.ir, overall=None, title="Impulse Response (IR)", x_unit="Time (s)", y_unit="Amplitude"
        )


class Plotter2DSettings:
    """Class for storing settings for the 2D plotter."""

    def __init__(
        self,
        plane=None,
        width=None,
        height=None,
        title_height=None,
        axis_title_height=None,
        label_height=None,
        title_offset=None,
        x_axis_title_offset=None,
        y_axis_title_offset=None,
        x_label_offset=None,
        y_label_offset=None,
        data_offset=None,
        title_justification=None,
        x_axis_title_justification=None,
        y_axis_title_justification=None,
        x_label_justification=None,
        y_label_justification=None,
        data_justification=None,
        font=None,
    ):
        self.plane = plane if plane is not None else rg.Plane.WorldXY
        self.width = width if width is not None else 10.0
        self.height = height if height is not None else 5.0
        self.title_height = title_height if title_height is not None else self.height * 0.05
        self.axis_title_height = axis_title_height if axis_title_height is not None else self.height * 0.04
        self.label_height = label_height if label_height is not None else self.height * 0.03
    
        self.title_justification = title_justification if title_justification is not None else rg.TextJustification.BottomCenter
        self.x_axis_title_justification = x_axis_title_justification if x_axis_title_justification is not None else rg.TextJustification.TopCenter
        self.y_axis_title_justification = y_axis_title_justification if y_axis_title_justification is not None else rg.TextJustification.BottomCenter
        self.x_label_justification = x_label_justification if x_label_justification is not None else rg.TextJustification.TopCenter
        self.y_label_justification = y_label_justification if y_label_justification is not None else rg.TextJustification.MiddleRight
        self.data_justification = data_justification if data_justification is not None else rg.TextJustification.BottomLeft


class Plotter2D:
    """Class for plotting 2D acoustic parameter as chart in Rhino."""

    def __init__(self, param: AcousticParameter, settings: Optional[Plotter2DSettings] = None):
        self.param = param
        self.settings = settings if settings is not None else Plotter2DSettings()
    
        self.plane = self.settings.plane
        self.width = self.settings.width
        self.height = self.settings.height

        # Rhino geometries for axes and ticks
        self.x_axis_line = None
        self.y_axis_line = None
        self.x_axis_points = []
        self.y_axis_points = []

        # Rhino geometries
        self.data_points = []
        self.data_polyline = None

        # Rhino text entity objects for title and axis labels
        self.title = None
        self.x_axis_title = None
        self.y_axis_title = None
        self.x_tick_labels = []
        self.y_tick_labels = []
        self.data_values = []

    @property
    def values(self):
        return self.param.values

    @property
    def overall(self):
        return self.param.overall

    @property
    def frequencies(self):
        return self.param.frequencies
        
    @staticmethod
    def text_entity(text, plane, font=None, text_height=None, justification=rg.TextJustification.MiddleCenter):
        txt = rg.TextEntity()
        txt.Plane = plane
        txt.PlainText = text
        # txt.Font = font
        txt.TextHeight = text_height
        txt.Justification = justification
        
        return txt

    def plot(self, plane=None, width=None, height=None):

        # 0. Initialize parameters
        if plane is None:
            plane = self.settings.plane
        if width is None:
            width = self.settings.width
        if height is None:
            height = self.settings.height
        self.plane = plane
        self.width = width
        self.height = height
        title_height = self.settings.title_height
        axis_title_height = self.settings.axis_title_height
        label_height = self.settings.label_height

        # 1. Draw x & y axis lines
        x_dir = plane.XAxis
        y_dir = plane.YAxis
        self.x_axis_line = rg.Line(plane.Origin, plane.Origin + x_dir * width)
        self.y_axis_line = rg.Line(plane.Origin, plane.Origin + y_dir * height)

        # 2. Draw x & y axis points
        count_x = len(self.frequencies)
        y_ticks = calculate_ticks(self.values)
        count_y = len(y_ticks)

        step_x = width / (count_x - 1)
        step_y = height / (count_y - 1)

        self.x_tick_labels = [str(freq) for freq in self.frequencies]
        self.y_tick_labels = [str(tick) for tick in y_ticks]

        self.x_axis_points = [plane.Origin + (x_dir * step_x * i) for i in range(count_x)]
        self.y_axis_points = [plane.Origin + (y_dir * step_y * i) for i in range(count_y)]

        # 3. Draw data points and polyline
        y_values = [remap_number(value, min(y_ticks), max(y_ticks), 0, height) for value in self.values]
        self.data_points = [pt + y_dir * y for pt, y in zip(self.x_axis_points, y_values)]
        self.data_polyline = rg.Polyline()
        self.data_polyline.AddRange(self.data_points)

        # 4. Draw texts such as title, x & y axis titles, and tick labels
        pln_title = rg.Plane(plane)
        pln_title.Origin = plane.Origin + x_dir * (width * .5) + y_dir * (height * 1.1)
        self.title = self.text_entity(
            self.param.title, pln_title, text_height=title_height, justification=self.settings.title_justification
        )

        pln_x_title = rg.Plane(plane)
        pln_x_title.Origin = plane.Origin + x_dir * (width * .5) + y_dir * (height * -.1)
        self.x_axis_title = self.text_entity(
            self.param.x_unit, pln_x_title, text_height=axis_title_height, justification=self.settings.x_axis_title_justification
        )

        pln_y_title = rg.Plane(plane)
        pln_y_title.XAxis = plane.YAxis
        pln_y_title.YAxis = -plane.XAxis
        pln_y_title.Origin = plane.Origin + x_dir * (width * -.1) + y_dir * (height * .5)
        self.y_axis_title = self.text_entity(
            self.param.y_unit, pln_y_title, text_height=axis_title_height, justification=self.settings.y_axis_title_justification
        )

        self.x_tick_labels = [
            self.text_entity(
                str(tick),
                plane=rg.Plane(pt + y_dir * -.1, plane.XAxis, plane.YAxis),
                text_height=label_height,
                justification=self.settings.x_label_justification
            )
            for tick, pt in zip(self.frequencies, self.x_axis_points)
        ]

        self.y_tick_labels = [
            self.text_entity(
                str(tick),
                plane=rg.Plane(pt + x_dir * -.1, plane.XAxis, plane.YAxis),
                text_height=label_height,
                justification=self.settings.y_label_justification
            )
            for tick, pt in zip(y_ticks, self.y_axis_points)
        ]

        self.data_values = [
            self.text_entity(
                str(value),
                plane=rg.Plane(pt + y_dir * .1, plane.XAxis, plane.YAxis),
                text_height=label_height,
                justification=self.settings.data_justification
            )
            for value, pt in zip(self.values, self.data_points)
        ]

        return (
            self.x_axis_line,
            self.y_axis_line,
            self.x_axis_points,
            self.y_axis_points,
            self.data_points,
            self.data_polyline,
            self.title,
            self.x_axis_title,
            self.y_axis_title,
            self.x_tick_labels,
            self.y_tick_labels,
            self.data_values,
        )