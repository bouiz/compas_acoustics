import math
from typing import Optional

import numpy as np
import Rhino.Geometry as rg  # type: ignore

from compas_acoustics.analyser import AcousticAnalyser
from compas_acoustics.analyser import frequency_response
from compas_acoustics.analyser import time_response
from compas_acoustics.ir import ImpulseResponse

# The frequency axis of a frequency response, and its ticks at the octave centres
FREQUENCY_AXIS = (20.0, 20000.0)
OCTAVE_TICKS = [(31.5, "31.5"), (63, "63"), (125, "125"), (250, "250"), (500, "500"),
                (1000, "1k"), (2000, "2k"), (4000, "4k"), (8000, "8k"), (16000, "16k")]

# A curve is drawn with the minimum and maximum of each of this many bins along the x axis
CURVE_BINS = 1000

__all__ = ["AcousticParameter", "Plotter2D", "Plotter2DSettings",]


# ----------------------------------------
# Mathematical helper functions
# ----------------------------------------
def order_of_magnitude(value):
    return 10 ** math.floor(math.log10(value))

def remap_number(value, from_min, from_max, to_min, to_max):
    return to_min + (to_max - to_min) * ((value - from_min) / (from_max - from_min))

def calculate_nice_step(graph_range, segment=5):
    """Heckbert's Label Placement Algorithm: a "nice" step that cuts ``graph_range`` in about ``segment`` steps."""
    raw_step = graph_range / segment
    step_magnitude = order_of_magnitude(raw_step)
    significand = raw_step / step_magnitude

    if significand < 1.5:
        return 1.0 * step_magnitude
    elif significand < 3.0:
        return 2.0 * step_magnitude
    elif significand < 7.0:
        return 5.0 * step_magnitude
    return 10.0 * step_magnitude

def linear_ticks(low, high):
    """Nice ticks from at or below ``low`` to at or above ``high``, e.g. for a time axis starting at 0."""
    if math.isclose(high, low):
        return [low]
    step = calculate_nice_step(high - low)
    decimal = max(0, -math.floor(math.log10(step)))
    first, last = math.floor(low / step + 1e-9), math.ceil(high / step - 1e-9)
    return [round(i * step, decimal) for i in range(first, last + 1)]

def min_max_indices(positions, values, bins=CURVE_BINS):
    """The indices of the minimum and maximum of ``values`` in each of ``bins`` along ``positions`` (0 to 1), in order.

    Drawn through these, a curve of many more points than the chart is wide looks as it does at full resolution.
    """
    positions, values = np.asarray(positions, dtype=float), np.asarray(values, dtype=float)
    if len(values) <= 2 * bins:
        return np.arange(len(values))
    index = np.clip((positions * bins).astype(int), 0, bins - 1)
    order = np.lexsort((values, index))
    sorted_index = index[order]
    starts = np.searchsorted(sorted_index, np.arange(bins), "left")
    ends = np.searchsorted(sorted_index, np.arange(bins), "right")
    full = ends > starts
    return np.unique(np.concatenate([order[starts[full]], order[ends[full] - 1]]))

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

    nice_step = calculate_nice_step(graph_range)

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

    # name: (title, x unit, y unit). STI and G need pyrato >= 1.0, which Rhino 8's Python 3.9 cannot install: Analyse IR
    # computes them in the worker, and leaves them out without one. The bands of STI are its modulation transfer indices.
    _PARAMETERS = {
        'EDT': ('Early Decay Time (EDT)', 'Frequency (Hz)', 'Time (s)'),
        'T20': ('Reverberation Time (T20)', 'Frequency (Hz)', 'Time (s)'),
        'T30': ('Reverberation Time (T30)', 'Frequency (Hz)', 'Time (s)'),
        'C50': ('Clarity (C50)', 'Frequency (Hz)', 'dB'),
        'C80': ('Clarity (C80)', 'Frequency (Hz)', 'dB'),
        'D50': ('Definition (D50)', 'Frequency (Hz)', 'Value (%)'),
        'DRR': ('Direct-to-Reverberant Ratio (DRR)', 'Frequency (Hz)', 'dB'),
        'Ts': ('Center Time (Ts)', 'Frequency (Hz)', 'Time (s)'),
        'STI': ('Speech Transmission Index (STI)', 'Frequency (Hz)', 'Index (0-1)'),
        'G': ('Sound Strength (G)', 'Frequency (Hz)', 'dB'),
    }
    NAMES = tuple(_PARAMETERS)

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
        receiver_name=None,
        x_scale="bands",
        channels=None,
    ):
        self.name = name
        # The x values: frequency bands, or the times or frequencies of a curve
        self.frequencies = frequencies
        # One value per band; for a curve (channels, points)
        self.values = values
        self.overall = overall

        self.title = title
        self.x_unit = x_unit
        self.y_unit = y_unit

        self.source_name = source_name
        self.receiver_name = receiver_name

        # 'bands' plots one labelled point per band; a curve is 'linear' (time) or 'log' (frequency)
        self.x_scale = x_scale
        self.channels = channels

    def __str__(self):
        # A Grasshopper panel shows this
        if self.frequencies is None:
            return str(self.title)
        if self.x_scale != "bands":
            return "{}: {} {:g} to {:g}, {} points".format(
                self.title, self.x_unit, self.frequencies[0], self.frequencies[-1], len(self.frequencies)
            )
        bands = ", ".join(f"{f:g} Hz {v:.3g}" for f, v in zip(self.frequencies, self.values))
        text = f"{self.name}, {self.y_unit}: {bands}"
        if self.overall is not None:
            text += f"; overall {self.overall:.3g}"
        if self.source_name or self.receiver_name:
            text += f" ({self.source_name} -> {self.receiver_name})"
        return text
    
    @classmethod
    def from_analyser(cls, analyser: AcousticAnalyser, name: str):
        if name not in cls._PARAMETERS:
            raise ValueError(f"Parameter '{name}' is not available in Grasshopper. Available parameters: {list(cls.NAMES)}")

        frequencies = analyser.frequency_bands
        values = analyser.analysis_data[name]
        overall = analyser.analysis_data[f"{name}_overall"]
        title, x_unit, y_unit = cls._PARAMETERS[name]
        # TODO: Add source and receiver names, but it needs to modify the AcousticAnalyser class
        return cls(name, frequencies, values, overall, title, x_unit, y_unit)

    @staticmethod
    def _plotted_channels(ir: ImpulseResponse):
        """The raw channels a curve shows, (samples, channels), and their names: W of an ambisonic response, both ears of a binaural one."""
        data = ir.denormalized()
        data = data[:, np.newaxis] if data.ndim == 1 else data
        if (ir.generation or {}).get("output") == "binaural":
            return data[:, :2], ["left", "right"]
        if data.shape[1] > 1:
            return data[:, :1], ["W"]
        return data, []

    @staticmethod
    def _curve_title(title, channels):
        return "{} ({})".format(title, ", ".join(channels)) if channels else title

    @classmethod
    def from_time_response(cls, ir: ImpulseResponse, db: bool = True):
        """The impulse response against time, as a curve for Plot 2D: its level in dB re the peak, or its amplitude."""
        data, channels = cls._plotted_channels(ir)
        times, values = time_response(data, ir.sampling_rate, db=db)
        return cls(
            name="IR", frequencies=times, values=values, title=cls._curve_title("Impulse Response", channels),
            x_unit="Time (s)", y_unit="Level (dB)" if db else "Amplitude", x_scale="linear", channels=channels,
        )

    @classmethod
    def from_frequency_response(cls, ir: ImpulseResponse):
        """The 1/3-octave smoothed magnitude response from 20 Hz to 20 kHz, in dB re the peak, as a curve for Plot 2D."""
        data, channels = cls._plotted_channels(ir)
        frequencies, values = frequency_response(data, ir.sampling_rate, num_fractions=3, frequency_range=FREQUENCY_AXIS)
        return cls(
            name="FR", frequencies=frequencies, values=values, title=cls._curve_title("Frequency Response, 1/3 octave", channels),
            x_unit="Frequency (Hz)", y_unit="Level (dB)", x_scale="log", channels=channels,
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
    def text_entity(text, plane, font=None, text_height=None, justification=rg.TextJustification.MiddleCenter, draw_forward=True):
        txt = rg.TextEntity()
        txt.Plane = plane
        txt.PlainText = text
        # txt.Font = font
        txt.TextHeight = text_height
        txt.Justification = justification
        if not draw_forward:
            # Rhino otherwise turns the text towards the reader, whatever direction its plane gives it
            txt.DrawForward = False
        
        return txt

    def _plot_curve(self, plane, width, height):
        """The x ticks and the polylines of a curve, one per channel; returns the x labels and the y ticks.

        The x axis is linear from nice ticks around the data (time), or logarithmic over
        :data:`FREQUENCY_AXIS` with ticks at the octave centres (frequency).
        """
        x = np.asarray(self.frequencies, dtype=float)
        values = np.atleast_2d(np.asarray(self.values, dtype=float))
        if self.param.x_scale == "log":
            low, high = FREQUENCY_AXIS
            ticks = [(f, label) for f, label in OCTAVE_TICKS if low <= f <= high]

            def position(v):
                return (np.log10(v) - math.log10(low)) / (math.log10(high) - math.log10(low))
        else:
            tick_values = linear_ticks(float(x[0]), float(x[-1]))
            ticks = [(t, "{:g}".format(t)) for t in tick_values]
            low, high = tick_values[0], tick_values[-1]

            def position(v):
                return (np.asarray(v, dtype=float) - low) / (high - low)

        positions = position(x)
        kept = [min_max_indices(positions, channel) for channel in values]
        y_ticks = calculate_ticks([float(v) for channel, keep in zip(values, kept) for v in channel[keep]])
        y_low, y_high = min(y_ticks), max(y_ticks)

        self.x_axis_points = [plane.Origin + plane.XAxis * (width * float(position(t))) for t, _ in ticks]
        self.data_points = []
        self.data_polyline = []
        for channel, keep in zip(values, kept):
            polyline = rg.Polyline()
            polyline.AddRange([
                plane.Origin + plane.XAxis * (width * float(positions[i])) + plane.YAxis * remap_number(float(channel[i]), y_low, y_high, 0, height)
                for i in keep
            ])
            self.data_polyline.append(polyline)
        return [label for _, label in ticks], y_ticks

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

        # 2. Draw x & y axis points, 3. data points and polyline
        curve = getattr(self.param, "x_scale", "bands") != "bands"
        if curve:
            x_labels, y_ticks = self._plot_curve(plane, width, height)
        else:
            x_labels = [str(freq) for freq in self.frequencies]
            y_ticks = calculate_ticks(self.values)
            count_x = len(self.frequencies)
            # half a step between the y axis and the first band, so its point and label are clear of the axis
            step_x = width / (count_x - 0.5)
            self.x_axis_points = [plane.Origin + (x_dir * step_x * (i + 0.5)) for i in range(count_x)]

            y_values = [remap_number(value, min(y_ticks), max(y_ticks), 0, height) for value in self.values]
            self.data_points = [pt + y_dir * y for pt, y in zip(self.x_axis_points, y_values)]
            self.data_polyline = rg.Polyline()
            self.data_polyline.AddRange(self.data_points)

        count_y = len(y_ticks)
        step_y = height / (count_y - 1)
        self.y_axis_points = [plane.Origin + (y_dir * step_y * i) for i in range(count_y)]

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
        # reads upwards
        pln_y_title.XAxis = plane.YAxis
        pln_y_title.YAxis = -plane.XAxis
        pln_y_title.Origin = plane.Origin + x_dir * (width * -.1) + y_dir * (height * .5)
        self.y_axis_title = self.text_entity(
            self.param.y_unit, pln_y_title, text_height=axis_title_height, justification=self.settings.y_axis_title_justification,
            draw_forward=False,
        )

        self.x_tick_labels = [
            self.text_entity(
                str(tick),
                plane=rg.Plane(pt + y_dir * -.1, plane.XAxis, plane.YAxis),
                text_height=label_height,
                justification=self.settings.x_label_justification
            )
            for tick, pt in zip(x_labels, self.x_axis_points)
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

        # A curve has too many points to label
        self.data_values = [] if curve else [
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