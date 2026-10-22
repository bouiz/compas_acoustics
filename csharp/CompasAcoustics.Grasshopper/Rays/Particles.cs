using System;
using System.Collections.Generic;
using System.Linq;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Where the sound of each specular ray is at a time: RayField.particles, for the billiard.</summary>
    public static class Particles
    {
        private const double Tolerance = 1e-9;

        public static double DurationMs(IEnumerable<RayGeometry.Ray> rays, double speedOfSound) =>
            rays.Select(r => r.TravelTimeMs(speedOfSound)).DefaultIfEmpty(0.0).Max();

        /// <summary>
        /// The position of every ray at ``timeMs`` (clamped to the rays' duration), with the number of reflections it has
        /// passed. A ray that has ended or not emerged yet gives none; a particle on a wall is reflected only once it moves on.
        /// </summary>
        public static List<(Point3d point, int order)> At(IReadOnlyList<RayGeometry.Ray> rays, double timeMs, double speedOfSound)
        {
            timeMs = Math.Min(Math.Max(timeMs, 0.0), DurationMs(rays, speedOfSound));
            var distance = speedOfSound * timeMs / 1000.0;
            var particles = new List<(Point3d, int)>();
            foreach (var ray in rays)
            {
                if (ray.Points.Length == 0 || ray.TravelTimeMs(speedOfSound) < timeMs)
                    continue;
                var cumulative = ray.CumulativeDistances;
                if (distance < cumulative[0])
                    continue;
                particles.Add((PointAt(ray.Points, cumulative, distance), cumulative.Skip(1).Count(reached => reached < distance - Tolerance)));
            }
            return particles;
        }

        private static Point3d PointAt(Point3d[] points, double[] cumulative, double distance)
        {
            if (distance <= cumulative[0])
                return points[0];
            for (var i = 1; i < points.Length; i++)
            {
                if (cumulative[i] <= distance)
                    continue;
                var segment = cumulative[i] - cumulative[i - 1];
                var t = segment > 0 ? (distance - cumulative[i - 1]) / segment : 0.0;
                var (a, b) = (points[i - 1], points[i]);
                return new Point3d(a.X + t * (b.X - a.X), a.Y + t * (b.Y - a.Y), a.Z + t * (b.Z - a.Z));
            }
            return points[^1];
        }
    }

    /// <summary>One frame of the billiard: the time shown, whether to ask for the next frame, the message, and a remark.</summary>
    public sealed record Frame(double T, bool Animating, string Message, string? Remark = null);

    /// <summary>
    /// The billiard's playback: Play (a button press, or a Boolean Toggle turned on) runs from 0 to ``time`` (default: the end)
    /// at ``speed`` ms of sound per second (default: the whole field in 10 s) and holds the last frame; moving ``time``
    /// stops playback and scrubs; a new speed runs on from the frame shown.
    /// </summary>
    public sealed class Playback
    {
        private sealed class State
        {
            public double Start;
            public double Offset;
            public double? Time;
            public double Speed;
        }

        private readonly Func<double> _now;
        private bool _lastPlay;
        private State? _state;
        // a stopped playback holds its frame until Play, or until the time input moves
        private double? _held;
        private double? _heldTime;
        private double _lastEnd;
        private double? _lastTime;

        /// <summary>Whether a playback runs, ended, or holds a frame (else the time input, or nothing, decides).</summary>
        public bool Active => _state != null || _held != null;

        /// <summary>Back to the start: a running playback starts over, a stopped or ended one shows its first frame.</summary>
        public void Rewind()
        {
            var now = _now();
            if (_state != null && _state.Offset + (now - _state.Start) * _state.Speed < _lastEnd)
            {
                _state.Start = now;
                _state.Offset = 0.0;
                return;
            }
            _heldTime = _state?.Time ?? _lastTime;
            _state = null;
            _held = 0.0;
        }

        /// <param name="now">The clock, in seconds.</param>
        public Playback(Func<double> now) => _now = now;

        /// <summary>Stop the animation where it is: the frame stays until Play starts over or the time input moves.</summary>
        public void Stop()
        {
            if (_state == null)
                return;
            _held = Math.Min(Math.Max(_state.Offset + (_now() - _state.Start) * _state.Speed, 0.0), _lastEnd);
            _heldTime = _state.Time;
            _state = null;
        }

        public Frame Update(double? time, bool play, double? speed, double durationMs)
        {
            string? remark = null;
            if (speed is double s && s <= 0)
            {
                remark = "speed must be positive; using the default.";
                speed = null;
            }
            // resolved on every solve: a retraced field changes the default end and speed
            var rate = speed ?? durationMs / 10.0;
            var end = Math.Min(Math.Max(time ?? durationMs, 0.0), durationMs);

            // a Boolean Toggle stays True: only its change to True starts playback, as a button's press does
            var pressed = play && !_lastPlay;
            _lastPlay = play;
            if (pressed)
            {
                _state = new State { Start = _now(), Offset = 0.0, Time = time, Speed = rate };
                _held = null;
            }
            if (_held != null && !pressed && time != _heldTime)
                _held = null;
            _lastEnd = end;
            _lastTime = time;
            if (_state != null && !pressed && time != _state.Time)
                // moving the slider during or after playback stops it and scrubs
                _state = null;

            double t;
            var animating = false;
            if (_state != null)
            {
                var now = _now();
                if (rate != _state.Speed)
                {
                    // a new speed runs on from the frame shown, not from 0 at the new rate
                    _state.Offset = Math.Min(_state.Offset + (now - _state.Start) * _state.Speed, end);
                    _state.Start = now;
                    _state.Speed = rate;
                }
                t = Math.Min(Math.Max(_state.Offset + (now - _state.Start) * rate, 0.0), end);
                animating = t < end;
            }
            else if (_held is double held)
                t = Math.Min(held, durationMs);
            else if (time is double given)
                t = Math.Min(Math.Max(given, 0.0), durationMs);
            else
                t = 0.0;
            var message = _state == null && _held == null && time == null ? "press Play or connect time" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "t = {0:0.0} ms", t);
            return new Frame(t, animating, message, remark);
        }
    }
}
