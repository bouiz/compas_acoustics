using System;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>One frame of a playback: the time shown, whether to ask for the next frame, the message, and a remark.</summary>
    public sealed record Frame(double T, bool Animating, string Message, string? Remark = null);

    /// <summary>
    /// The playback of Visualise Rays and Visualise Billiard: Play (a button press, or a Boolean Toggle turned on) runs from 0
    /// to <c>time</c> (default: the end) at <c>speed</c> ms of sound per second (default: the whole field in 10 s) and holds
    /// the last frame; moving <c>time</c> stops playback and scrubs; a new speed runs on from the frame shown.
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
        // a paused playback holds its frame until Play, or until the time input moves
        private double? _held;
        private double? _heldTime;
        private double _lastEnd;
        private double? _lastTime;

        /// <summary>Whether the playback is moving now (not stopped, not at its end).</summary>
        public bool Running => _state != null && _state.Offset + (_now() - _state.Start) * _state.Speed < _lastEnd;

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

        /// <summary>Pause the animation where it is: the frame stays until Play goes on from it, Rewind, or the time input moves.</summary>
        public void Pause()
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
                // on from a paused frame (the start after Rewind); from the start after the end or a scrub
                var from = _held is double h && h < end && time == _heldTime ? h : 0.0;
                _state = new State { Start = _now(), Offset = from, Time = time, Speed = rate };
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
