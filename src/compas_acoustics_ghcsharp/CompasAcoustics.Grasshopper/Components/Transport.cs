using System;

namespace CompasAcoustics.Grasshopper
{
    public enum TransportState { Stopped, Playing, Paused }

    /// <summary>What the Audio Player asks the worker to do.</summary>
    public abstract record TransportCommand;
    public sealed record PlayCommand(double Start) : TransportCommand;
    public sealed record PauseCommand : TransportCommand;
    public sealed record ResumeCommand : TransportCommand;
    public sealed record StopCommand : TransportCommand;

    /// <summary>
    /// The Audio Player's transport: its own clock while playing, corrected from the worker's playback status. The methods
    /// return the command to send to the worker, or null when nothing needs sending.
    /// </summary>
    public sealed class Transport
    {
        private readonly Func<double> _now;
        private double _position;
        private double _anchor;

        public Transport(Func<double> now) => _now = now;

        public double Duration { get; set; }

        public TransportState State { get; private set; } = TransportState.Stopped;

        public double Position => State == TransportState.Playing ? Clamp(_position + (_now() - _anchor)) : _position;

        private double Clamp(double t) => Math.Min(Math.Max(t, 0), Duration);

        private void Anchor(double position)
        {
            _position = Clamp(position);
            _anchor = _now();
        }

        public TransportCommand? Play()
        {
            switch (State)
            {
                case TransportState.Playing:
                    return null;
                case TransportState.Paused:
                    Anchor(_position);
                    State = TransportState.Playing;
                    return new ResumeCommand();
                default:
                    // a play at the end starts over
                    var start = _position >= Duration ? 0 : _position;
                    Anchor(start);
                    State = TransportState.Playing;
                    return new PlayCommand(_position);
            }
        }

        public TransportCommand? Pause()
        {
            if (State != TransportState.Playing)
                return null;
            _position = Position;
            State = TransportState.Paused;
            return new PauseCommand();
        }

        public TransportCommand? Stop()
        {
            var wasActive = State != TransportState.Stopped;
            _position = 0;
            State = TransportState.Stopped;
            return wasActive ? new StopCommand() : null;
        }

        public TransportCommand? Seek(double seconds)
        {
            if (State == TransportState.Playing)
            {
                Anchor(seconds);
                return new PlayCommand(_position);
            }
            // a paused stream would resume from where it paused: the next play starts afresh from here
            _position = Clamp(seconds);
            State = TransportState.Stopped;
            return null;
        }

        /// <summary>The worker's view of the playback: it ended (or another sound took over), or where it is.</summary>
        public void Sync(bool playing, double position)
        {
            if (State != TransportState.Playing)
                return;
            if (!playing)
            {
                _position = Duration;
                State = TransportState.Stopped;
                return;
            }
            Anchor(position);
        }
    }
}
