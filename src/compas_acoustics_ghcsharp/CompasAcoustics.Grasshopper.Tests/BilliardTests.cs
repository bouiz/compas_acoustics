using System.Linq;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    /// <summary>Visualise Billiard's playback, the tests of the Python component with a fake clock in seconds.</summary>
    public class PlaybackTests
    {
        private double _now;
        private const double Duration = 40.0;
        private const double Longer = 80.0;

        private Playback Make() => new(() => _now);

        [Fact]
        public void WithoutPlayOrTime_ShowsTheStart()
        {
            var frame = Make().Update(null, false, null, Duration);
            Assert.Equal(0.0, frame.T);
            Assert.False(frame.Animating);
            Assert.Equal("press Play or connect time", frame.Message);
        }

        [Fact]
        public void TimeAlone_IsAFrozenFrame()
        {
            var frame = Make().Update(10.0, false, null, Duration);
            Assert.Equal(10.0, frame.T);
            Assert.False(frame.Animating);
            Assert.Equal("t = 10.0 ms", frame.Message);
        }

        [Fact]
        public void Pause_HaltsThePlaybackWhereItIs_AndPlayResumesFromThere()
        {
            var p = Make();
            _now = 0;
            p.Update(null, true, 5.0, Duration);
            _now = 1;
            Assert.True(p.Update(null, false, 5.0, Duration).Animating);
            p.Pause();
            _now = 4;
            var frame = p.Update(null, false, 5.0, Duration);
            Assert.Equal((5.0, false), (frame.T, frame.Animating));
            Assert.Equal("t = 5.0 ms", frame.Message);
            // Play again: on from the frame it held
            frame = p.Update(null, true, 5.0, Duration);
            Assert.Equal((5.0, true), (frame.T, frame.Animating));
            _now = 5;
            Assert.Equal(10.0, p.Update(null, false, 5.0, Duration).T);
            // moving the time slider after a pause scrubs
            p.Pause();
            Assert.Equal(12.0, p.Update(12.0, false, 5.0, Duration).T);
        }

        [Fact]
        public void Rewind_WhilePlaying_StartsOver_AndWhenStopped_ShowsTheStart()
        {
            var p = Make();
            _now = 0;
            p.Update(null, true, 5.0, Duration);
            _now = 1;
            Assert.Equal(5.0, p.Update(null, false, 5.0, Duration).T);
            p.Rewind();
            var frame = p.Update(null, false, 5.0, Duration);
            Assert.Equal((0.0, true), (frame.T, frame.Animating));
            _now = 2;
            Assert.Equal(5.0, p.Update(null, false, 5.0, Duration).T);
            p.Pause();
            p.Rewind();
            _now = 5;
            frame = p.Update(null, false, 5.0, Duration);
            Assert.Equal((0.0, false), (frame.T, frame.Animating));
            Assert.True(p.Active);
            Assert.False(Make().Active);
            // Play after Rewind: from the start
            Assert.Equal((0.0, true), (p.Update(null, true, 5.0, Duration).T, true));
        }

        [Fact]
        public void TheRays_DrawNothingUntilPlay_ThenAsFarAsSoundHasTravelled()
        {
            Assert.Equal(0.0, VisualiseRaysComponent.DrawnDistance(active: false, timeMs: 50, speedOfSound: 343));
            Assert.Equal(170.0, VisualiseRaysComponent.DrawnDistance(active: true, timeMs: 500, speedOfSound: 340), 9);
        }

        [Fact]
        public void TheBilliardAndTheRays_HaveOnePlayStopButtonAndRewind_AndAllThreeAPreviewToggle()
        {
            Assert.Equal(new[] { "play", "rewind" }, new VisualiseBilliardComponent().Controls.OfType<ButtonControl>().Select(b => b.Port));
            Assert.Equal(new[] { "play", "rewind" }, new VisualiseRaysComponent().Controls.OfType<ButtonControl>().Select(b => b.Port));
            var reflections = new VisualiseReflectionsComponent();
            Assert.Contains(reflections.Controls, c => c is ToggleControl { Port: "preview" });
            Assert.True(reflections.IsChoice("preview"));
            reflections.SetChoice("preview", "off");
            Assert.True(reflections.Hidden);
            reflections.Hidden = false;
            Assert.Equal("on", reflections.Choice("preview"));
            Assert.True(new VisualiseBilliardComponent().IsChoice("preview") && new VisualiseRaysComponent().IsChoice("preview"));
            // the drawing is the component's own: its curve outputs would draw every ray before Play
            var rays = new VisualiseRaysComponent();
            Assert.All(rays.Params.Output, p => Assert.False(rays.PreviewsOutput(p)));
        }

        [Fact]
        public void ThePlayButton_ReadsPauseWhilePlaying()
        {
            var p = Make();
            Assert.False(p.Running);
            _now = 0;
            p.Update(null, true, 5.0, Duration);
            _now = 1;
            Assert.True(p.Running);
            Assert.Equal("Pause", VisualiseBilliardComponent.PlayLabel(p));
            p.Pause();
            Assert.False(p.Running);
            Assert.Equal("Play", VisualiseBilliardComponent.PlayLabel(p));
            // a playback that has reached its end is not running
            p.Update(null, true, 5.0, Duration);
            _now = 100;
            p.Update(null, false, 5.0, Duration);
            Assert.False(p.Running);
        }

        [Fact]
        public void Play_AnimatesUpToTimeAndStops()
        {
            var p = Make();
            _now = 0;
            var frame = p.Update(10.0, true, 5.0, Duration);
            Assert.Equal(0.0, frame.T);
            Assert.True(frame.Animating);
            _now = 1;
            frame = p.Update(10.0, false, 5.0, Duration);
            Assert.Equal(5.0, frame.T);
            Assert.True(frame.Animating);
            _now = 3;
            frame = p.Update(10.0, false, 5.0, Duration);
            Assert.Equal(10.0, frame.T);
            Assert.False(frame.Animating);
            _now = 9;
            Assert.Equal(10.0, p.Update(10.0, false, 5.0, Duration).T);
        }

        [Fact]
        public void PlayWithoutTime_RunsToTheEndAtTheDefaultSpeed()
        {
            var p = Make();
            p.Update(null, true, null, Duration);
            _now = 5;
            Assert.Equal(Duration / 2, p.Update(null, false, null, Duration).T, 9);
            _now = 20;
            Assert.Equal(Duration, p.Update(null, false, null, Duration).T, 9);
        }

        [Fact]
        public void PressingPlayWhilePlaying_Restarts()
        {
            var p = Make();
            p.Update(10.0, true, 5.0, Duration);
            _now = 1.5;
            Assert.Equal(7.5, p.Update(10.0, false, 5.0, Duration).T);
            Assert.Equal(0.0, p.Update(10.0, true, 5.0, Duration).T);
        }

        [Fact]
        public void ScrubbingTimeAfterAFinishedPlay_Scrubs()
        {
            var p = Make();
            p.Update(10.0, true, 5.0, Duration);
            _now = 100;
            Assert.Equal(10.0, p.Update(10.0, false, 5.0, Duration).T);
            var frame = p.Update(3.0, false, 5.0, Duration);
            Assert.Equal(3.0, frame.T);
            Assert.False(frame.Animating);
            _now = 200;
            Assert.Equal(3.0, p.Update(3.0, false, 5.0, Duration).T);
        }

        [Fact]
        public void ChangingTimeDuringPlayback_StopsIt()
        {
            var p = Make();
            p.Update(10.0, true, 5.0, Duration);
            _now = 1;
            var frame = p.Update(4.0, false, 5.0, Duration);
            Assert.Equal(4.0, frame.T);
            Assert.False(frame.Animating);
        }

        [Fact]
        public void TheHeldEndFrame_StaysWhileTimeStaysEmpty()
        {
            var p = Make();
            p.Update(null, true, null, Duration);
            _now = 50;
            Assert.Equal(Duration, p.Update(null, false, null, Duration).T, 9);
            _now = 60;
            var frame = p.Update(null, false, null, Duration);
            Assert.Equal(Duration, frame.T, 9);
            Assert.False(frame.Animating);
        }

        [Fact]
        public void Times_AreClampedToTheDuration()
        {
            var p = Make();
            var frame = p.Update(-5.0, true, 5.0, Duration);
            Assert.Equal(0.0, frame.T);
            Assert.False(frame.Animating);
            _now = 3;
            Assert.Equal(0.0, p.Update(-5.0, false, 5.0, Duration).T);
            Assert.Equal(Duration, p.Update(1e9, false, 5.0, Duration).T);
            frame = p.Update(-5.0, false, 5.0, Duration);
            Assert.Equal(0.0, frame.T);
            Assert.Equal("t = 0.0 ms", frame.Message);
        }

        [Fact]
        public void AHeldPlayToggle_PlaysOnceAndHolds()
        {
            var p = Make();
            Assert.Equal(0.0, p.Update(10.0, true, 5.0, Duration).T);
            _now = 1;
            Assert.Equal(5.0, p.Update(10.0, true, 5.0, Duration).T);
            _now = 3;
            Assert.Equal(10.0, p.Update(10.0, true, 5.0, Duration).T);
            _now = 4;
            var frame = p.Update(10.0, true, 5.0, Duration);
            Assert.Equal(10.0, frame.T);
            Assert.False(frame.Animating);
        }

        [Fact]
        public void PlayTurnedOffAndOnAgain_Restarts()
        {
            var p = Make();
            p.Update(10.0, true, 5.0, Duration);
            _now = 1;
            Assert.Equal(5.0, p.Update(10.0, false, 5.0, Duration).T);
            Assert.Equal(0.0, p.Update(10.0, true, 5.0, Duration).T);
            _now = 1.5;
            Assert.Equal(2.5, p.Update(10.0, true, 5.0, Duration).T);
        }

        [Fact]
        public void ScrubbingWithAHeldPlayToggle_Scrubs()
        {
            var p = Make();
            p.Update(10.0, true, 5.0, Duration);
            _now = 1;
            Assert.Equal(4.0, p.Update(4.0, true, 5.0, Duration).T);
            _now = 2;
            Assert.Equal(4.0, p.Update(4.0, true, 5.0, Duration).T);
        }

        [Fact]
        public void PlayingToTheEnd_FollowsALongerRayField()
        {
            var p = Make();
            p.Update(null, true, 5.0, Duration);
            _now = 1000;
            Assert.Equal(Duration, p.Update(null, false, 5.0, Duration).T, 9);
            Assert.Equal(Longer, p.Update(null, false, 5.0, Longer).T, 9);
        }

        [Fact]
        public void TheDefaultSpeed_FollowsANewRayField()
        {
            var p = Make();
            p.Update(null, true, null, Duration);
            _now = 2;
            Assert.Equal(Duration / 5, p.Update(null, false, null, Longer).T, 9);
            _now = 3;
            Assert.Equal(Duration / 5 + Longer / 10, p.Update(null, false, null, Longer).T, 9);
        }

        [Fact]
        public void ANewSpeedWhilePlaying_RunsOnFromTheCurrentFrame()
        {
            var p = Make();
            p.Update(null, true, 1.0, Duration);
            _now = 2;
            Assert.Equal(2.0, p.Update(null, false, 1.0, Duration).T, 9);
            Assert.Equal(2.0, p.Update(null, false, 4.0, Duration).T, 9);
            _now = 3;
            Assert.Equal(6.0, p.Update(null, false, 4.0, Duration).T, 9);
        }

        [Fact]
        public void ANonPositiveSpeed_UsesTheDefault()
        {
            var p = Make();
            var frame = p.Update(null, true, 0.0, Duration);
            Assert.Equal("speed must be positive; using the default.", frame.Remark);
            _now = 5;
            Assert.Equal(Duration / 2, p.Update(null, false, -1.0, Duration).T, 9);
        }
    }

    public class ParticlesTests
    {
        // a ray (0,0,1) -> (3.43,0,1) -> (3.43,3.43,1): 10 ms per segment at 343 m/s
        private static readonly RayGeometry.Ray Ray = new(new[] { new Point3d(0, 0, 1), new Point3d(3.43, 0, 1), new Point3d(3.43, 3.43, 1) }, 0.0, 0, false);
        private static readonly RayGeometry.Ray Late = new(new[] { new Point3d(0, 0, 0), new Point3d(1, 0, 0) }, 3.43, 0, false);

        [Fact]
        public void AParticle_MovesAlongItsRay()
        {
            var (point, order) = Particles.At(new[] { Ray }, 5.0, 343.0).Single();
            Assert.Equal(1.715, point.X, 9);
            Assert.Equal(0, order);
            (point, order) = Particles.At(new[] { Ray }, 15.0, 343.0).Single();
            Assert.Equal(3.43, point.X, 9);
            Assert.Equal(1.715, point.Y, 9);
            Assert.Equal(1, order);
        }

        [Fact]
        public void AParticleOnAWall_IsNotYetReflected()
        {
            var (point, order) = Particles.At(new[] { Ray }, 10.0, 343.0).Single();
            Assert.Equal(3.43, point.X, 9);
            Assert.Equal(0, order);
        }

        [Fact]
        public void AFinishedRay_GivesNone()
        {
            var shorter = new RayGeometry.Ray(new[] { new Point3d(0, 0, 0), new Point3d(3.43, 0, 0) }, 0.0, 0, false);
            var particles = Particles.At(new[] { Ray, shorter }, 15.0, 343.0);
            Assert.Single(particles);
        }

        [Fact]
        public void BeforeEmergence_GivesNone_AndAtItTheFirstPoint()
        {
            Assert.Empty(Particles.At(new[] { Late }, 5.0, 343.0));
            var (point, _) = Particles.At(new[] { Late }, 10.0, 343.0).Single();
            Assert.Equal(0.0, point.X, 9);
        }

        [Fact]
        public void Duration_IsTheLongestTravelTime()
        {
            Assert.Equal(20.0, Particles.DurationMs(new[] { Ray, Late }, 343.0), 9);
            Assert.Equal(0.0, Particles.DurationMs(new RayGeometry.Ray[0], 343.0));
        }
    }
}
