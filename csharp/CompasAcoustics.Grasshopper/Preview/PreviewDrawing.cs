using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Rhino.Display;
using Rhino.Geometry;

namespace CompasAcoustics.Grasshopper
{
    /// <summary>Draws the previews in Grasshopper's preview colour (``color`` is the selected or unselected one).</summary>
    internal static class PreviewDrawing
    {
        public static readonly Color ObstacleEdge = Color.FromArgb(255, 127, 14);
        private const int LabelHeight = 11;

        public static void Source(DisplayPipeline pipeline, SourcePreview source, Color color, int thickness)
        {
            // a loudspeaker: a cube with its driver on the face toward the source's direction
            pipeline.DrawLines(source.Cube, color, thickness);
            pipeline.DrawCircle(new Circle(new Plane(source.Driver, source.Facing), source.DriverRadius), color, thickness);
            pipeline.DrawArrow(source.Arrow, color);
            pipeline.DrawPolyline(source.Lobe, color, thickness);
            pipeline.Draw2dText(source.Label, color, source.Centre + new Vector3d(0, 0, PreviewGeometry.SourceRadius), false, LabelHeight);
        }

        public static void Receiver(DisplayPipeline pipeline, ReceiverPreview receiver, Color color, int thickness)
        {
            // a head, where a source is a loudspeaker
            pipeline.DrawSphere(new Sphere(receiver.Centre, PreviewGeometry.HeadRadius), color, thickness);
            pipeline.DrawCircle(new Circle(new Plane(receiver.LeftEar, receiver.EarAxis), PreviewGeometry.EarRadius), color, thickness);
            pipeline.DrawCircle(new Circle(new Plane(receiver.RightEar, receiver.EarAxis), PreviewGeometry.EarRadius), color, thickness);
            pipeline.DrawArrow(receiver.Facing, color);
            pipeline.DrawLine(receiver.Up, color, thickness);
            pipeline.Draw2dText(receiver.Label, color, receiver.Centre + new Vector3d(0, 0, PreviewGeometry.HeadRadius), false, LabelHeight);
        }

        public static void SceneWires(DisplayPipeline pipeline, ScenePreview scene, Color color, int thickness)
        {
            // the sources and receivers are drawn by their own components, under their own preview toggles
            ObjectWires(pipeline, scene.Objects, color, thickness);
            pipeline.Draw2dText(scene.Label, color, scene.LabelAt, true, LabelHeight);
        }

        /// <summary>The objects' edges: obstacles in orange, the others in the preview colour.</summary>
        public static void ObjectWires(DisplayPipeline pipeline, IEnumerable<ObjectPreview> objects, Color color, int thickness)
        {
            foreach (var o in objects)
            {
                var edge = o.Obstacle ? ObstacleEdge : color;
                if (o.Polyline != null)
                    pipeline.DrawPolyline(o.Polyline, edge, thickness);
                foreach (var face in o.Faces)
                    pipeline.DrawPolyline(face.Append(face[0]).Select(i => o.Vertices[i]), edge, thickness);
            }
        }

        public static void Graph(DisplayPipeline pipeline, IEnumerable<PairPreview> pairs, Color color, int thickness)
        {
            foreach (var pair in pairs)
            {
                pipeline.DrawLine(pair.Line, color, thickness);
                pipeline.Draw2dText(pair.Label, color, pair.Line.PointAt(0.5), true, LabelHeight);
            }
        }

        public static BoundingBox Box(Point3d centre, double radius) =>
            new(centre - new Vector3d(radius, radius, radius), centre + new Vector3d(radius, radius, radius));
    }
}
