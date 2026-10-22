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
            pipeline.DrawSphere(new Sphere(source.Centre, PreviewGeometry.SourceRadius), color, thickness);
            pipeline.DrawArrow(source.Arrow, color);
            pipeline.DrawPolyline(source.Lobe, color, thickness);
            pipeline.Draw2dText(source.Label, color, source.Centre + new Vector3d(0, 0, PreviewGeometry.SourceRadius), false, LabelHeight);
        }

        public static void Receiver(DisplayPipeline pipeline, ReceiverPreview receiver, Color color, int thickness)
        {
            // a cube, where a source is a sphere
            pipeline.DrawLines(receiver.Cube, color, thickness);
            pipeline.DrawCircle(new Circle(new Plane(receiver.LeftEar, receiver.EarAxis), PreviewGeometry.EarRadius), color, thickness);
            pipeline.DrawCircle(new Circle(new Plane(receiver.RightEar, receiver.EarAxis), PreviewGeometry.EarRadius), color, thickness);
            pipeline.DrawArrow(receiver.Facing, color);
            pipeline.DrawLine(receiver.Up, color, thickness);
            pipeline.Draw2dText(receiver.Label, color, receiver.Centre + new Vector3d(0, 0, PreviewGeometry.HeadRadius), false, LabelHeight);
        }

        /// <summary>The shaded objects as one mesh with vertex colours, and the unshaded ones (no material) as another.</summary>
        public static (Mesh? shaded, Mesh? plain) Meshes(IEnumerable<ObjectPreview> objects)
        {
            Mesh? shaded = null, plain = null;
            foreach (var o in objects.Where(o => o.Faces.Length > 0))
            {
                var mesh = new Mesh();
                mesh.Vertices.AddVertices(o.Vertices);
                foreach (var face in o.Faces)
                {
                    if (face.Length == 3)
                        mesh.Faces.AddFace(face[0], face[1], face[2]);
                    else if (face.Length == 4)
                        mesh.Faces.AddFace(face[0], face[1], face[2], face[3]);
                    else
                        for (var i = 1; i + 1 < face.Length; i++)
                            mesh.Faces.AddFace(face[0], face[i], face[i + 1]);
                }
                if (o.Shade is Color shade)
                {
                    mesh.VertexColors.CreateMonotoneMesh(shade);
                    (shaded ??= new Mesh()).Append(mesh);
                }
                else
                    (plain ??= new Mesh()).Append(mesh);
            }
            shaded?.Normals.ComputeNormals();
            plain?.Normals.ComputeNormals();
            return (shaded, plain);
        }

        public static void SceneWires(DisplayPipeline pipeline, ScenePreview scene, Color color, int thickness)
        {
            ObjectWires(pipeline, scene.Objects, color, thickness);
            foreach (var s in scene.Sources)
                Source(pipeline, s, color, thickness);
            foreach (var r in scene.Receivers)
                Receiver(pipeline, r, color, thickness);
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
