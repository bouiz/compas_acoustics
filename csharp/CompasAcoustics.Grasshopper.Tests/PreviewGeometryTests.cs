using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CompasAcoustics.Pb;
using CompasPb.Data;
using Rhino.Geometry;
using Xunit;

namespace CompasAcoustics.Grasshopper.Tests
{
    public class PreviewGeometryTests
    {
        private static PointData P(double x, double y, double z) => new() { X = x, Y = y, Z = z };
        private static VectorData V(double x, double y, double z) => new() { X = x, Y = y, Z = z };

        private static SourceData Source(string directivity, VectorData direction) =>
            new() { Name = "src", Point = P(1, 2, 3), Direction = direction, Directivity = directivity };

        [Fact]
        public void Source_OmniLobeIsACircle()
        {
            var preview = PreviewGeometry.Source(Source("omnidirectional", V(1, 0, 0)));
            var centre = new Point3d(1, 2, 3);
            Assert.Equal(centre, preview.Centre);
            Assert.Equal(37, preview.Lobe.Count);
            Assert.Equal(preview.Lobe[0], preview.Lobe[36]);
            Assert.All(preview.Lobe, p => Assert.Equal(0.3, p.DistanceTo(centre), 9));
            Assert.Equal(new Point3d(1.5, 2, 3), preview.Arrow.To);
            Assert.Equal("src", preview.Label);
        }

        [Fact]
        public void Source_CardioidLobeIsZeroBehind_InThePlaneOfTheDirectionAndZ()
        {
            var preview = PreviewGeometry.Source(Source("cardioid", V(0, 2, 0)));
            var centre = new Point3d(1, 2, 3);
            Assert.Equal(0.3, preview.Lobe[0].DistanceTo(centre), 9);
            Assert.Equal(0, preview.Lobe[0].DistanceTo(new Point3d(1, 2.3, 3)), 9);
            Assert.Equal(0, preview.Lobe[18].DistanceTo(centre), 9);
            // a quarter turn from the direction, towards z: half the radius
            Assert.Equal(0, preview.Lobe[9].DistanceTo(new Point3d(1, 2, 3.15)), 9);
        }

        [Fact]
        public void Source_FacingUp_StillHasALobe()
        {
            var preview = PreviewGeometry.Source(Source("supercardioid", V(0, 0, 1)));
            Assert.All(preview.Lobe, p => Assert.True(p.IsValid));
            Assert.Equal(0, preview.Lobe[0].DistanceTo(new Point3d(1, 2, 3.3)), 9);
        }

        [Fact]
        public void Receiver_EarsAreOnTheYAxis()
        {
            var preview = PreviewGeometry.Receiver(new ReceiverData { Name = "rec", Point = P(0, 0, 1.5), Direction = V(0, 1, 0) });
            Assert.Equal(new Point3d(-0.1, 0, 1.5), preview.LeftEar);
            Assert.Equal(new Point3d(0.1, 0, 1.5), preview.RightEar);
            Assert.Equal(new Point3d(0, 0.4, 1.5), preview.Facing.To);
            Assert.Equal(0, preview.Up.To.DistanceTo(new Point3d(0, 0, 1.65)), 9);
            Assert.Equal("rec", preview.Label);
        }

        [Fact]
        public void ASource_IsALoudspeaker_ACubeTurnedWithItAndADriverOnItsFront()
        {
            var preview = PreviewGeometry.Source(Source("cardioid", V(1, 1, 0)));
            var centre = new Point3d(1, 2, 3);
            var h = PreviewGeometry.SourceRadius;
            Assert.Equal(12, preview.Cube.Length);
            Assert.All(preview.Cube, edge => Assert.Equal(2 * h, edge.Length, 9));
            var facing = new Vector3d(1, 1, 0) / Math.Sqrt(2);
            Assert.Contains(preview.Cube, edge => Math.Abs(Math.Abs((edge.To - edge.From) * facing) - edge.Length) < 1e-9);
            // the driver: a circle on the face toward the source's direction
            Assert.Equal(0, preview.Driver.DistanceTo(centre + h * facing), 9);
            Assert.True(preview.DriverRadius > 0 && preview.DriverRadius < h);
        }

        internal static SceneData Room(bool obstacle = false)
        {
            var material = new AcousticMaterialData { Name = "wall" };
            material.AbsorptionCoefficients.Add(new[] { 0.5, 0.5 });
            var scene = new SceneData { Name = "room" };
            foreach (var wall in WorkerIntegrationTests.ShoeboxWalls(7, 5, 3))
                scene.AcousticObjects.Add(new AcousticObjectData { Mesh = wall, Material = material, Role = "boundary" });
            if (obstacle)
                scene.AcousticObjects.Add(new AcousticObjectData
                {
                    Mesh = WorkerIntegrationTests.Quad(new double[] { 1, 1, 0 }, new double[] { 2, 1, 0 }, new double[] { 2, 1, 2 }, new double[] { 1, 1, 2 }),
                    Role = "obstacle",
                });
            scene.Sources.Add(new SourceData { Name = "src", Point = P(1.5, 1.5, 1.6), Direction = V(1, 0, 0), Directivity = "omnidirectional" });
            scene.Receivers.Add(new ReceiverData { Name = "rec", Point = P(5, 3, 1.5), Direction = V(-1, 0, 0) });
            return scene;
        }

        [Fact]
        public void Scene_LabelHasVolumeAndArea()
        {
            var preview = PreviewGeometry.Scene(Room());
            Assert.Equal(105, preview.Volume, 9);
            Assert.Equal(142, preview.Area, 9);
            Assert.Equal("room, V 105 m3, S 142 m2", preview.Label);
            Assert.Equal(new Point3d(3.5, 2.5, 1.5), preview.LabelAt);
            Assert.Equal(6, preview.Objects.Count);
        }

        [Fact]
        public void Scene_ObstaclesAreFlagged_AndLeftOutOfVolumeAndArea()
        {
            var preview = PreviewGeometry.Scene(Room(obstacle: true));
            Assert.Equal(105, preview.Volume, 9);
            Assert.Equal(142, preview.Area, 9);
            var obstacle = preview.Objects.Last();
            Assert.True(obstacle.Obstacle);
            Assert.Equal(new[] { 4 }, obstacle.Faces.Select(f => f.Length));
        }

        [Fact]
        public void ModelData_DrawOnlyFromTheirOwnParameters_NotFromMergeOrData()
        {
            // Grasshopper's generic parameters (Merge, Data) draw any data that is IGH_PreviewData
            Assert.False(typeof(global::Grasshopper.Kernel.IGH_PreviewData).IsAssignableFrom(typeof(AcousticObjectGoo)));
            Assert.False(typeof(global::Grasshopper.Kernel.IGH_PreviewData).IsAssignableFrom(typeof(SceneGoo)));
            Assert.True(typeof(global::Grasshopper.Kernel.IGH_PreviewObject).IsAssignableFrom(typeof(Param_AcousticObject)));
            Assert.True(typeof(global::Grasshopper.Kernel.IGH_PreviewObject).IsAssignableFrom(typeof(Param_Scene)));
            // nor sources and receivers: a Merge of sources drew them past the Source's preview toggle
            Assert.False(typeof(global::Grasshopper.Kernel.IGH_PreviewData).IsAssignableFrom(typeof(SourceGoo)));
            Assert.False(typeof(global::Grasshopper.Kernel.IGH_PreviewData).IsAssignableFrom(typeof(ReceiverGoo)));
            Assert.True(typeof(global::Grasshopper.Kernel.IGH_PreviewObject).IsAssignableFrom(typeof(Param_Source)));
            Assert.True(typeof(global::Grasshopper.Kernel.IGH_PreviewObject).IsAssignableFrom(typeof(Param_Receiver)));
        }

        [Fact]
        public void ObjectsAndScenes_PreviewOnlyTheirEdges_NoSurfacesOverTheRhinoGeometry()
        {
            // a mesh on the very surface it was made from covers it and fights with it in depth
            Assert.Null(typeof(IOwnPreview).GetMethod("DrawMeshes"));
            Assert.Null(typeof(AcousticObjectGoo).GetMethod("DrawMeshes"));
            Assert.Null(typeof(SceneGoo).GetMethod("DrawMeshes"));
        }

        [Fact]
        public void SourcesAndReceivers_AreDrawnOnlyByTheirOwnComponents()
        {
            // the Source and Receiver previews switch them off; a scene or a graph drawing them again would not
            Assert.Null(typeof(ScenePreview).GetProperty("Sources"));
            Assert.Null(typeof(ScenePreview).GetProperty("Receivers"));
            var graph = new DeconstructIrGraphComponent();
            Assert.Equal(new[] { "edges" }, graph.Params.Output.Where(graph.PreviewsOutput).Select(p => p.Name));
            var source = new SourceComponent();
            Assert.All(source.Params.Output, p => Assert.True(source.PreviewsOutput(p)));
        }

        [Fact]
        public void AnAcousticObject_PreviewsItsSurface()
        {
            var obj = Room(obstacle: true).AcousticObjects[0];
            var preview = new AcousticObjectGoo(obj).Preview!;
            Assert.Equal(4, preview.Vertices.Length);
            Assert.False(preview.Obstacle);
            Assert.True(new AcousticObjectGoo(Room(obstacle: true).AcousticObjects.Last()).Preview!.Obstacle);
        }

        [Fact]
        public void ScenePreview_BuildsOnce()
        {
            var goo = new SceneGoo(Room());
            Assert.Same(goo.Preview, goo.Preview);
            Assert.NotSame(goo.Preview, new SceneGoo(Room()).Preview);
            Assert.True(goo.ClippingBox.Contains(new Point3d(3.5, 2.5, 1.5)));
        }

        [Fact]
        public void AComponent_PreviewsOnlyItsOutputs_NotTheGeometryWiredIn()
        {
            // the geometry, point and plane inputs would draw what is already in the viewport again; the outputs draw
            Assert.True(new AcousticObjectComponent().IsPreviewCapable);
            Assert.True(new SourceComponent().IsPreviewCapable);
            Assert.True(new RoomSceneComponent().IsPreviewCapable);
            // the surfaces are in Rhino already: their shaded preview is there to turn on, off on a new component
            Assert.True(new AcousticObjectComponent().PreviewOffByDefault);
            Assert.True(new RoomSceneComponent().PreviewOffByDefault);
            Assert.False(new SourceComponent().PreviewOffByDefault);
            // the Visualise components draw their rays themselves
            Assert.True(new VisualiseReflectionsComponent().IsPreviewCapable);
        }

        [Fact]
        public void IrGraphPreview_DrawsALinePerPairWithItsNames()
        {
            var scene = new SceneGoo(Room());
            var ir = new ImpulseResponseData { Source = scene.Value.Sources[0], Receiver = scene.Value.Receivers[0], SamplingRate = 44100 };
            var graph = new IrGraphGoo(new IrGraphGoo.Content("pyroomacoustics", scene.Sources, scene.Receivers, new[] { new ImpulseResponseGoo(ir) }));
            var pair = Assert.Single(PreviewGeometry.Graph(graph.Value));
            Assert.Equal(new Line(new Point3d(1.5, 1.5, 1.6), new Point3d(5, 3, 1.5)), pair.Line);
            Assert.Equal("src -> rec", pair.Label);
        }
    }
}
