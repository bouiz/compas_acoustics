namespace X
{
    public sealed class FixtureAComponent : ComponentBase
    {
        public FixtureAComponent() : base("CA_Fixture_A") { }
        protected override string SpecKey => "CA_Fixture_A";
        public override Guid ComponentGuid => new Guid("410e7b44-f239-5e1e-ba74-5884e208b73e");
    }

    public sealed class Plot2DComponent : ComponentBase
    {
        public Plot2DComponent() : base("CA_Plot_2D") { }
        protected override string SpecKey => "CA_Plot_2D";
        public override Guid ComponentGuid => new Guid("DDCA21B5-bcc8-5806-85f8-fc0a45ced2e5");
    }
}
