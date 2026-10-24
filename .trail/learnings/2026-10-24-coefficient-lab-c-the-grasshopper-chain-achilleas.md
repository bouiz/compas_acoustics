---
title: "Coefficient lab C: the Grasshopper chain (four components, worker functions, run rules, orientation)"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 109, tokens: 2331493 }
touches:
  - CHANGELOG.md
  - csharp/CompasAcoustics.Grasshopper.Tests/LabComponentTests.cs
  - csharp/CompasAcoustics.Grasshopper.Tests/WorkerIntegrationTests.cs
  - csharp/CompasAcoustics.Grasshopper/Components/LabComponents.cs
  - csharp/CompasAcoustics.Grasshopper/Components/LabRequests.cs
  - csharp/CompasAcoustics.Grasshopper/Components/LabResults.cs
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Lab_Material/metadata.json
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Polar_Preview/metadata.json
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Scattering_Lab/metadata.json
  - csharp/CompasAcoustics.Grasshopper/Components/Specs/CA_Scattering_Sample/metadata.json
---

## Decided
- Chain: Scattering Sample -> Scattering Lab -> Polar Preview / Lab Material (subcategory "08 Lab"); messages ScatteringSampleData, ScatteringResultData, ArrayData; worker functions make_scattering_sample, run_scattering_lab, polar_balloon, lab_material; the worker installs compas_acoustics[lab] (an old worker fails the schema check with "run the installer again"). Component GUIDs are uuid5(NAMESPACE_URL, "compas_acoustics/grasshopper/<key>").
- Run rule (Scattering Lab): `RunInBackground(..., start: geometric || Pressed("run"), restart: Pressed("run"))`; kirchhoff/bem never start from an input change. The run key is `LabRequests.RunKey` = the call with the sample's guid cleared (every make_scattering_sample call makes a fresh guid, which would cancel a running job on any recompute). With a stale wave result the Lab warns "inputs differ from the last run".
- Orientation: the lab assumes the object lies on World XY facing +z. Scattering Sample has an optional `plane` input (origin on the mounting surface, Z out into the room), sent as FrameData in the kwarg `frame`; the WORKER transforms mesh, reference and disc tile frame -> World XY. A 1.2x0.8x0.3 m box on a wall without the plane gave s 0.84 instead of 0.37; with it, identical to the floor within 1e-3. A flat footprint is a ValueError, not a ZeroDivisionError.
## Tried and ruled out
- A PolarResponseData message per balloon (the worker's polar_balloon returns the mesh instead); LabGridData/ScatteringSettingsData (settings travel as kwargs like simulate's).
- Letting Lab Material take absorption only from B's AbsorptionResult (B is not built): numbers or a material for now.
## Watch out
- C# tests run outside Rhino: no Rhino Mesh/Brep/Plane in tests (native constructors); keep Rhino-only code to a few lines (MeshOf, the plane read, drawing) and test everything else as static helpers (LabRequests, LabResults). Task 10 (a lamcp check in Grasshopper) has NOT been run: MeshOf for Brep/Box/Extrusion/SubD, dropdowns, SetValueOptions, balloon drawing, save/reload.
- Adding a spec bumps two hard-coded counts (ComponentBaseTests.Specs_AreEmbeddedForEveryComponent, StatusTests.Specs_HaveAWorkerComponent: 27 now); `python tools/icons.py` rewrites five other icons byte-differently, write only the new one.
- Known failures outside this work: 5 pyrato `center_time` analysis tests, the test_pb entry-point test, and RealWorker_AnalysesAResponseAndReadsASoundFile.
