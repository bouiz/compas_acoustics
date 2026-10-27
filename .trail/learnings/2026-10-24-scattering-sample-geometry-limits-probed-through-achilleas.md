---
title: "Scattering Sample geometry limits, probed through make_scattering_sample"
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 13, tokens: 94710 }
---

## Decided
- Limits of CA_Scattering_Sample were mapped by running 22 synthetic meshes through worker.api.make_scattering_sample (the component is a thin wrapper; LabRequests.MeshOf meshes Brep/Surface/Extrusion/Box/SubD with QualityRenderMesh and appends parts). Visual guide published as an artifact: https://claude.ai/artifact/LaqyKvi1RTNn6Qm63FR7mC
- Accepted: open relief with one rim (+Z, or any orientation with `plane`), vertical faces incl. at the rim, unwelded quads, non-convex simple outlines (ear clipping), disc tiles that tile seamlessly, closed bodies with a flat bottom, proxy of the same kind and outline (as_drawn only).
- Rejected (ValueError): holes / separate parts (found N outlines), pinched outline or a fin on an inner edge (non-manifold), disc tile of islands, footprint with no XY area without `plane`, closed body without flat bottom (sphere), proxy open/closed mismatch or other outline, disc without diameter.

## Tried and ruled out
- Not exercised in Grasshopper via lamcp; Brep-meshing behaviour (unjoined surfaces leaving T-junction gaps) is inferred from code, not verified.

## Watch out
- Silent mis-measure 1: closed body with a backing thicker than `baseplate` -> backing counted as depth (flat 0.1 m box gives h = 0.09 m). Already parked in the D1+D2 learning.
- Silent mis-measure 2 (new): closed body with an overhang (T-section prism) -> reference = outline of the bottom face only, i.e. the stem; d = 1.02 m instead of the cap's width. No warning.
- Bug (new): an open relief whose rim folds back over itself in plan fails with a raw "No ears found for triangulation." from earclip_polygon in _cap, not the _OUTLINE message. Catch it in _cap.
- When writing synthetic test grids, a sine of period 0.2 sampled every 0.1 m aliases to flat (h ~ 1e-17).
