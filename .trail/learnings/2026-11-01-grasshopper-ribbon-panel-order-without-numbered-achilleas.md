---
title: Grasshopper ribbon panel order without numbered subcategories
author: Achilleas
date: 2026-11-01
branch: release/public-prep
cost: { minutes: 2, tokens: 67510 }
touches:
  - src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/Components/ComponentBase.cs
  - tools/icons.py
---

## Decided
- metadata.json "subcategory" holds plain names (Elements, Model, ...); tools/icons.py ACCENTS keys match them.
- Ribbon order lives in one list, ComponentBase.Subcategories; ComponentBase.RibbonSubcategory pads each name with leading spaces (one more than the panel after it). Params.cs uses RibbonSubcategory("Params"). Order: Elements, Params, Model, Simulation, Analysis, Auralisation, Display, Util, Optimization.
- ComponentBaseTests checks the padded names sort in order under Ordinal, InvariantCulture and CurrentCulture.

## Tried and ruled out
- Zero-width spaces (U+200B) as the prefix: GH_RibbonPanel sorting is culture-aware on Rhino 8 / .NET 10 (macOS, ICU) and ignores them, so panels came out alphabetical. Checked live via lamcp by sorting GH_RibbonPanel objects.

## Watch out
- Rhino loads a GHA only at startup: a live ribbon check needs the build copied into Grasshopper's Libraries/compas_acoustics folder and a Rhino restart.
- Windows was not checked live; the reasoning is that a space sorts before letters under NLS, ICU and ordinal comparison alike.
