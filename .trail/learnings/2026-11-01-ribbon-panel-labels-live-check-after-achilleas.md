---
title: "Ribbon panel labels: live check after the leading-space ordering"
author: Achilleas
date: 2026-11-01
branch: release/public-prep
cost: { minutes: 5, tokens: 15736 }
---

## Decided
- Checked live in Rhino 8 (macOS, .NET 10): the COMPAS Acoustics tab shows Elements, Model, Simulation, Analysis, Auralisation, Display, Util, Optimization in that order. The Params panel never shows because every param is hidden.
- Kept plain spaces as the prefix. Narrow panels (Model, Simulation, Analysis, Auralisation, Optimization) truncate their labels, but the old "0N " names truncated the same panels, so nothing got worse.

## Tried and ruled out
- Thin space (U+2009) and hair space (U+200A) as narrower prefixes: GH's culture-aware sort keeps them in order, but ordinal comparison puts them after letters. A hair space still adds ~7 px for 8 characters, so Model (32 px bar) would truncate anyway.

## Watch out
- GH_RibbonPanel width comes from its icons, not its label; labels never widen a panel. Measure with panel.RibbonFont() against panel.PanelBarRegion().Width.
