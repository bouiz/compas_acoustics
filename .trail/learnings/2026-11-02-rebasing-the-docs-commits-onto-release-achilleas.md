---
title: "Rebasing the docs commits onto release/public-prep's description and test cleanup"
author: Achilleas
date: 2026-11-02
branch: release/public-prep
cost: { minutes: 6, tokens: 51920 }
touches:
  - tests/test_docs.py
  - docs/_ext/ghcomponents/render.py
  - docs/_ext/ghcomponents/__init__.py
  - docs/_images/icons/CA_Acoustic_Object.png
  - docs/_images/icons/CA_Analyse_IR.png
  - docs/_images/icons/CA_Audio_Player.png
  - docs/_images/icons/CA_Auralise.png
  - docs/_images/icons/CA_Channel.png
  - docs/_images/icons/CA_Construct_Material.png
  - docs/_images/icons/CA_Deconstruct_IR.png
---

## Decided
- 2026-11-02: rebased 19 local docs/ribbon commits onto 4 remote commits (a6be1ab descriptions rewrite, ed5a7fe unit-only tests, fd37af5, 3bfc6d0 installer 'worker' ignored).
- Component Specs/*/metadata.json conflicts: keep upstream's description, strip the "NN " prefix from subcategory (the local commit e33cc04 changed only subcategories).
- tests/test_release_metadata.py stays deleted (upstream dropped it on purpose).
- README install section lives in docs/installation/{python,grasshopper}.rst now; upstream README wording changes must be carried into those pages by hand.
- tests/test_docs.py now checks docs/python/examples/<name>.rst for "/../examples/<name>.py" and docs/python/tutorial.rst; shoebox is simple_simulation_shoebox.

## Tried and ruled out
- Taking either side of the metadata.json conflicts wholesale: one side loses the new descriptions, the other the unnumbered subcategories.

## Watch out
- In a rebase, --ours is upstream, --theirs the replayed commit.
- A clean rebase can still break tests: upstream's new tests may assume the old docs layout. Run pytest and `dotnet test` in src/compas_acoustics_ghcsharp after rebasing.
