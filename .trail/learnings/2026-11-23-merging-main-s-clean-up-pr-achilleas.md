---
title: "Merging main's clean-up (PR #37) into feat/coefficient-lab: the breaks git did not flag"
author: Achilleas
date: 2026-11-23
branch: feat/coefficient-lab
cost: { minutes: 8, tokens: 72917 }
touches:
  - tests/docs/test_specs.py
---

## Decided
- 2026-11-23: merged origin/main (clean-up PR #37) into feat/coefficient-lab as 198d493 (not pushed). pytest: 1362 passed. dotnet test: 537 passed.
- Main's clean-up 7 removed reading saved Grasshopper inputs by name (ControlPorts.ReadParameters) and its old-definition tests. Main's policy wins: the lab was never released, so the test for Scattering Sample's removed `name` input was dropped, and so was the CHANGELOG promise. Don't bring ReadParameters back for the lab.

## Tried and ruled out
- Trusting a clean merge: the C# didn't compile. AsyncRuns is now BackgroundRuns, and MeshConversion.ToFrameData was removed as unused, but the lab's plane input needs it (it's back, with a test; ToPlane stays removed). In Python, api._messages(caught) is gone: use `with _user_warnings() as messages:`.

## Watch out
- Main added ruff W505 (docstrings at most 120 columns): 371 lines of the lab's code break it. It's not fixed yet, and the lint will fail until it is.
- tests/docs/test_specs.py had been counting 28 specs since 83144cf removed Lab Material: there are 27 (main's 24 + the lab's 3).
- zsh: `git show $B:src/...` breaks, because `:s` is a zsh modifier. Write "${B}:src/...".
