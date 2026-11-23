---
title: "Merging main (clean-up PR #37) into feat/coefficient-lab: conflicts and how to resolve"
author: Achilleas
date: 2026-11-23
branch: feat/coefficient-lab
cost: { minutes: 1, tokens: 47436 }
touches:
  - tests/test_worker.py
  - CHANGELOG.md
---

## Decided
2026-11-23: bring origin/main into feat/coefficient-lab by MERGE, not rebase. The branch already holds a merge of main (a78dbc2), and main's 19-commit clean-up (PR #37) renamed and moved a lot of code, so a rebase would replay 75 commits over it and hit the same conflicts again and again. Not merged yet: waiting on the user.

## Tried and ruled out
Rebasing: it would flatten a78dbc2, and most of the branch (121 commits) isn't pushed to origin/feat/coefficient-lab yet.

## Watch out
A trial merge (git merge-tree --write-tree HEAD origin/main) gives 8 conflicts:
- worker/api.py: main moved all imports to the top of the module and added sound_info to __all__. Keep both sides' imports and __all__ entries, plus the lab section.
- tests/test_worker.py: main removed install_worker's `hybrid` param and the "hybrid" key in worker.json. Take main's version but keep the `compas_acoustics[lab] @` install string. install.py itself merges cleanly and right.
- ComponentBaseTests.cs / StatusTests.cs: main renamed ComponentSpec.Names() to EmbeddedSpecs.Names() (24 specs). The count with the lab is probably 27; confirm it with the tests.
- Choices.cs, Control.cs, ArchiveCompatibilityTests.cs: main only reworded comments next to the branch's new members/tests. Keep both.
- CHANGELOG.md: keep both sides' entries.
ruff finds no undefined names in the Python that merges without conflicts. The C# couldn't be built with conflicts open: after resolving, run dotnet test and pytest.
