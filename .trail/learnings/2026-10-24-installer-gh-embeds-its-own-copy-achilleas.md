---
title: Installer .gh embeds its own copy of installer.py
author: Achilleas
date: 2026-10-24
branch: feat/coefficient-lab
cost: { minutes: 1, tokens: 49004 }
touches:
  - installer/compas_acoustics_installer.gh
  - installer/installer.py
  - tasks.py
---

## Decided
installer/compas_acoustics_installer.gh holds a copy of installer/installer.py as the script component's text; Grasshopper runs that copy, not the .py file. The .gh is GH_IO raw-deflate compressed: zlib.decompress(data, -15), and the script is a base64 run of 400+ chars inside it, so a test can extract it and compare it with installer.py (normalise CRLF).

## Tried and ruled out
Searching the decompressed .gh for function names in plain text finds nothing: the script is base64-encoded.

## Watch out
fb7e6cc (C# plugin) rewrote installer.py (install_gha instead of install_userobjects) but did not update the .gh. The bundle then shipped the old script, which queries compas_acoustics_ghpython, a package the wheel no longer ships, and the install failed after pip succeeded. Whenever installer.py changes, paste it into the component in Rhino and save the .gh in the same commit. tasks.py installer_bundle copies the .gh unchecked.
