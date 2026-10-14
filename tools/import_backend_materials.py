"""Import acoustic materials from simulation backends into the compas_acoustics materials library.

Materials are fetched straight from the upstream GitHub repositories, converted to
:class:`compas_acoustics.AcousticMaterial` (7 octave bands, 125 Hz - 8 kHz), assigned a
:class:`compas_acoustics.MaterialCategory` and written to the materials library as
``<key>_from_<backend>.json``, named ``"<key> (from <backend>)"``.

The script can be re-run at any time: the files previously imported from a backend
are replaced, hand-made materials in the library are left untouched.

Usage::

    python tools/import_backend_materials.py                      # all backends
    python tools/import_backend_materials.py pyroomacoustics      # only one
    python tools/import_backend_materials.py --dry-run            # show what would change

Set ``GITHUB_TOKEN`` to avoid GitHub API rate limits.

Backends:

* pyroomacoustics (MIT): ``pyroomacoustics/data/materials.json``. Entries with only
  scattering coefficients (diffusers, audience scattering) are skipped, since an
  ``AcousticMaterial`` requires absorption.
* misuka (BSD-3/MIT): has no material database, materials are defined per scene
  (see its tutorials). The script checks for one and reports it if it ever appears.

edg-acoustics is intentionally not imported: it is GPL-3.0 licensed and only ships
three example boundary conditions (carpet, panel, hard wall), fitted for
normal incidence below 3 kHz.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

from compas_acoustics import AcousticMaterial
from compas_acoustics import Defaults
from compas_acoustics import MaterialCategory as C
from compas_acoustics import MaterialsLibrary

BANDS = Defaults().DEFAULT_FREQUENCY_BANDS

# ============================================================================
# helpers
# ============================================================================


def _request(url):
    request = urllib.request.Request(url, headers={"User-Agent": "compas_acoustics-material-import"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", "Bearer {}".format(token))
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def github_commit(repo, ref):
    """Resolve a branch/tag to a commit SHA, so imports are traceable."""
    data = json.loads(_request("https://api.github.com/repos/{}/commits/{}".format(repo, ref)))
    return data["sha"]


def github_tree(repo, sha):
    data = json.loads(_request("https://api.github.com/repos/{}/git/trees/{}?recursive=1".format(repo, sha)))
    return [item["path"] for item in data["tree"] if item["type"] == "blob"]


def github_raw(repo, sha, path):
    return _request("https://raw.githubusercontent.com/{}/{}/{}".format(repo, sha, path))


def pad_to_bands(coefficients):
    """Pad coefficients given for the lower bands only by repeating the last value."""
    coefficients = [round(float(c), 4) for c in coefficients]
    missing = len(BANDS) - len(coefficients)
    return coefficients + [coefficients[-1]] * missing, missing


def material_name(key, backend):
    return "{} (from {})".format(key, backend)


# ============================================================================
# pyroomacoustics
# ============================================================================

PRA_REPO = "LCAV/pyroomacoustics"
PRA_PATH = "pyroomacoustics/data/materials.json"

# default category for each group of the pyroomacoustics database
PRA_GROUPS = {
    "Massive constructions and hard surfaces": C.MASONRY,
    "Lightweight constructions and linings": C.PLASTER_AND_GYPSUM,
    "Glazing": C.GLASS,
    "Wood": C.WOOD,
    "Floor coverings": C.FLOOR_COVERINGS,
    "Curtains": C.TEXTILES_AND_CURTAINS,
    "Seating": C.SEATING,
    "Audience": C.AUDIENCE,
    "Wall absorbers": C.ABSORBERS,
    "Ceiling absorbers": C.ABSORBERS,
    "Special absorbers": C.ABSORBERS,
}

# materials that belong to a different category than their group
PRA_OVERRIDES = {
    "rough_concrete": C.CONCRETE,
    "unpainted_concrete": C.CONCRETE,
    "concrete_floor": C.CONCRETE,
    "rough_lime_wash": C.PLASTER_AND_GYPSUM,
    "ceramic_tiles": C.STONE_AND_TILES,
    "limestone_wall": C.STONE_AND_TILES,
    "marble_floor": C.STONE_AND_TILES,
    "reverb_chamber": C.OTHER,
    "wooden_lining": C.WOOD,
    "facing_brick": C.MASONRY,
    "acoustical_plaster_25mm": C.PLASTER_AND_GYPSUM,
    "gypsum_board": C.PLASTER_AND_GYPSUM,
    "perforated_veneered_chipboard": C.WOOD,
    "ceiling_plasterboard": C.PLASTER_AND_GYPSUM,
    "ceiling_perforated_gypsum_board": C.PLASTER_AND_GYPSUM,
    "ceiling_metal_panel": C.METAL,
    "microperforated_glass_sheets": C.GLASS,
}


def pra_group_category(group):
    # groups carry a qualifier, e.g. "Seating (2 seats per m2)"
    for prefix, category in PRA_GROUPS.items():
        if group.startswith(prefix):
            return category
    raise ValueError("Unknown pyroomacoustics material group '{}': add it to PRA_GROUPS.".format(group))


def import_pyroomacoustics(ref):
    backend = "pyroomacoustics"
    sha = github_commit(PRA_REPO, ref)
    database = json.loads(github_raw(PRA_REPO, sha, PRA_PATH))

    if database["center_freqs"] != BANDS:
        raise ValueError("pyroomacoustics bands {} differ from {}".format(database["center_freqs"], BANDS))

    materials = []
    for group, entries in database["absorption"].items():
        group_category = pra_group_category(group)
        for key, entry in entries.items():
            coefficients, padded = pad_to_bands(entry["coeffs"])
            notes = ["{}.".format(entry["description"].rstrip(".")), "Imported from pyroomacoustics ({}) at {}.".format(group, sha[:7])]
            if padded:
                notes.append("Not given above {} Hz upstream, the last value is repeated.".format(BANDS[-padded - 1]))
            materials.append(
                AcousticMaterial(
                    name=key,
                    absorption_coefficients=coefficients,
                    description=" ".join(notes),
                    category=PRA_OVERRIDES.get(key, group_category),
                    data_origin=backend,
                )
            )

    skipped = sorted(key for entries in database["scattering"].values() for key in entries)
    print("  skipped {} scattering-only entries (no absorption data): {}".format(len(skipped), ", ".join(skipped)))
    return materials


# ============================================================================
# misuka
# ============================================================================

MISUKA_REPO = "misuka-renderer/misuka"


def import_misuka(ref):
    sha = github_commit(MISUKA_REPO, ref)
    candidates = [p for p in github_tree(MISUKA_REPO, sha) if re.search(r"material", p, re.I) and p.endswith((".json", ".csv", ".yaml", ".yml", ".xml"))]
    if candidates:
        print("  misuka now has material-like data files, consider writing an importer: {}".format(", ".join(candidates)))
    else:
        print("  misuka has no material database at {} (materials are defined per scene), nothing to import".format(sha[:7]))
    return []


# ============================================================================
# main
# ============================================================================

IMPORTERS = {
    "pyroomacoustics": import_pyroomacoustics,
    "misuka": import_misuka,
}


def generated_files(library_path, backend):
    suffix = "_from_{}.json".format(re.sub(r"[^a-z0-9.]+", "_", backend))
    return sorted(f for f in os.listdir(library_path) if f.endswith(suffix))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backends", nargs="*", metavar="backend", help="backends to import: {} (default: all)".format(", ".join(IMPORTERS)))
    parser.add_argument("--ref", default="HEAD", help="branch, tag or commit to import from (default: the default branch)")
    parser.add_argument("--library", default=Defaults().MATERIALS_LIBRARY_PATH, help="materials library folder")
    parser.add_argument("--dry-run", action="store_true", help="only report what would change")
    args = parser.parse_args()
    unknown = set(args.backends) - set(IMPORTERS)
    if unknown:
        parser.error("unknown backend(s): {}".format(", ".join(sorted(unknown))))

    for backend in args.backends or list(IMPORTERS):
        print("{}:".format(backend))
        materials = IMPORTERS[backend](args.ref)

        names = [m.name for m in materials]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate material names from {}".format(backend))

        stale = generated_files(args.library, backend)
        print("  {} materials imported, replacing {} previously imported files".format(len(materials), len(stale)))
        for material in materials:
            print("    [{}] {}".format(material.category.value, material.name))

        if args.dry_run:
            continue
        for filename in stale:
            os.remove(os.path.join(args.library, filename))
        for material in materials:
            material.to_json(os.path.join(args.library, "{}.json".format(material.slug)), pretty=True, minimal=True)

    library = MaterialsLibrary(args.library)
    print("\nLibrary now has {} materials:".format(len(library.material_names)))
    for category, names in library.materials_by_category.items():
        print("  {:<20} {}".format(category.value, len(names)))


if __name__ == "__main__":
    sys.exit(main())
