"""Generate the pages and run the checks: ``python docs/_ext/ghcomponents``, as ``invoke docs-components`` does."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghcomponents import DocsCheckError  # noqa: E402
from ghcomponents import generate  # noqa: E402

try:
    written = generate()
except DocsCheckError as error:
    print(error, file=sys.stderr)
    sys.exit(1)
print("Generated {} documentation files.".format(len(written)))
