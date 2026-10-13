from __future__ import print_function

import glob
import os
import shutil
import sys

from compas_invocations2 import build
from compas_invocations2 import docs
from compas_invocations2 import style
from compas_invocations2 import tests
from invoke import Collection
from invoke import task


@task(pre=[build.build_cpython_ghuser_components], help={"output": "Folder where the installer bundle is created (default: dist/)"})
def installer_bundle(ctx, output="dist"):
    """Build the Grasshopper components and a wheel, and zip the wheel with the Grasshopper installer."""
    base = os.path.dirname(__file__)
    bundle = os.path.join(base, output, "compas_acoustics_installer")
    if os.path.isdir(bundle):
        shutil.rmtree(bundle)
    os.makedirs(bundle)

    with ctx.cd(base):
        ctx.run('"{}" -m pip wheel --no-deps --wheel-dir "{}" .'.format(sys.executable, bundle))
    shutil.copy2(os.path.join(base, "installer", "compas_acoustics_installer.gh"), bundle)

    wheels = glob.glob(os.path.join(bundle, "compas_acoustics-*.whl"))
    archive = shutil.make_archive(bundle, "zip", os.path.dirname(bundle), os.path.basename(bundle))
    print("Bundled {} and the installer into {}".format(os.path.basename(wheels[0]), archive))

ns = Collection(
    docs.help,
    style.check,
    style.lint,
    style.format,
    docs.docs,
    docs.linkcheck,
    tests.test,
    tests.testdocs,
    tests.testcodeblocks,
    build.prepare_changelog,
    build.clean,
    build.release,
    build.build_cpython_ghuser_components,
    installer_bundle,
)
ns.configure(
    {
        "base_folder": os.path.dirname(__file__),
        "ghuser_cpython": {
            "source_dir": "src/compas_acoustics_ghpython/components",
            "target_dir": "src/compas_acoustics_ghpython/components/ghuser",
            "prefix": "compas_acoustics: ",
        },
    }
)
