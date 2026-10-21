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


@task(help={"version": "Version substituted into the component templates (optional)"})
def build_cpython_ghuser_components(ctx, version=None):
    """Build the Rhino 8 CPython Grasshopper components (.ghuser) with ghpython_componentizer."""
    config = ctx.ghuser_cpython
    with ctx.cd(ctx.base_folder):
        shutil.rmtree(config.target_dir, ignore_errors=True)
        cmd = '"{}" -m ghpython_componentizer "{}" "{}" --prefix "{}"'.format(sys.executable, config.source_dir, config.target_dir, config.prefix)
        if version:
            cmd += ' --version "{}"'.format(version)
        ctx.run(cmd)


@task(pre=[build_cpython_ghuser_components])
def pre_build(ctx):
    """Generate the files that go into the wheel but are not in git: the Grasshopper components."""


@task(pre=[pre_build], help={"output": "Folder where the installer bundle is created (default: dist/)"})
def installer_bundle(ctx, output="dist"):
    """Build the Grasshopper components and a wheel, and zip the wheel with the Grasshopper installer."""
    base = os.path.dirname(__file__)
    bundle = os.path.join(base, output, "compas_acoustics_installer")
    if os.path.isdir(bundle):
        shutil.rmtree(bundle)
    os.makedirs(bundle)
    # setuptools copies into build/lib without emptying it, so removed modules and components would ship again
    shutil.rmtree(os.path.join(base, "build"), ignore_errors=True)

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
    build_cpython_ghuser_components,
    pre_build,
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
