"""Setuptools builds the same native executor for source and editable Apple Silicon installs."""

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeExecutor(build_ext):
    def build_extension(self, ext):
        root = Path(__file__).resolve().parent
        stage = Path(self.build_temp).resolve() / "engine"
        try:
            subprocess.run([str(root / "native/build.sh"), sys.executable, "--stage", str(stage), "--no-selftest"],
                           check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            # A source install on a Mac without the build tools still installs and runs:
            # the engine reports itself unavailable (``mtplx.engine.runtime.unavailable()``,
            # /health) and every replay site serves from MLX. Release builds set
            # MTPLX_REQUIRE_NATIVE=1 so a missing executor fails the build instead.
            if os.environ.get("MTPLX_REQUIRE_NATIVE") == "1":
                raise
            print(f"warning: mtplx native executor not built ({exc}); MTPLX runs without it", file=sys.stderr)
            # Nothing was built, so nothing may be expected: an editable install copies every
            # listed extension's output and would otherwise fail where the normal install succeeds.
            self.extensions = [other for other in self.extensions if other is not ext]
            return
        output = stage / sysconfig.get_config_var("SOABI") / "out" / ("_engine" + sysconfig.get_config_var("EXT_SUFFIX"))
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, destination)


# The release assembler builds its platform wheels from a neutral staging wheel.
native = (sys.platform == "darwin" and platform.machine() == "arm64"
          and os.environ.get("MTPLX_BUILD_BASE_WHEEL") != "1")
setup(ext_modules=[Extension("mtplx._engine", sources=[])] if native else [],
      cmdclass={"build_ext": NativeExecutor})
