"""The native executor's module is loaded before any test of this folder runs.

The module learns each Metal kernel when MLX compiles it; a kernel compiled
before the module was in the process is unknown to it. Loading here, at
collection, puts it ahead of every kernel these tests build.
"""

from mtplx.engine.site import load_early

load_early()
