import os
import sys

from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    # CMake can otherwise select a framework Python outside the venv; both
    # hints are needed because find_package(Python ...) and any transitive
    # find_package(Python3 ...) read different cache variables.
    os.environ.setdefault("CMAKE_ARGS", "")
    os.environ["CMAKE_ARGS"] = " ".join(
        part
        for part in (
            os.environ["CMAKE_ARGS"],
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        )
        if part
    )
    setup(
        name="mtplx_qsa_kernels",
        version="0.0.1",
        description=(
            "Qwen4 QSA sparse-GQA Steel Metal kernel for MTPLX "
            "(vendored from oMLX PR #3244, Apache-2.0)."
        ),
        ext_modules=[extension.CMakeExtension("mtplx_qsa_kernels._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_qsa_kernels"],
        package_data={
            "mtplx_qsa_kernels": [
                "*.so",
                "*.dylib",
                "*.metallib",
                "LICENSE.txt",
                "NOTICE",
                "MLX_LICENSE.txt",
                "MLX_SERVE_LICENSE.txt",
            ]
        },
        zip_safe=False,
        python_requires=">=3.11",
    )
