from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mtplx_native_ple_cpu_rows",
        version="0.0.0",
        description="MLX-owned CPU-stream PLE row staging primitive for MTPLX.",
        ext_modules=[extension.CMakeExtension("mtplx_native_ple_cpu_rows._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_native_ple_cpu_rows"],
        package_data={"mtplx_native_ple_cpu_rows": ["*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
