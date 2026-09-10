from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mtplx_native_qsa",
        version="0.0.0",
        description="Native MLX direct-index sparse-GQA attention for MTPLX QSA.",
        ext_modules=[extension.CMakeExtension("mtplx_native_qsa._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_native_qsa"],
        package_data={"mtplx_native_qsa": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
