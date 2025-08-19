from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "vtk_slice_backend",                 # name of the compiled module
        ["vtk_slice_backend.cpp"],           # C++ source file
        include_dirs=[pybind11.get_include()],
        language="c++",
    )
]

setup(
    name="vtk_slice_backend",
    version="0.1",
    py_modules=[],         # <- no automatic packages
    ext_modules=ext_modules,
)
