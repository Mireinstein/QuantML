"""Builds the quantiq_cpp extension module from the real C++ order book.

    python3 setup.py build_ext --inplace
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "quantiq_cpp",
        ["../cpp/src/order_book.cpp", "../cpp/src/bindings.cpp"],
        include_dirs=["../cpp/include"],
        cxx_std=17,
    ),
]

setup(
    name="quantiq_cpp",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
