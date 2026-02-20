from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("local_utils/compute_points.pyx"),
)
