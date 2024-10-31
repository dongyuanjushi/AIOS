# setup.py
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy

extensions = [
    Extension(
        "fifo_scheduler_core",
        ["fifo_scheduler_core.pyx"],
        language="c++",
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"]
    )
]

setup(
    ext_modules=cythonize(extensions)
)