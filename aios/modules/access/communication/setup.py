# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Platform-specific compiler optimizations
extra_compile_args = []
if os.name == 'posix':  # Linux/Mac
    extra_compile_args = [
        '-O3',  # Maximum optimization
        # '-march=native',  # CPU-specific optimizations
        '-ftree-vectorize',  # Enable vectorization
        '-ffast-math'  # Fast math operations
    ]
elif os.name == 'nt':  # Windows
    extra_compile_args = [
        '/O2',  # Maximum optimization
        '/arch:AVX2'  # Use Advanced Vector Extensions
    ]

ext_modules = [
    Extension(
        "message_queue",
        ["message_queue.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name='message_queue',
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
            'infer_types': True,
        },
        annotate=True  # Generate HTML annotation file
    ),
)