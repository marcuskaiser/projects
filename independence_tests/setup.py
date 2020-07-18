from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.annotate = True

setup(
    name='*',
    ext_modules=cythonize((
        Extension("*", sources=["*.pyx"],
                  include_dirs=[np.get_include()]),
    )),
    zip_safe=False)
