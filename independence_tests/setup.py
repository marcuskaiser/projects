from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

# Instruction: Build with python setup.py build_ext

Options.annotate = True

ext_modules = cythonize([
    Extension("*", sources=["./hsic/*.pyx"],
              include_dirs=[np.get_include()]),
    Extension("*", sources=["./dcorr/*.pyx"],
              include_dirs=[np.get_include()]),
    Extension("*", sources=["./corr/*.pyx"],
              include_dirs=[np.get_include()]),
])

setup(name='*',
      ext_modules=ext_modules,
      zip_safe=False)
