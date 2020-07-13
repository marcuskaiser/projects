from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy as np

setup(
    name='*',
    ext_modules=cythonize((
        Extension("*", sources=["*.pyx"],
                  include_dirs=[np.get_include()],
                  annotate=True),
    )),
    zip_safe=False,
    annotate=True
)
