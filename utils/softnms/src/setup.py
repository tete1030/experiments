# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        "softnms",
        ["softnms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs = [numpy_include]
    )
]

setup(
    name='softnms',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': build_ext},
)
