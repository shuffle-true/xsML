from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = { 'build_ext': build_ext},
    ext_modules = [Extension("_gradient_boosting", ['_gradient_boosting.pyx'], include_dirs=[np.get_include(), "."])]
)
