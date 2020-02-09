from distutils.core import setup
from Cython.Build import cythonize
from subprocess import call
import numpy

setup(name='py_quadtree',
        ext_modules=cythonize('py_quadtree.pyx', include_path=[numpy.get_include()]))
