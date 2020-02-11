from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from subprocess import call
import numpy

setup(name='py_quadtree',
        ext_modules=cythonize([
            Extension('py_quadtree', ['py_quadtree.pyx'],
                include_dirs=[numpy.get_include()],
                extra_compile_args=[
                    '-lm', '-Ofast', '-finline-functions',
                    #'-DQT_MBMI2', '-mbmi2'
                    ]
                )]))
