from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
#setup(ext_modules= cythonize("counter.pyx"));
# setup(
#     ext_modules = cythonize('counter.pyx', compiler_directives={'language_level': 3}),
# )

setup(
    cmdclass={'build_ext': build_ext}, ext_modules=[Extension('counter', ["counter.pyx"])]
)