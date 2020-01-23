from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("build.first",    # location of the resulting .so
             ["src/first.pyx"], include_dirs=[numpy.get_include()]),
]


setup(name='package',
      packages=find_packages(),
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
     )
