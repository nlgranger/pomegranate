from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize


files = ["base",
         "utils",
         "distributions/normal",
         "distributions/uniform"]

extensions = [
    Extension("pomegranate.{}".format(file.replace("/", ".")),
        ["pomegranate/{}.pyx".format(file)],
        include_dirs=[np.get_include()])
    for file in files
]

setup(
    name='pomegranate',
    version='0.6.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['pomegranate'],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='LICENSE',
    description="Pomegranate is a graphical models library for Python, " \
        "implemented in Cython for speed.",
    ext_modules=cythonize(extensions),
    install_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "networkx >= 1.8.1",
        "scipy >= 0.17.0"
    ],
    extras_require={
        'doc': ['numpydoc'],
    },
    package_data={
        'pomegranate': ['*.pxd']
    }
)
