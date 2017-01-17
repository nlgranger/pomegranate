from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
    ext = 'c'
else:
    use_cython = True
    ext = 'pyx'

filenames = [ "base",
              "distributions",
              "Mixture"
            ]

if not use_cython:
    extensions = [
        Extension( "pomegranate.{}".format( name ), 
                   [ "pomegranate/{}.{}".format( name, ext ) ], 
                   include_dirs=[np.get_include()] ) for name in filenames
    ]
else:
    extensions = [
            Extension( "pomegranate.{}".format(name),
                       [ "pomegranate/{}.pyx".format(name) ],
                       include_dirs=[np.get_include()] ) for name in filenames
    ]

    extensions = cythonize( extensions )

setup(
    name='pomegranate',
    version='0.6.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['pomegranate'],
    url='http://pypi.python.org/pypi/pomegranate/',
    license='LICENSE.txt',
    description='Pomegranate is a graphical models library for Python, implemented in Cython for speed.',
    ext_modules=extensions,
    install_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "networkx >= 1.8.1",
        "scipy >= 0.17.0"
    ],
    package_data={
        'pomegranate': ['*.pxd']
    }
)
