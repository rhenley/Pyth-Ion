from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from Cython.Distutils import build_ext

setup(
    name='PythIon',
    version='0.2.0',
    author='Robert Henley',
    author_email='roberthenley89@gmail.com',
    packages=['PythIon'],
    description='Nanopore Data Analysis package.'
    
)