from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

APP = ['Pythion']

setup(name='Pythion',
      version='1.1',
      description='Nanopore Data Analysis Package',
      author='Robert Henley',
      author_email='roberthenley89@gmail.com',
      url='https://github.com/rhenley/Pyth-Ion/', 
      packages=find_packages()
#      data_files=[('C:\Users\rober\Documents\GitHub\Pyth-Ion', 
#                   ['3500bp-200mV.mat','3500bp-200mV.log'])]
     )
