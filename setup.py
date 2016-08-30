from setuptools import setup

APP = ['Pythion']

setup(name='Pythion',
      version='1.0',
      description='Nanopore Data Analysis Package',
      author='Robert Henley',
      author_email='roberthenley89@gmail.com',
      url='https://github.com/rhenley/Pyth-Ion/',
      packages = ['Pythion'],
#      scripts = ['main.py','PlotGUI.py'],
      data_files=[('Sample Data', ['3500bp-200mV.mat','3500bp-200mV.log'])]
     )
