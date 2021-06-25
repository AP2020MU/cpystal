from setuptools import setup, find_packages

setup(
    name="cpystal",
    version="0.0.0",
    description='This is a Python package for dealing with crystals, experimental data of physical property.',
    packages=find_packages(where='cpystal'),
    package_dir={'': 'cpystal'},
)