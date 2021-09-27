from setuptools import setup, find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="cpystal",
    version="0.0.0",
    description="This is a Python package for dealing with crystals and experimental data of physical property.",
    packages=find_packages(where="cpystal"),
    package_dir={"": "cpystal"},
    install_requires=_requires_from_file("requirements.txt"),
    author="AP2020MU",
    url='https://github.com/AP2020MU/cpystal',
)