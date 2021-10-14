from distutils.core import setup

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="cpystal",
    version="0.0.0",
    description="This is a Python package for dealing with crystals and experimental data of their physical property.",
    packages=[""],
    package_dir={"": "cpystal"},
    install_requires=_requires_from_file("/Users/ut/Desktop/AP2020MU/cpystal/requirements.txt"),
    author="AP2020MU",
    url='https://github.com/AP2020MU/cpystal',
)