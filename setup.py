import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="sparseSpACE",
    version="1.0.0",
    description="sparseSpACE - the Sparse Grid Spatially Adaptive Combination Environment implements different variants of the spatially adaptive combination technique",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/obersteiner/sparseSpACE",
    author="Michael Obersteiner",
    author_email="michael.obersteiner@mytum.de",
    license="LGPL-3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
    ],
    packages=["sparseSpACE"],
    include_package_data=True,
    install_requires=["numpy", "scipy","matplotlib","dill","scikit-learn","chaospy","sympy"],
)

