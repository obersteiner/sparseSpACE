# sparseSpACE - The Sparse Grid Spatially Adaptive Combination Environment

This python projects implements different variants of the spatially adaptive Combination Technique. 
It was first targeted to solve high dimensional numerical integration with the spatially adaptive Combination Technique but it now supports the implementation of arbitrary grid operations. It supports already numerical integration, interpolation, Uncertainty Quantification, Sparse Grid Density Estimation and PDE calculations. The github page can be found [here](https://github.com/obersteiner/sparseSpACE.git).

# Installation
Install from PyPI using
```
pip install sparseSpACE
```
or (Linux example):
```
git clone https://github.com/obersteiner/sparseSpACE.git
cd sparseSpACE
pip install .
```
# Tutorials

A short introduction in how to use the framework can be found in the ipynb tutorials (see ipynb folder at https://github.com/obersteiner/sparseSpACE.git):
- Tutorial.ipynb
- Grid_Tutorial.ipynb
- Extend_Split_Strategy_Tutorial.ipynb
- Tutorial_DensityEstimation.ipynb
- Tutorial_DEMachineLearning.ipynb
-Tutorial_Extrapolation.ipynb
- Tutorial_UncertaintyQuantification.ipynb

# Plotting

The framework also supports various options for plotting the results. Examples can be found in the ipynb/Diss folder or the other Tutorials.

# Software requirements

These software requirements are automatically installed when using pip. But as a reference we list here the necessary libraries and versions:
- python3 (3.5 or higher)
- scipy (1.1.0 or higher)
- numpy
- matplotlib
- ipython3 (for Tutorials)
- ipython3 notebooks or jupyter notebook (for Tutorials)
- chaospy (for UQ)
- scikit-learn (for SGDE)
- dill (for saving/loading the current state of the refinement to/from a file)
- sympy (1.6 or higher)

# Development
For development clone the repository from github and use the configure script which will install the library in modifiable mode and apply the git hooks used for the project.
```
./configure 
```
