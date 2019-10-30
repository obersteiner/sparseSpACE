This folder contains Python programs for testing the single dimension spatial
adaptive refinement with the UncertaintyQuantification grid
operation.


## Error Plots

The Python programs with the `CalculateErrors.py` or a similar suffix
calculate solutions
for a test function with various integration methods and number of evaluations
and then save relative or absolute deviations between the results
and a reference solution to a temporary file in `$XDG_RUNTIME_DIR`.
Programs with the `PlotErrors.py` create a plot which shows the calculated
errors of the respective function.
Before rerunning the tests after changing the code,
the previous errors need to be removed.
```sh
# Calculate errors for the G-function test
python3 GFunctionCalculateErrors.py
# Plot the errors
python3 GFunctionPlotErrors.py
# Remove the calculated errors
rm $XDG_RUNTIME_DIR/uqtest*
```
The errors are saved in a file instead of directly plotting them in the error
calculation programs
so that it is possible to abort the calculations at any point in time
and plot the errors calculated so far;
since the program may be used during testing,
it could abort due to a crash.

`$XDG_RUNTIME_DIR` is the path to a tmpfs file system on many linux systems;
on other operating systems the path for saving and loading
the errors may need to be changed
in the error calculation and plot Python programs.

Tests for the Predator-Prey model are located in a subfolder
because there are many of them.
The `ErrorsOverTime.py` program calculates solutions and errors
for a fixed number of evaluations and quadrature method,
and shows them on plots.


## Other Test Programs

`TestsUQ.py` contains many test cases.
If it is `import`ed in a jupyter notebook,
the program plots functions, refinement objects, sparse grids and more.
