import numpy as np
import matplotlib.pyplot as plotter
import os


tmpdir = os.getenv("XDG_RUNTIME_DIR")
results_path = tmpdir + "/uqtest.npy"
assert os.path.isfile(results_path)
solutions_data = list(np.load(results_path, allow_pickle=True))

# Only one entry for each number of evals
# ~ data_uniq = dict()
# ~ for evals, errs in solutions_data:
	# ~ data_uniq[evals] = errs

solutions_data.sort(key = lambda v: v[0])
xvalues = [v[0] for v in solutions_data]
mean_errs = np.array([v[1] for v in solutions_data]).T
mean_err_descs = ("E prey", "E predator", "P10 prey", "P10 predator",
	"P90 prey", "P90 predator", "Var prey", "Var predator")

figure = plotter.figure(1, figsize=(13,10))
figure.canvas.set_window_title('Stocha')

for i,desc in enumerate(mean_err_descs):
	plotter.subplot(4, 2, 1 + i)
	plotter.plot(xvalues, mean_errs[i], label=desc)
	plotter.xlabel('function evaluations')
	plotter.ylabel('mean relative error')
	plotter.yscale("log")
	plotter.legend(loc=2)
	plotter.grid(True)

# ~ fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
# ~ plotter.savefig(fileName, format='pdf')

plotter.show()
