import numpy as np
import matplotlib.pyplot as plotter
import sys
import os

tmpdir = os.getenv("XDG_RUNTIME_DIR")
results_path = tmpdir + "/uqtestG.npy"
assert os.path.isfile(results_path)
solutions_data = list(np.load(results_path, allow_pickle=True))

data_spars = []
data_spars_ho = []
data_cp = []
for v in solutions_data:
	(num_evals, use_spatially_adaptive, do_HighOrder, err_E, err_Var) = v
	if use_spatially_adaptive:
		if do_HighOrder:
			data_spars_ho.append((num_evals, err_E, err_Var))
			continue
		data_spars.append((num_evals, err_E, err_Var))
		continue
	data_cp.append((num_evals, err_E, err_Var))
# data_* should be sorted here
data_spars = np.array(data_spars).T
data_spars_ho = np.array(data_spars_ho).T
data_cp = np.array(data_cp).T

figure = plotter.figure(1, figsize=(15,10))
figure.canvas.set_window_title('Stocha')

for i,desc in enumerate(("E", "Var")):
	plotter.subplot(2, 1, 1 + i)
	plotter.plot(data_spars[0], data_spars[i + 1], label=f"adaptive {desc}")
	plotter.plot(data_spars_ho[0], data_spars_ho[i + 1], label=f"adaptive HO {desc}")
	plotter.plot(data_cp[0], data_cp[i + 1], label=f"CC {desc}")
	plotter.xlabel('function evaluations')
	plotter.ylabel('relative error')
	plotter.yscale("log")
	plotter.legend(loc=2)
	plotter.grid(True)


fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()
