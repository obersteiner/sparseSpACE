import numpy as np
import matplotlib.pyplot as plotter
import sys
import os

tmpdir = os.getenv("XDG_RUNTIME_DIR")
results_path = tmpdir + "/uqtest.npy"
assert os.path.isfile(results_path)
solutions_data = list(np.load(results_path, allow_pickle=True))

typ_descs = ("full grid Gauß", "Trapez", "HighOrder", "nonadaptive Trapez")

datas = [[] for _ in typ_descs]
for v in solutions_data:
	(num_evals, typid, mean_errs) = v
	typid = int(typid)
	if typid < 0 or typid >= len(typ_descs):
		continue
	if num_evals > 900:
		# Gauss reference
		continue
	datas[typid].append((num_evals, *mean_errs))

for typid in range(len(typ_descs)):
	datas[typid] = np.array(datas[typid]).T

mean_err_descs = ("E prey relative", "E predator relative", "P10 prey", "P10 predator",
	"P90 prey", "P90 predator", "Var prey", "Var predator")

figure = plotter.figure(1, figsize=(13,10))
figure.canvas.set_window_title('Stocha')

legend_shown = False
for i,desc in enumerate(mean_err_descs):
	plotter.subplot(4, 2, 1 + i)
	for typid, typdesc in enumerate(typ_descs):
		if len(datas[typid]) < 1:
			print("No points for", typdesc)
			continue
		plotter.plot(datas[typid][0], datas[typid][i + 1], ".-", label=typdesc)
	plotter.xlabel('function evaluations')
	plotter.ylabel(f'{desc} mean error')
	plotter.yscale("log")
	if not legend_shown:
		plotter.legend(loc="upper right")
		legend_shown = True
	plotter.grid(True)

fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()
