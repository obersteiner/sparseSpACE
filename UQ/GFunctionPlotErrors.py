import numpy as np
import matplotlib.pyplot as plotter
import sys
import os

tmpdir = os.getenv("XDG_RUNTIME_DIR")
results_path = tmpdir + "/uqtestG.npy"
assert os.path.isfile(results_path)
solutions_data = list(np.load(results_path, allow_pickle=True))

typ_descs = ("full Gauß", "Trapez", "HighOrder", "BSpline", "Lagrange", "sparse Gauß")

datas = [[] for _ in typ_descs]
for v in solutions_data:
    (num_evals, typid, err_E, err_Var) = v
    typid = int(typid)
    if typid < 0 or typid >= len(typ_descs):
        continue
    if num_evals < 21:
        # No adaptive sparse grid results
        continue
    datas[typid].append((num_evals, err_E, err_Var))

# datas should be sorted here
for typid in range(len(typ_descs)):
    datas[typid] = np.array(datas[typid]).T

figure = plotter.figure(1, figsize=(11,11))
figure.canvas.set_window_title('G-Function Errors')

legend_shown = False
for i,desc in enumerate(("E", "Var")):
    plotter.subplot(2, 1, 1 + i)
    for typid, typdesc in enumerate(typ_descs):
        if len(datas[typid]) < 1:
            print("No points for", typdesc)
            continue
        plotter.plot(datas[typid][0], datas[typid][i + 1], ".-", label=typdesc)
    plotter.xlabel('function evaluations')
    plotter.ylabel(f"{desc} relative error")
    if desc == "E":
        # Change the height because perfect solutions sometimes appear
        plotter.ylim(10**-7, 10**-1)
    plotter.yscale("log")
    plotter.xscale("log")
    if not legend_shown:
        plotter.legend(loc="lower left")
        legend_shown = True
    plotter.grid(True)


fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()
