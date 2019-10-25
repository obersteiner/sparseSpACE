import numpy as np
import matplotlib.pyplot as plotter
import sys
import os

tmpdir = os.getenv("XDG_RUNTIME_DIR")
results_path = tmpdir + "/uqtestFUQ.npy"
assert os.path.isfile(results_path)
solutions_data = list(np.load(results_path, allow_pickle=True))

typ_descs = ("full grid Gauß", "Trapez", "HighOrder", "Lagrange", "sparse Gauß", "trans Trapez", "MC Halton")

datas = [[] for _ in typ_descs]
for v in solutions_data:
    (num_evals, typid, errs) = v
    typid = int(typid)
    if typid < 0 or typid >= len(typ_descs):
        continue
    if num_evals < 7:
        # No adaptive refinement points
        continue
    datas[typid].append((num_evals, *errs))

for typid in range(len(typ_descs)):
    datas[typid] = np.array(datas[typid]).T
err_descs = ("E absolute", "E relative", "Var absolute", "Var relative")

figure = plotter.figure(1, figsize=(11,11))
figure.canvas.set_window_title('FunctionUQ Errors')

legend_shown = False
for i,desc in enumerate(err_descs):
    if not i & 1:
        continue
    plotter.subplot(2, 1, 1 + (i-1)//2)
    for typid, typdesc in enumerate(typ_descs):
        if len(datas[typid]) < 1:
            print("No points for", typdesc)
            continue
        plotter.plot(datas[typid][0], datas[typid][i + 1], ".-", label=typdesc)
    plotter.xlabel('function evaluations')
    plotter.ylabel(f'{desc} error')
    plotter.yscale("log")
    plotter.xscale("log")
    if not legend_shown:
        plotter.legend(loc="lower left")
        legend_shown = True
    plotter.grid(True)

fileName = os.path.splitext(sys.argv[0])[0] + '.pdf'
plotter.savefig(fileName, format='pdf')

plotter.show()
