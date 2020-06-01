import random
import numpy as np
from matplotlib import pyplot as plt

def hat_function_non_symmetric(point, domain, x):
    """
    This method calculates the hat function value of the given coordinates with the given parameters
    :param : d-dimensional center point of the hat function
    :param : d-dimensional list of 2-dimensional tuples that describe the start and end values of the domain of the hat function
    :param : d-dimensional coordinates whose function value are to be calculated
    :return: value of the function at the coordinates given by x
    """
    assert len(point) == len(x) == len(domain)# sanity check
    result = 1.0
    for dim in range(len(x)):
        if x[dim] >= point[dim]:
            test = max(0.0, 1.0 - (abs(x[dim] - point[dim]) / abs(domain[dim][1] - point[dim])))
            result *= max(0.0, 1.0 - (abs(x[dim] - point[dim]) / abs(domain[dim][1] - point[dim])))
        elif x[dim] < point[dim]:
            test = max(0.0, 1.0 - (abs(x[dim] - point[dim]) / abs(domain[dim][0] - point[dim])))
            result *= max(0.0, 1.0 - (abs(x[dim] - point[dim]) / abs(domain[dim][0] - point[dim])))
    return result

def hat_function(ivec, lvec, x):
    """
    This method calculates the value of the hat function at the point x
    :param ivec: Index of the hat function
    :param lvec: Levelvector of the component grid
    :param x: datapoint
    :return: Value of the hat function at x
    """
    dim = len(lvec)
    result = 1.0
    for d in range(dim):
        result *= max((1 - abs(2 ** lvec[d] * x[d] - ivec[d])), 0)
    return result

dim = 1
levels = 3

comparison = []

testValues = [[random.random()] for x in range(10)]

print('~' * 10)
print('1D tests')
print('~' * 10)
#
# for lvl in range(1, levels):
#     for d in range(0, dim):
#         for index in range(1, 2 ** lvl):
#             p = [1 / 2 ** lvl * index for d in range(dim)]
#             dom = [(1 / 2 ** lvl * (index - 1), 1 / 2 ** lvl * (index + 1)) for x in range(dim)]
#             for t in testValues:
#                 if not hat_function_non_symmetric(p, dom, t) == hat_function([index for x in range(dim)], [lvl for x in range(dim)], t):
#                     print(t, ' : ', hat_function_non_symmetric(p, dom, t), ' : ', hat_function([index for x in range(dim)], [lvl for x in range(dim)], t))
#                 #comparison.append(hat_function_non_symmetric(p, dom, t) == hat_function(i, l, t))
# print('finished 1D tests')
#
# ######################
#
# print('~' * 10)
# print('2D tests')
# print('~' * 10)
#
# dim = 2
# levels = 3
#
# testValues = [[random.random(), random.random()] for x in range(100)]
#
# l = [1 for x in range(0, dim)]
# i = [1 for x in range(0, dim)]
#
# for level in range(2, levels):
#     for di in range(0, dim):
#         for index in range(1, 2 ** level):
#             point = [1 / 2 ** level * index for d in range(dim)]
#             domain = [(1 / 2 ** level * (index - 1), 1 / 2 ** level * (index + 1)) for x in range(dim)]
#             if domain[0] != domain[1]:
#                 print('dom error')
#             for t in testValues:
#                 if not hat_function_non_symmetric(point, domain, t) == hat_function([index for x in range(dim)], [level for x in range(dim)], t):
#                     print(t, ' : ', hat_function_non_symmetric(point, domain, t), ' : ', hat_function([index for x in range(dim)], [level for x in range(dim)], t))
#                 #comparison.append(hat_function_non_symmetric(p, dom, t) == hat_function(i, l, t))

print('finished 2D tests')


print('plotting of non-symmetric hat 1D')

x = [[x / 100.0] for x in range(0,100)]
print(x)
dom = [(0.0, 1.0)]
point = [0.25]
func = lambda x: [hat_function_non_symmetric(point, dom, e) for e in x]
print(func(list(x)))
plt.plot(list(x), func(list(x)))
plt.show()

#####################

print('plotting of symmetric hat 2D')
def plot_hat(ivec, lvec):
    func = lambda x: hat_function(ivec, lvec, x)

    X = [x / 100.0 for x in range(0,100)]
    Y = [y / 100.0 for y in range(0,100)]

    f_values = [func([x, y]) for x in X for y in Y]

    # Make data.
    X = np.arange(0.0, 1.0, 0.01)
    Y = np.arange(0.0, 1.0, 0.01)
    X, Y = np.meshgrid(X, Y)
    func = lambda x, y: hat_function(ivec, lvec, [x, y])
    func_vectorized = np.vectorize(func)
    Z = func_vectorized(X, Y)

    plt.figure(figsize=(20, 10))
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')

    # Plot the surface.
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='terrain', edgecolor='000', alpha=0.5)
    ax.set(xlabel='x1', ylabel='y2', zlabel='f(x1, x2)',
           title='symm hat')

    plt.show()

ivec = [1, 1]
lvec = [1, 1]
plot_hat(ivec, lvec)
#####################

print('plotting of non-symmetric hat 2D')
def plot_non_symm_hat(point, dom):
    func = lambda x: hat_function_non_symmetric(point, dom, x)

    X = [x / 100.0 for x in range(0,100)]
    Y = [y / 100.0 for y in range(0,100)]

    f_values = [func([x, y]) for x in X for y in Y]

    # Make data.
    X = np.arange(0.0, 1.0, 0.01)
    Y = np.arange(0.0, 1.0, 0.01)
    X, Y = np.meshgrid(X, Y)
    func = lambda x, y: hat_function_non_symmetric(point, dom, [x, y])
    func_vectorized = np.vectorize(func)
    Z = func_vectorized(X, Y)

    plt.figure(figsize=(20, 10))
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')

    # Plot the surface.
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='terrain', edgecolor='000', alpha=0.5)
    ax.set(xlabel='x1', ylabel='y2', zlabel='f(x1, x2)',
           title='non symm hat')

    plt.show()

dom = [(0.0, 1.0), (0.0, 1.0)]
point = [0.25, 0.25]
plot_non_symm_hat(point, dom)

#######################ü
from scipy.integrate import nquad

def calculate_L2_scalarproduct(point_i, domain_i, point_j, domain_j):
    """
    This method calculates the L2-scalarproduct of the two hat functions
    :param ivec: Index of the first hat function
    :param jvec: Index of the second hat function
    :param lvec: Levelvector of the component grid
    :return: L2-scalarproduct of the two hat functions plus the error of the calculation
    """
    if not (len(point_i) == len(point_j) == len(domain_i) == len(domain_j)):
        print('error')
    # check adjacency
    if all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(len(domain_i)))):
        dim = len(domain_i)
        f = lambda *x: (hat_function_non_symmetric(point_i, domain_i, [*x]) * hat_function_non_symmetric(point_j, domain_j, [*x]))
        start = [min(domain_i[d][0], domain_j[d][0]) for d in range(dim)]
        end = [max(domain_i[d][1], domain_j[d][1]) for d in range(dim)]
        if True:
            print("-" * 100)
            print("Calculating")
            print("Gridpoints: ", point_i, point_j)
            print("Domain: ", start, end)
        return nquad(f, [[start[d], end[d]] for d in range(dim)],
                     opts={"epsabs": 10 ** (-15), "epsrel": 1 ** (-15)})
    else:
        return (0, 0)

dom_1 = [(0.0, 1.0)]
point_1 = [0.25]
res = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
print('A calculate_L2_scalarproduct', res)
print('A calculate_L2_scalarproduct abs err', 1.0/3.0 - res[0])
print('A calculate_L2_scalarproduct rel err', (1.0/3.0 - res[0]) / (1.0 / 3.0))

dom_1 = [(0.0, 1.0), (0.0, 1.0)]
point_1 = [0.25, 0.25]
res = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
print('A 2D calculate_L2_scalarproduct', res)
# print('A 2D calculate_L2_scalarproduct abs err', (1.0/3.0)**2 - res[0])
# print('A 2D calculate_L2_scalarproduct rel err', ((1.0/3.0)**2 - res[0]) / (1.0 / 3.0)**2)


dom_1 = [(-1.0, 1.0)]
point_1 = [0.0]
res =  calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
print('B calculate_L2_scalarproduct', res)
print('B calculate_L2_scalarproduct abs err', 2.0/3.0 - res[0])
print('B calculate_L2_scalarproduct rel err', (2.0/3.0 - res[0]) / (2.0 / 3.0))

dom_1 = [(-1.0, 1.0), (-1.0, 1.0)]
point_1 = [0.0, 0.0]
res = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
print('B 2D calculate_L2_scalarproduct', res)
print('B 2D calculate_L2_scalarproduct abs err', (4.0/9.0) - res[0])
print('B 2D calculate_L2_scalarproduct rel err', ((4.0/9.0) - res[0]) / (2.0 / 3.0)**2)