import random
import numpy as np
from matplotlib import pyplot as plt


def get_neighbors(point, gridPointCoordsAsStripes, dim):
    """
    This method
    :param point: d-dimensional Sequence containing the coordinates of the grid point
    :param gridPointCoordsAsStripes:
    :return: d-dimenisional Sequence of 2-dimensional tuples containing the start and end of the function domain in each dimension
    """
    # check if the coordinate is on the boundary and if we have points on the boundary
    boundary_check = lambda x: False

    # create a tuple for each point whose elements are the coordinates that are within the domain
    neighbor_tuple = lambda n: tuple((n[d] for d in range(0, dim) if
                                      n[d] >= point_domain[d][0] and n[d] <= point_domain[d][1] and boundary_check(
                                          n[d])))
    all_points = list(get_cross_product(gridPointCoordsAsStripes))
    point_domain = get_hat_domain(point, gridPointCoordsAsStripes)
    # pick only tuples where both coordinates are within the domain
    neighbors = [neighbor_tuple(p) for p in all_points if len(neighbor_tuple(p)) == dim]
    return neighbors

def get_hat_domain(point, gridPointCoordsAsStripes, dim):
    """
    This method
    :param point: d-dimensional tuple containing the indices of the point
    :param xxx: Sequence of length d with the maximum level for each dimension
    :return: d-dimenisional Sequence of 2-dimensional tuples containing the start and end of the function domain in each dimension
    """
    # go through stripes and collect 2 coordinates with lowest distance to the point for each dimension
    domain = []
    for d in range(0, dim):
        upper = [coord for coord in gridPointCoordsAsStripes[d] if coord > point[d]]
        lower = [coord for coord in gridPointCoordsAsStripes[d] if coord < point[d]]
        element = (0 if not lower else max(lower), 1.0 if not upper else min(upper))
        domain.append(element)
    return domain

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
                     opts={"epsabs": 10 ** (-30), "epsrel": 1 ** (-30), "limit": 200})
    else:
        return (0, 0)


def calculate_R_value_analytically(point_i, domain_i, point_j, domain_j):
    """
    This method calculates the integral of the overlap between the two hat functions i and j, which are parameterised
    by point_i, domain_i and point_j and domain_j respectively.
    Parameters
    ----------
    point_i
    domain_i
    point_j
    domain_j

    Returns
    -------
    The integral of the overlap between the domains of i and j
    -------
    An illustration of what is being calculated:
    domain_j[d][upper] = d_j; domain_i[d][lower] = d_i; point_i[d] = p_i; point_j[d] = p_j
    For each dimension: (NOTE: comments are in italic, so the diagonals and the Pipe symbol "|" are a bit skewed)
    f_..(x)
    ^
    |        *        *
    |       /'\     / ' \
    |      / ' \   /  '  \
    |     /  '  \ /   '   \
    |    /   '   X    '    \
    |   /    '  / \   '     \
    |  /     ' /   \  '      \
    | /      '/     \ '       \
    |/       .        .        \
    |d_i----p_i------p_j------d_j----> x
    f_i1(x) = 1 - 1/(p_i-d_i) * (p_i - x), with x in [d_i, p_i)
    f_i2(x) = 1 - 1/(p_j-p_i) * (x - p_i), with x in (p_i, p_j]
    f_j1(x) = 1 - 1/(p_j-p_i) * (p_j - x), with x in [p_i, p_j)
    f_j2(x) = 1 - 1/(d_j-p_j) * (x - p_j), with x in [p_j, d_j]
    => f_j1(x) = 1 - f_i2(x) => For an overlap where point_i[d] != point_j[d], we only need to use f_i2(x) to integrate
    we calculate: int_{p_i,p_j} [f_i2(x)*(1 - f_i2(x))]
    => for an overlap with point_i[d] == point_j[d], f_i. and f_j. are the same.
    we calculate: (int_{d_lower, p}[f_1(x) * f_1(x)]) + (int_{p, d_upper} [f_2(x) * f_2(x)])
    """
    # check adjacency
    if not all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(len(domain_i)))):
        return 0.0
    res = 1.0
    for d in range(0, len(point_i)):
        if point_i[d] != point_j[d]:
            slope = 1.0 / abs(point_i[d] - point_j[d])
            # f_2(x) = 1 - slope * (x - min(point_i[d], point_j[d])) = c - slope * x
            t = 1.0 + slope * min(point_i[d], point_j[d])
            a = min(point_i[d], point_j[d]) # lower end of integral
            b = max(point_i[d], point_j[d]) # upper end of integral
            int_ab = (t - t**2) * (b - a) + (t * slope - 0.5 * slope) * (b**2 - a**2) + (slope**2)/3.0 * (a**3 - b**3)
            res *= int_ab
        else:
            width_low = abs(point_i[d] - point_j[d])
            slope_1 = 1.0 / abs(point_i[d] - domain_i[d][0])
            slope_2 = 1.0 / abs(domain_i[d][1] - point_i[d])

            a = domain_i[d][0]  # lower end of first integral
            b = point_i[d]  # upper end of first integral, lower end of second integral
            c = domain_i[d][1]  # upper end of second integral

            t_1 = 1.0 - slope_1 * domain_i[d][0]
            t_2 = 1.0 + slope_2 * domain_j[d][1]

            int_1 = t_1**2 * (b - a) + slope_1 * (b**2 - a**2) + (slope_1**2)/3.0 * (b**3 - a**3)
            int_2 = t_2**2 * (c - b) + slope_2 * (b**2 - c**2) + (slope_2**2)/3.0 * (c**3 - b**3)

            res *= (int_1 + int_2)
    return res


def calculate_R_value_analytically_alt(point_i, domain_i, point_j, domain_j):
    # check adjacency
    if not all((domain_i[d][0] <= point_j[d] and domain_i[d][1] >= point_j[d] for d in range(len(domain_i)))):
        return 0.0
    res = 1.0
    for d in range(0, len(point_i)):
        if point_i[d] != point_j[d]:
            m = 1.0 / abs(point_i[d] - point_j[d])  # slope
            # f_2(x) = 1 - slope * (x - min(point_i[d], point_j[d])) = c - slope * x
            a = min(point_i[d], point_j[d])  # lower end of integral
            b = max(point_i[d], point_j[d])  # upper end of integral
            p_i = point_i[d]
            p_j = point_j[d]

            int_upper = b + \
                        (m * p_i * b) - \
                        (m * p_i * b) + \
                        ((m ** 2) * (1.0 / 3.0) * (b ** 3)) - \
                        ((m ** 2) * 0.5 * (b ** 2)) + \
                        ((m ** 2) * 0.5 * (b ** 2)) - \
                        (p_i * p_j * (m ** 2) * b)
            int_lower = a + \
                        (m * p_i * a) - \
                        (m * p_i * a) + \
                        ((m ** 2) * (1.0 / 3.0) * (a ** 3)) - \
                        ((m ** 2) * 0.5 * (a ** 2)) + \
                        ((m ** 2) * 0.5 * (a ** 2)) - \
                        (p_i * p_j * (m ** 2) * a)

            res *= (int_upper - int_lower)
        else:
            m1 = 1.0 / (domain_i[d][0] - point_i[d])
            m2 = 1.0 / (domain_j[d][1] - point_j[d])

            a = domain_i[d][0]  # lower end of first integral
            b = point_i[d]  # upper end of first integral, lower end of second integral
            c = domain_i[d][1]  # upper end of second integral

            p = point_i[d]

            int_1_upper = b - \
                          (2 * m1 * (p * b)) - \
                          m1 * (b ** 2) + \
                          (m1 ** 2) * b * (p ** 2) - \
                          (m1 ** 2) * p * b + \
                          (m1 ** 2) * (1.0 / 3.0) * (b ** 3)
            int_1_lower = a - \
                          (2 * m1 * (p * a)) - \
                          m1 * (a ** 2) + \
                          (m1 ** 2) * a * (p ** 2) - \
                          (m1 ** 2) * p * a + \
                          (m1 ** 2) * (1.0 / 3.0) * (a ** 3)

            int_2_upper = c - \
                          (2 * m2 * (p * c)) - \
                          m2 * (c ** 2) + \
                          (m2 ** 2) * c * (c ** 2) - \
                          (m2 ** 2) * c * c + \
                          (m2 ** 2) * (1.0 / 3.0) * (c ** 3)
            int_2_lower = b - \
                          (2 * m2 * (p * b)) - \
                          m2 * (b ** 2) + \
                          (m2 ** 2) * b * (p ** 2) - \
                          (m2 ** 2) * p * b + \
                          (m2 ** 2) * (1.0 / 3.0) * (b ** 3)

            res *= ((int_1_upper - int_1_lower) + (int_2_upper - int_2_lower))
    return res


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


########
def get_domain_overlap_width(point_i, domain_i, point_j, domain_j):
    """
    This method calculates the width of the overlap between the domains of points i and j in each dimension.
    :param point_i:  d-dimensional sequence of coordinates
    :param domain_i: d=dimensional sequence of 2-element tuples with the start and end value of the domain
    :param point_j:  d-dimensional sequence of coordinates
    :param domain_j: d=dimensional sequence of 2-element tuples with the start and end value of the domain
    :return: d-dimensional sequence of that describe the width of the overlap between domains
    """

    assert len(point_i) == len(point_j) == len(domain_i) == len(domain_j)
    widths = []
    for d in range(0, len(point_i)):
        lower = max(domain_i[d][0], domain_j[d][0])
        upper = min(domain_i[d][1], domain_j[d][1])
        widths.append(abs(upper - lower))
    return widths
######################################################
########
######################################################
print('#'*100)
print('compare calculate_L2_scalarproduct for rotated points')
dom_1 = [(0.0, 0.25), (0.0, 1.0)]
point_1 = [0.125, 0.5]
dom_2 = [(0.0, 0.25), (0.0, 1.0)]
point_2 = [0.125, 0.5]
res_1 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct', res_1)
dom_1 = [(0.0, 1.0), (0.0, 0.25)]
point_1 = [0.5, 0.125]
dom_2 = [(1.0, 0.0), (0.0, 0.25)]
point_2 = [0.5, 0.125]
res_2 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct rotated', res_2)
print('C 2D calculate_L2_scalarproduct abs err', res_1[0] - res_2[0])
print('C 2D calculate_L2_scalarproduct rel err', (res_1[0] - res_2[0]) / res_1[0])
print('domain overlap: ', get_domain_overlap_width(point_1, dom_1, point_2, dom_2))

print('#'*100)
print('compare calculate_L2_scalarproduct for translated points')
dom_1 = [(0.0, 0.25), (0.0, 1.0)]
point_1 = [0.125, 0.5]
dom_2 = [(0.0, 0.25), (0.0, 1.0)]
point_2 = [0.125, 0.5]
res_1 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct', res_1)
dom_1 = [(0.125, 0.375), (0.0, 1.0)]
point_1 = [0.25, 0.5]
dom_2 = [(0.125, 0.375), (0.0, 1.0)]
point_2 = [0.25, 0.5]
res_2 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct rotated', res_2)
print('C 2D calculate_L2_scalarproduct abs err', res_1[0] - res_2[0])
print('C 2D calculate_L2_scalarproduct rel err', (res_1[0] - res_2[0]) / res_1[0])
print('domain overlap: ', get_domain_overlap_width(point_1, dom_1, point_2, dom_2))

print('#'*100)
print('compare calculate_L2_scalarproduct for translated and rotated points')
dom_1 = [(0.0, 0.25), (0.0, 1.0)]
point_1 = [0.125, 0.5]
dom_2 = [(0.0, 0.25), (0.0, 1.0)]
point_2 = [0.125, 0.5]
res_1 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct', res_1)
dom_1 = [(0.0, 1.0), (0.125, 0.375)]
point_1 = [0.5, 0.25]
dom_2 = [(0.0, 1.0), (0.125, 0.375)]
point_2 = [0.5, 0.25]
res_2 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct rotated', res_2)
print('C 2D calculate_L2_scalarproduct abs err', res_1[0] - res_2[0])
print('C 2D calculate_L2_scalarproduct rel err', (res_1[0] - res_2[0]) / res_1[0])
print('domain overlap: ', get_domain_overlap_width(point_1, dom_1, point_2, dom_2))


print('~'*100)
print('~'*100)
print('~'*100)

dom_1 = [(0.0, 0.5), (0.0, 0.5)]
point_1 = (0.25, 0.25)
dom_2 = [(0.25, 0.75), (0.0, 0.5)]
point_2 = (0.5, 0.25)
res_1 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('C 2D calculate_L2_scalarproduct', res_1)
dom_1 = [(0.0, 0.375), (0.0, 0.5)]
point_1 = (0.25, 0.25)
dom_2 = [(0.625, 1.0), (0.0, 0.5)]
point_2 = (0.75, 0.25)
res_2 = calculate_L2_scalarproduct(point_1, dom_1, point_2, dom_2)
print('D 2D calculate_L2_scalarproduct rotated', res_2)
print('D 2D calculate_L2_scalarproduct abs err', res_1[0] - res_2[0])
print('D 2D calculate_L2_scalarproduct rel err', (res_1[0] - res_2[0]) / res_1[0])
print('domain overlap: ', get_domain_overlap_width(point_1, dom_1, point_2, dom_2))

########################################################################################################################
print('\n\n\n')
print('~'*100)
print('~'*100)
print('~'*100)
print('Analytical and Numerical R value calculation comparison')


dom_1 = [(0.0, 0.5)]
point_1 = [0.25]
res_n = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
res_a = calculate_R_value_analytically_alt(point_1, dom_1, point_1, dom_1)
print('A calculate_L2_scalarproduct', res_n)
print('A calculate_R_value_analytically', res_a)
print('A calculate_L2_scalarproduct abs err', 1.0/3.0 - res_n[0])
print('A calculate_L2_scalarproduct rel err', (1.0/3.0 - res_n[0]) / (1.0 / 3.0))
print('A calculate_R_value_analytically abs err', 1.0/3.0 - res_a)
print('A calculate_R_value_analytically rel err', (1.0/3.0 - res_a) / (1.0 / 3.0))

dom_1 = [(0.0, 1.0), (0.0, 1.0)]
point_1 = [0.25, 0.25]
res_n = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
res_a = calculate_R_value_analytically_alt(point_1, dom_1, point_1, dom_1)
print('A 2D calculate_L2_scalarproduct', res_n)
print('A 2D calculate_R_value_analytically', res_a)
# print('A 2D calculate_L2_scalarproduct abs err', (1.0/3.0)**2 - res[0])
# print('A 2D calculate_L2_scalarproduct rel err', ((1.0/3.0)**2 - res[0]) / (1.0 / 3.0)**2)


dom_1 = [(-1.0, 1.0)]
point_1 = [0.0]
res_n = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
res_a = calculate_R_value_analytically_alt(point_1, dom_1, point_1, dom_1)
print('B calculate_L2_scalarproduct', res_n)
print('B calculate_R_value_analytically', res_a)
print('B calculate_L2_scalarproduct abs err', 2.0/3.0 - res_n[0])
print('B calculate_L2_scalarproduct rel err', (2.0/3.0 - res_n[0]) / (2.0 / 3.0))
print('B calculate_R_value_analytically abs err', 2.0/3.0 - res_a)
print('B calculate_R_value_analytically rel err', (2.0/3.0 - res_a) / (2.0 / 3.0))

dom_1 = [(-1.0, 1.0), (-1.0, 1.0)]
point_1 = [0.0, 0.0]
res_n = calculate_L2_scalarproduct(point_1, dom_1, point_1, dom_1)
res_a = calculate_R_value_analytically_alt(point_1, dom_1, point_1, dom_1)
print('B 2D calculate_L2_scalarproduct', res_n)
print('B 2D calculate_R_value_analytically', res_a)
print('B 2D calculate_L2_scalarproduct abs err', (4.0/9.0) - res_n[0])
print('B 2D calculate_L2_scalarproduct rel err', ((4.0/9.0) - res_n[0]) / (2.0 / 3.0)**2)
print('B 2D calculate_R_value_analytically abs err', (4.0/9.0) - res_a)
print('B 2D calculate_R_value_analytically rel err', ((4.0/9.0) - res_a) / (2.0 / 3.0)**2)

dom_1 = [(0.0, 1.0), (0.0, 1.0)]
point_1 = [0.25, 0.25]
print('alt and analytically:')
r_1 = calculate_R_value_analytically(point_1, dom_1, point_1, dom_1)
r_2 = calculate_R_value_analytically_alt(point_1, dom_1, point_1, dom_1)
print('ana: ', r_1)
print('alt: ', r_2)

def get_manhattan_distance(i, j, grid, dim):
    indices_i = [grid[x].index for x in range(0, dim)]
    indices_j = [grid[x].index for x in range(0, dim)]
    manhattan = [abs(indices_i[x] - indices_j[x]) for x in range(0, dim)]
    return manhattan

# check invariance of R values
#     Grid A     Grid B     level                            Grid C
#------------------------------------------------------------------------------------------------
#                           4       .   4   .   4   .   4   .   4   .   4   4   4   4   4   .   #
#   .   .   .       .       3       .   3   .   3   .   3   .   3   .   3   3   3   3   3   .   #
#   .   .   .       .       4       .   4   .   4   .   4   .   4   .   4   4   4   4   4   .   #
#   .   .   .       .       2       .   2   .   2   .   2   .   2   .   2   2   2   2   2   .   #
#   .   .   .       .       4       .   4   .   4   .   4   .   4   .   4   4   4   4   4   .   #
#   .   .   .       .       3       .   3   .   3   .   3   .   3   .   3   3   3   3   3   .   #
#                           4       .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   #
#   .   .   .       .       1       .   3   .   2   .   3   .   1   .   3   4   2   4   3   .   #
#                           4       .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   #
#   .   .   .       .       3       .   3   .   3   .   3   .   3   .   3   3   3   3   3   .   #
#                           4       .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   #
#   .   .   .       .       2       .   2   .   2   .   2   .   2   .   2   2   2   2   2   .   #
#                           4       .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   #
#   .   .   .       .       3       .   3   .   3   .   3   .   3   .   3   3   3   3   3   .   #
#                           4       .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   #
#------------------------------------------------------------------------------------------------
#   2   1   2       1               4   3   4   2   4   3   4   1   4   3   4   2   4   3   4
# dim = 2
# grid_A = [[0.25, 0.5, 0.75], [0.125, 0.25, 0.375, 0.5, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]] # max levels 2,4
# grid_B = [[0.5], [0.125, 0.25, 0.375, 0.5, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]]# max levels 1,4
# grid_C = [[0.125, 0.25, 0.375, 0.5, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375], [0.125, 0.25, 0.375, 0.5, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]] # max levels 4,4
# levels_A = [[2, 1, 2], [3, 2, 3, 1, 3, 4, 2, 4, 3, 4]]
# levels_B = [[1], [[3, 2, 3, 1, 3, 4, 2, 4, 3, 4]]]
# levels_C = [[3, 2, 3, 1, 3, 4, 2, 4, 3, 4], [3, 2, 3, 1, 3, 4, 2, 4, 3, 4]]
#
# point_list_A = [x for x in list(get_cross_product(grid_A)) if 0.0 not in x and 1.0 not in x]
# point_list_B = [x for x in list(get_cross_product(grid_B)) if 0.0 not in x and 1.0 not in x]
# point_list_C = [x for x in list(get_cross_product(grid_C)) if 0.0 not in x and 1.0 not in x]
#
# R_old = {}
#
# R_A = np.zeros(len(grid_A[0]) + len(grid_A[1]), len(grid_A[0]) + len(grid_A[1]))
# R_B = np.zeros(len(grid_B[0]) + len(grid_B[1]), len(grid_B[0]) + len(grid_B[1]))
# R_C = np.zeros(len(grid_C[0]) + len(grid_C[1]), len(grid_C[0]) + len(grid_C[1]))
#
# for i in range(len(point_list_A)):
#     for j in range(i, len(point_list_A)):
#         point_i = point_list_A[i]
#         domain_i = get_hat_domain(point_i, grid_A, dim)
#         levels_i = [levels_A.index(point_i[0]), levels_A.index(point_i[1])]
#
#         point_j = point_list_A[j]
#         levels_j = [levels_A.index(point_j[0]), levels_A.index(point_j[1])]
#         domain_j = get_hat_domain(point_j, grid_A, dim)
#
#         # check if the two points overlap
#         if sum([True for x in get_manhattan_distance(point_i, point_j, grid_A, dim) if x < 2]) is dim:
#             res = calculate_L2_scalarproduct(point_i, domain_i, point_j, domain_j)
#         else:
#             res = 0.0
#
#             R_A[i][j] = res
#             R_A[j][i] = res
#
# for i in range(len(point_list_B)):
#     for j in range(i, len(point_list_B)):
#         point_i = point_list_B[i]
#         levels_i = [levels_B.index(point_i[0]), levels_B.index(point_i[1])]
#         domain_i = get_hat_domain(point_i, grid_B, dim)
#
#         point_j = point_list_B[j]
#         levels_j = [levels_B.index(point_j[0]), levels_B.index(point_j[1])]
#         domain_j = get_hat_domain(point_j, grid_B, dim)
#
#         # check if the two points overlap
#         if sum([True for x in get_manhattan_distance(point_i, point_j, grid_B, dim) if x < 2]) is dim:
#             res = calculate_L2_scalarproduct(point_i, domain_i, point_j, domain_j)
#         else:
#             res = 0.0
#
#         R_B[i][j] = res
#         R_B[j][i] = res
#
# for i in range(len(point_list_C)):
#     for j in range(i, len(point_list_C)):
#         point_i = point_list_C[i]
#         levels_i = [levels_C.index(point_i[0]), levels_C.index(point_i[1])]
#         domain_i = get_hat_domain(point_i, grid_C, dim)
#
#         point_j = point_list_C[j]
#         levels_j = [levels_C.index(point_j[0]), levels_C.index(point_j[1])]
#         domain_j = get_hat_domain(point_j, grid_C, dim)
#
#         # check if the two points overlap
#         if sum([True for x in get_manhattan_distance(point_i, point_j, grid_C, dim) if x < 2]) is dim:
#             # check if we calculated this overlap before
#
#             res = calculate_L2_scalarproduct(point_i, domain_i, point_j, domain_j)
#         else:
#             res = 0.0
#
#         R_C[i][j] = res
#         R_C[j][i] = res
