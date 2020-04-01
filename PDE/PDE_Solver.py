from fenics import *
import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt

class PDE_Solver(object):
    ''' Interface class'''
    
    @abstractmethod
    def solve(self):
        pass


class FEniCS_Solver(PDE_Solver):
    ''' Provides interface between general and FEniCSs solvers
        - f: String - C++ style expression (including cmath header file) for PDE's input
        - f_degree: int - degree of the input expression
                    e.g f='1 + x[0]*x[0] + 2*x[1]*x[1]’, f_degree=2
        - u_D: String - C++ style expression for boundary
        - u_D_degree: int
    '''
    def __init__(self, f, f_degree, u_D, u_D_degree):
        self.f= Expression(f, degree=f_degree) # rhs expression
        self.u_D = Expression(u_D, degree=u_D_degree) # expression for the boundary

    def define_mesh_from_points(self, coords, el_type: str = 'P', degree: int = 1):
        """ Defines mesh from a point coordinates specified by grid
            Parameters:
                coords: colection of vectors specifying node coordinates along each axis
        """
        pass

    def define_unit_hypercube_mesh(self, N, el_type: str = 'P', degree: int = 1):
        """ N: #grid points in the doman for every dimension [x, y, z]
            el_type: by default Lagrange elements
            degree: element's degree (by default linear) 
        """
        mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
        dim = np.size(N)
        self.a = np.ones(dim)
        self.b = np.ones(dim)
        self.N = N
        self.mesh = mesh_classes[dim - 1](*(n-1 for n in N)) #takes number of intervals (N-1) as an argument
        self.V = FunctionSpace(self.mesh, el_type, degree)

    def define_rectangle_mesh(self, a, b, N_x, N_y, el_type: str = 'P', degree: int = 1):
        self.a = a
        self.b = b
        self.N = [N_x, N_y]
        self.mesh = RectangleMesh(Point(*a), Point(*b), N_x-1, N_y-1) #takes number of intervals (N-1) as an argument
        self.V = FunctionSpace(self.mesh, el_type, degree)

        
    def solve(self):
        pass

    def plot_mesh(self):
        plot(self.mesh, title='Finite element mesh')

    def computeL2Error(self, exact, approx):
        return errornorm(exact, approx, 'L2')
    
    def computeMaxError(self, exact, approx):
        return np.max(np.abs(exact - approx))

# Genaral Poisson problem in unit hypercube domain (1D, 2D or 3D)
class Poisson(FEniCS_Solver):
    ''' Defines Poisson equation in the form -Laplace(u) = f '''
    def __init__(self, f, f_degree, u_D, u_D_degree, reference_solution=None, rs_degree=None): 
        FEniCS_Solver.__init__(self, f, f_degree, u_D, u_D_degree) 
        self.reference_solution = Expression(reference_solution, degree=rs_degree)

    def solve(self):
        ''' Solves PDE on a specified predefined mesh -> returns nodal values as np.array '''
        assert self.mesh != None, " First, define the mesh "
        # Define boundary
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.V, self.u_D, boundary)

        # Define Variational Problem
        v= TestFunction(self.V)
        u= TrialFunction(self.V)
        a= dot(grad(u), grad(v))*dx
        L= self.f*v*dx

        # Solve
        self.u = Function(self.V)
        solve(a == L, self.u, bc)

        return self.u.compute_vertex_values(self.mesh).reshape(*self.N[::-1]) # reverse order due to matrix shape convention


    def plot_solution(self):
        plot(self.u, title='Finite element solution')

    # Possible only in the simple case of Poisson:
    # Not really necessary though because vertex values are computed with machine precision accuracy
    def computeL2Error(self):
        assert  self.reference_solution != None, "No reference solution specified"
        return errornorm(self.reference_solution.compute_vertex_values(self.mesh), 
                            self.u.compute_vertex_values(self.mesh), 'L2')

    def computeMaxError(self):
        assert  self.reference_solution != None, "No reference solution specified"
        return np.max(np.abs(self.reference_solution.compute_vertex_values(self.mesh) - 
                                self.u.compute_vertex_values(self.mesh)))


class GaussianHill(FEniCS_Solver):
    """ Defines u'= Laplace(u) + f  on a specified mesh with
            u = u_D             on the boundary
            u = u_0             at t = 0 (chosen as a Gaussian hill)
            u_D = f = 0
    """
    def __init__(self, dt=0.05, t_max=1):
        self.dt = dt
        self.t_max = t_max

    def solve(self):
        ''' 
        Solves instationary diffusion equation (implicit Euler) on predefined mesh 
        Paremeters:
            dt - time-step size
            t_max - end time
        Returns: nodal values at each timestep as an array of arrays'''

        assert self.mesh != None, " First, define the mesh "
        num_steps = int(self.t_max/self.dt)
        self.result = np.zeros((num_steps+1,*self.N[::-1]))

        # Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(self.V, Constant(0), boundary)

        # Define initial value
        u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5) # Gaussian Hill
        u_n = interpolate(u_0, self.V)
        self.result[0] = u_n.compute_vertex_values(self.mesh).reshape(*self.N[::-1])

        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        f = Constant(0)

        F = u*v*dx + self.dt*dot(grad(u), grad(v))*dx - (u_n + self.dt*f)*v*dx
        a, L = lhs(F), rhs(F)

        # Time-stepping
        u = Function(self.V)
        t = 0
        for n in range(num_steps):
            t += self.dt
            solve(a == L, u, bc)
            self.result[n+1] = u.compute_vertex_values(self.mesh).reshape(*self.N[::-1])
            u_n.assign(u)

        return self.result

    def plot_solution(self, t):
        assert t<=self.t_max, "t must be <= {}".format(self.t_max)
        coord = []
        for i, n in enumerate(self.N):
            coord.append(np.linspace(self.a[i],self.b[i],n))
        plt.contourf(*coord,self.result[int(t/self.dt)])
        plt.colorbar()
        plt.show()
        # else isinstance(t, Sequence[int]):
        #     pass # to be implemented