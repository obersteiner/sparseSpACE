from fenics import *
import numpy as np
from abc import abstractmethod

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
        - u_D: String - C++ style expression for Dirichlet boundary
        - u_D_degree: int
    '''
    def __init__(self, f, f_degree, u_D, u_D_degree):
        self.f= Expression(f, degree=f_degree)
        self.u_D = Expression(u_D, degree=u_D_degree)

    def solve(self):
        pass

    def computeL2Error(self, exact, approx):
        return errornorm(exact, approx, 'L2')
    
    def computeMaxError(self, exact, approx):
        return np.max(np.abs(exact - approx))

# Genaral Poisson problem in unit hypercube domain (1D, 2D or 3D)
class Poisson(FEniCS_Solver):
    '''
    Defines Poisson equation in the form -Laplace(u) = f evaluated using linear Lagrange elements 
    of specified degree over a hypercube mesh with Drirchlet BC
    '''
    def __init__(self, f, f_degree, u_D, u_D_degree, reference_solution=None, rs_degree=None): 
        FEniCS_Solver.__init__(self, f, f_degree, u_D, u_D_degree) 
        self.reference_solution = Expression(reference_solution, degree=rs_degree)

    def solve(self, N, degree=1):
        ''' Solves PDE on a specified grid with Lagrange elements of specified degree
            - N: list - # of cells in the doman for every dimension [x, y, z]
            - degree: int - degree of Lagrange elements
        '''
        # Create mesh and define function space
        def createUnitMesh(N):
            mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
            d = np.size(N)
            mesh = mesh_classes[d - 1](*N)
            return mesh
        
        self.mesh = createUnitMesh(N)
        V = FunctionSpace(self.mesh, 'P', degree)

        # Define boundary
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, self.u_D, boundary)

        # Define Variational Problem
        v= TestFunction(V)
        u= TrialFunction(V)
        a= dot(grad(u), grad(v))*dx
        L= self.f*v*dx

        # Solve
        self.u = Function(V)
        solve(a == L, self.u, bc)

        # Compute vertex values
        self.u_vertex_values = self.u.compute_vertex_values(self.mesh).reshape(*N+1)
        self.u_e_vertex_values = self.reference_solution.compute_vertex_values(self.mesh).reshape(*N+1)

    def get_vertex_values(self):
        # Returns vertex values in order
        return self.u_vertex_values

    def plot_mesh(self):
        plot(self.mesh, title='Finite element mesh')
    
    def plot_solution(self):
        plot(self.u, title='Finite element solution')

    # Possible only in the simple case of Poisson:
    # Not really necessary though because vertex values are computed with machine precision accuracy
    def get_reference_vertex_values(self):
        return self.u_e_vertex_values

    def computeL2Error(self):
        return errornorm(self.u_e_vertex_values, self.u_vertex_values, 'L2')

    def computeMaxError(self):
        return np.max(np.abs(self.u_e_vertex_values - self.u_vertex_values))


    
    