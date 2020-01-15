from fenics import *
import numpy as np

class PDE_Solver(object):
    ''' Interface class'''
    
    @abc.abstractmethod
    def solvePDE(self):
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

    def solvePDE(self):
        pass

# Genaral Poisson problem in unit hypercube domain (1D, 2D or 3D)
class Poisson(FEniCS_Solver):
    '''
    Defines Poisson equation in the form -Laplace(u) = f evaluated using linear Lagrange elements 
    of specified degree over a hypercube mesh with Drirchlet BC
    '''
    def __init__(self, f, f_degree, u_D, u_D_degree, reference_solution=None, rs_degree=None): 
        FEniCS_Solver.__init__(self, f, f_degree, u_D, u_D_degree) 
        self.reference_solution = Expression(reference_solution, degree=rs_degree)

    def solvePDE(self, grid, degree=1):
        ''' Solves PDE on a specified grid with Lagrange elements of specified degree
            - grid: list - # of cells in the doman for every dimension [x, y, z]
            - degree: int - degree of Lagrange elements
        '''
        # Create mesh and define function space
        def createUnitMesh(grid:list):
            mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
            d = len(grid)
            mesh = mesh_classes[d - 1](*grid)
            return mesh

        self.mesh = createUnitMesh(grid)
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
        self.u_vertex_values = self.u.compute_vertex_values(self.mesh)
        self.u_e_vertex_values = self.reference_solution.compute_vertex_values(self.mesh)
    

    def getVertexValues(self):
        return self.u_vertex_values, self.u_e_vertex_values

    def computeL2Error(self):
        return errornorm(self.u_e_vertex_values, self.u_vertex_values, 'L2')

    def computeMaxError(self):
        return np.max(np.abs(self.u_e_vertex_values - self.u_vertex_values))

    def plotMesh(self):
        plot(self.mesh, title='Finite element mesh')
    
    def plotSolution(self):
        plot(self.u, title='Finite element solution')
    
    