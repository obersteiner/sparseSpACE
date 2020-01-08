from fenics import *

class PDE_Solver(object):
    @abc.abstractmethod
    def PDE_solve(self):
        pass

# Genaral 2D Poisson in UnitSquare domain
# Currently only FEniCS solver supported
class UnitSquare2DPoissonDirichletBC(PDE_Solver):
    '''
    Defines Poisson eqiation in the form alfa*u_xx + u_yy = f 
    evaluated using linear Lagrange elements over a unit square mesh with Drirchlet BC
        - alfa: float - a const term
        - f: String - C++ style expression (including cmath header file) for PDE's input
        - f_degree: int - degree of the input expression
                    e.g f='1 + x[0]*x[0] + 2*x[1]*x[1]’, degree=2
        - u_D: String - C++ style expression for Dirichlet boundary
        - u_D_degree: int
        - grid: tuple - # of nodes in the domain in x and y axis

        this class is an input to general PDE_Solver
    '''
    def __init__(self, f, f_degree, u_D, u_D_degree, grid, reference_solution=None):  
        # Create mesh and define function space
        self.mesh = UnitSquareMesh(grid[1],grid[0])
        V = FunctionSpace(self.mesh, 'P', 1)

        # Define boundary
        u_D = Expression(u_D, degree=u_D_degree)

        def boundary(x, on_boundary):
            return on_boundary

        self.bc = DirichletBC(V, u_D, boundary)

        # Define Variational Problem
        u= TrialFunction(V)
        v= TestFunction(V)
        f= Expression(f, f_degree)
        self.a= dot(grad(u), grad(v))*dx
        self.L= f*v*dx
        self.u = Function(V)
            
    def PDE_solve(self):
        solve(self.a == self.L, self.u, self.bc)
        return u

    def plotMesh(self):
        plot(self.mesh, title='Finite element mesh for grid: ')
    
    def plotResponse(u):
        plot(u, title='Finite element solution for grid: ')

    def getAnalyticSolution(self):
        return reference_solution

    def computeL2Error(self, u_e, u):
        return errornorm(u_e, u, 'L2')
    
    def getVertexValues(u)
        return u.compute_vertex_values()