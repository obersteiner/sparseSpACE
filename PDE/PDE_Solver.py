from fenics import *
import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt

class PDE_Solver(object):
    
    @abstractmethod
    def solve(self):
        pass


class FEniCS_Solver(PDE_Solver):
    ''' Provides interface between general and FEniCSs solvers.'''

    def init(self):
        pass

    def define_mesh_from_points(self, coords:list):
        """ Defines mesh from point coordinates in each dimension (up to 3D)
            Parameters:
            - coords: colection of vectors specifying node coordinates along each axis 
                        e.g. [[0,0.5,1],[0,0.25,0.5,0.75,1]]
        """
        # Uniform mesh creation
        dim = len(coords)
        self.N = [len(l) for l in coords]
        mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
        self.mesh = mesh_classes[dim - 1](*(n-1 for n in self.N)) #takes number of intervals (N-1) as an argument

        # Mesh transformation
        points = list(zip(*[X.flatten() for X in np.meshgrid(*coords)]))
        for i,m in enumerate(self.mesh.coordinates()):
            m[0] = points[i][0]
            m[1] = points[i][1]


    def define_unit_hypercube_mesh(self, N:list):
        """ Defines mesh in a unit hypercube domain
            Parameters:
            - N: number of grid points in the domain in each dimension e.g. [N_x, N_y, N_z] 
        """
        mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
        dim = np.size(N)
        self.N = N
        self.mesh = mesh_classes[dim - 1](*(n-1 for n in N)) #takes number of intervals (N-1) as an argument


    def define_rectangle_mesh(self, a, b, N_x, N_y):
        """ Parameters:
            - a,b: point coordinates defining the domain e.g [0,0] and [1,1] for unit square domain
        """
        self.N = [N_x, N_y]
        self.mesh = RectangleMesh(Point(*a), Point(*b), N_x-1, N_y-1) #takes number of intervals (N-1) as an argument

        
    def solve(self):
        pass

    def plot_mesh(self):
        plot(self.mesh, title='Finite element mesh')



class Poisson(FEniCS_Solver):
    ''' Genaral Poisson Solver in unit hypercube domain in the form -Laplace(u) = f 
        Parameters:
        - f: C++ style expression (including cmath header file) for PDE's input e.g f='1 + x[0]*x[0] + 2*x[1]*x[1]'
        - f_degree: degree of the input expression e.g. f_degree=2
        - u_D: C++ style expression for the Dirichlet BC
        - u_D_degree: degree of the expression for boundary
        - reference_solution and rs_degree: Analtyical solution if it exists
    '''

    def __init__(self, f:str, f_degree:int, u_D:str, u_D_degree:int, reference_solution=None, rs_degree:int = None): 
        self.f = Expression(f, degree = f_degree) # PDE's rhs
        self.u_D = Expression(u_D, degree = u_D_degree) # expression for the boundary
        self.reference_solution = Expression(reference_solution, degree=rs_degree)
        self.instationary = False
        self.data_dim = 1 # scalar valued output
        

    def solve(self, el_type:str = 'P', el_deg:int = 1):
        ''' Solves PDE on a specified predefined mesh
            Parameters:
            - el_type: finite element type (by default Lagrange elements)
            - degree: piecewise polynomial degree (by default linear) 
             
            Returns: Nodal values as np.array in a row major order
        '''
        assert self.mesh != None, " First, define the mesh "
        set_log_level(30) # notifz onlz about warnings

        # Define function space
        V = FunctionSpace(self.mesh, el_type, el_deg)

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

        return self.u.compute_vertex_values(self.mesh).reshape(*self.N[::-1]) # reverse order due to matrix shape convention


    def plot_solution(self):
        cf = plot(self.u, title='Finite element solution')
        plt.colorbar(cf)

    def computeL2Error(self):
        assert  self.reference_solution != None, "No reference solution specified"
        return errornorm(self.reference_solution.compute_vertex_values(self.mesh), 
                            self.u.compute_vertex_values(self.mesh), 'L2')

    def computeMaxError(self):
        assert  self.reference_solution != None, "No reference solution specified"
        return np.max(np.abs(self.reference_solution.compute_vertex_values(self.mesh) - 
                                self.u.compute_vertex_values(self.mesh)))


class Gaussian_Hill(FEniCS_Solver):
    """ Defines u'= Laplace(u) + f  on a specified mesh with
            u = u_D             on the boundary
            u = u_0             at t = 0 (chosen as a Gaussian hill)
            u_D = f = 0
    """
    def __init__(self, dt=0.05, t_max=1):
        self.dt = dt                # time-step size
        self.t_max = t_max          # final time
        self.instationary = True
        self.data_dim = 1           # scalar valued output
        self.u = []

    def solve(self, el_type: str = 'P', el_deg: int = 1):
        ''' 
        Solves instationary diffusion equation (implicit Euler) on predefined mesh 
        Returns: nodal values at each timestep as an array of arrays
        '''

        assert self.mesh != None, " First, define the mesh "
        set_log_level(30) # notifz onlz about warnings
        num_steps = int(self.t_max/self.dt)
        result = np.zeros((num_steps+1,*self.N[::-1]))
        
        # Define function space
        V = FunctionSpace(self.mesh, el_type, el_deg)

        # Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary
        bc = DirichletBC(V, Constant(0), boundary)

        # Define initial value
        u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5) # Gaussian Hill
        u_n = interpolate(u_0, V)
        result[0] = u_n.compute_vertex_values(self.mesh).reshape(*self.N[::-1])

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)

        F = u*v*dx + self.dt*dot(grad(u), grad(v))*dx - (u_n + self.dt*f)*v*dx
        a, L = lhs(F), rhs(F)

        # Time-stepping
        u = Function(V)
        t = 0
        for n in range(num_steps):
            t += self.dt
            solve(a == L, u, bc)
            result[n+1] = u.compute_vertex_values(self.mesh).reshape(*self.N[::-1])
            u_n.assign(u)

        return result



class Navier_Stokes(FEniCS_Solver):
    """
    Incompressible Navier-Stokes equations for channel flow (Poisseuille) on the unit square 
    using Incremental Pressure Correction Scheme (IPCS).
        u' + u . nabla(u)) - div(sigma(u, p)) = f
        div(u) = 0

    """
    def __init__(self, dt=0.02, t_max=10, mu=1, rho=1):
        self.dt = dt                # time step size
        self.t_max = t_max          # final time
        self.mu = mu                # kinematic viscosity
        self.rho = rho              # density
        self.instationary = True
        self.data_dim = 2           # vector valued output

    def solve(self, print_error=False):
        """ Three step solution:
            1. Calculate velocity from momentum eq. using old pressure values
            2. Pressure is updated based on the newly computed velocities
            3. Velocity is corrected using new pressure field
        
            Returns: Instationary solution as a structure of arrays [[v_x], [v_y]] for each timestep
        """
        num_steps = int(self.t_max/self.dt) 
        result = np.zeros((num_steps+1,2,*self.N[::-1]))
        set_log_level(30) # notifz onlz about warnings
        
        # Define function spaces
        assert self.mesh != None, " First, define the mesh "
        V = VectorFunctionSpace(self.mesh, 'P', 2)
        Q = FunctionSpace(self.mesh, 'P', 1)

        # Define boundaries
        inflow  = 'near(x[0], 0)'
        outflow = 'near(x[0], 1)'
        walls   = 'near(x[1], 0) || near(x[1], 1)'

        # Define boundary conditions
        bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
        bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
        bcp_outflow = DirichletBC(Q, Constant(0), outflow)
        bcu = [bcu_noslip]
        bcp = [bcp_inflow, bcp_outflow]

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        # Define functions for solutions at previous and current time steps
        u_n = Function(V)
        u_  = Function(V)
        p_n = Function(Q)
        p_  = Function(Q)

        ic = interpolate(u_n, V)
        result[0]=ic.compute_vertex_values(self.mesh).reshape([2,*self.N[::-1]])

        # Define expressions used in variational forms
        U   = 0.5*(u_n + u)
        n   = FacetNormal(self.mesh)
        f   = Constant((0, 0))
        k   = Constant(self.dt)
        mu  = Constant(self.mu)
        rho = Constant(self.rho)

        # Define strain-rate tensor
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2*mu*epsilon(u) - p*Identity(len(u))

        # Define variational problem for step 1
        F1 = rho*dot((u - u_n) / k, v)*dx + \
            rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
        + inner(sigma(U, p_n), epsilon(v))*dx \
        + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
        - dot(f, v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

        # Assemble matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        # Apply boundary conditions to matrices
        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]

        # Time-stepping
        t = 0
        for n in range(num_steps):

            # Update current time
            t += self.dt

            # Step 1: Tentative velocity step
            b1 = assemble(L1)
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1)

            # Step 2: Pressure correction step
            b2 = assemble(L2)
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2)

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3)

            if print_error==True:
                u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
                u_e = interpolate(u_e, V)
                error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
                print('t = %.2f: error = %.3g' % (t, error))
                print('max u:', u_.vector().get_local().max())

            # Store results
            # result[n+1] = p_.compute_vertex_values(self.mesh).reshape(*self.N[::-1])
            result[n+1] = u_.compute_vertex_values(self.mesh).reshape([2,*self.N[::-1]])

            # Update previous solution
            u_n.assign(u_)
            p_n.assign(p_)

        return result

