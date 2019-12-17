def poisson_model(x):
'''
    x: a vector of
    bc: string
'''
    
    mesh = UnitSquareMesh(x[1],x[0])
    V = FunctionSpace(mesh, 'P', 1)
    # Creating boundary
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    bc = DirichletBC(V, u_D, boundary)
    # Defining Functions
    u= TrialFunction(V)
    v= TestFunction(V)
    f= Constant(-6.0)
    a= dot(grad(u), grad(v))*dx
    L= Constant(-6.0)*v*dx
    u = Function(V)
        
    # Solving PDE
    solve(a == L, u, bc)

    return u


def operation()
    # Defining lmax and lmin
    lmax = (5,5)
    lmin = (1,1)
    # Preparing combination technique
    factory = ActiveSetFactory.ClassicDiagonalActiveSet(lmax, lmin, 0)
    activeSet = factory.getActiveSet()
    scheme = combinationSchemeArbitrary(activeSet)
    keys = scheme.dictOfScheme.keys()
    combiGrid = combineGrids(scheme)

    for key in keys:
        ## FEniCS code
        # calculating number of grid points
        s = 2**np.array(key)
        u = poisson_model()
            
        ## combination technique
        # creating grid
        grid = DummyGridArbitraryDim(key,tuple([True for __ in range(len(key))]))
        # getting data
        data = np.reshape(convert_dof(u.vector().get_local(),dof_to_vertex_map(V)),2**np.array(key)+1)
        # filling grid
        grid.fillData(f=data)
        # adding grid to combine Grids
        combiGrid.addGrid(grid)
    # combining grids        
    erg_combi = real(combiGrid.getCombination())

    ## plot
    sz = 12,12
    fig = figure(figsize=sz)
    title('Combi solution',fontsize=24)
    cmap = 'YlGnBu'
    list_x = np.linspace(0,1,erg_combi.shape[1])
    list_y = np.linspace(0,1,erg_combi.shape[0])
    plt.pcolor(list_x, list_y, erg_combi, cmap=cmap)
    plt.axis([list_x.min(), list_x.max(), list_y.min(), list_y.max()])
    plt.colorbar()
    display(fig)

    return list_x, list_y, erg_combi