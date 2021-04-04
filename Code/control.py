import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp

def solve_optimal_euclidean(nbody_sol, masses, x0, v0, m0, xf, **kwargs):
    """Solve the optimal control problem for a spacecraft (see our PDF for more info) with a Euclidean thrust control.
    The guess given to the BVP solver is initialized using valid state and costate configurations using the assumption
    u(t) = 0 (there is no control), unless p12_guess_strat is changed, in which p1 and p2 are not valid.
    
    Inputs:
    -------
        nbody_sol (OdeSolution) the solution returned by solving an the IVP for the nbody problem with the spacecraft as
            the last body of negligable mass compared to the other bodies. This is used for the position of the other bodies
            and a guess at the optimal spacecraft flight. We assume this was solved by our code in simulation.py that was
            writted to work for 3 dimensions (and as such has 3d positions), however we only use the first 2, asssuming the
            3rd dimension component is always 0.
        masses (ndarray) shape (n-1,) the masses of the n-1 bodies (excluding the spacecraft) used to solve the IVP
        x0 (array-like) shape (2,) the xy initial position of the spacecraft
        v0 (array-like) shape (2,) the xy initial velocity of the spacecraft
        m0 (array-like) shape (1,) the initial mass of the spacecraft
        xf (array-like) shape (2,) the desired final position of the spacecraft (the cost functional is designed to
            make the solution get close to xf, however it is not forced to be at xf)
    
    Keyword Arguments (**kwargs) with default values
    ------------------------------------------------
    [Solver configuration]
        tf_guess (= 1, float) the guess for the optimal final time t_f (you should probably override this one). This should not
            exceed the final time nbody_sol was solved for.
        t_steps (= 100, int) the initial number of time steps for the guess at the optimal state
        p12_guess_strat (= "bolza", string) the guessing strategy for guessing the costates p_1 and p_2 using the
            assumption u(t) = 0. "random" randomly initializes the guess. "zeros" initializes them at 0 (which would be
            true if the spacecraft trajectory exactly hit the target). "bolza" initializes them as constants (valid if u=0) 
            based on the endpoint condition derived from the bolza cost: p_12(t_f) = -Dphi/Dy_12(t_f)
            
    [Control problem parameters]
        v_e (= 2, float) the thrust per unit of fuel |u(t)|
        alpha (= 1, float) the cost associated with acceleration
        beta (= 1, float) the cost associated with fuel usage
        delta (= 1, float) the cost associated with the distance from the final position and the target xf.
    
    [**kwargs]
        Finally, any remaining keyword arguments will be passed directly to scipy's solve_bvp method, allowing you to
        configure, for instance max_nodes, or the verbosity of the output.
        
    Returns:
    --------
        sol (OdeSolution) The solution to the optimal control problem returned by scipy's solve_bvp() method. The shape of
            sol.y should b (10,N) which contains the 5 states and 5 costates of the problem over N time steps. The optimal
            final time t_f is given by the solved parameter sol.p[0]
        u (ndarray) shape (2,N) The optimal control derived from the above solution, over the same time step. The optimal
            control is calculated from the costates p3 and p4.
    """
    y0 = np.concatenate([x0, v0, m0])
    n = len(masses)+1 #The number of bodies used by the
    
    #pop kwarg params if available, else use defaults
    #Solver configuration
    tf_guess = kwargs.pop("tf_guess", 1)
    t_steps = kwargs.pop("t_steps", 100)
    p12_guess_strat = kwargs.pop("p12_guess_strat", "bolza")
    
    #Optimal control parameters
    v_e = kwargs.pop("v_e", 2)
    alpha = kwargs.pop("alpha", 1)
    beta = kwargs.pop("beta", 1)
    delta = kwargs.pop("delta", 1)
    #vf = kwargs.pop("vf", None)
    
    def ode(t, y, tf):
        N = len(t)
        #Unpack y after adjusting dimensions for batch matrix operations
        #y shape (10,N) -> (N,10) ->(N, 10, 1)
        y = np.expand_dims(y.T,2)
        x = y[:, :2, :] #shape (N,2,1)
        v = y[:, 2:4, :] #shape (N,2,1)
        m = y[:, 4:5, :] #shape (N, 1,1)
        p = y[:, 5:, :] #shape (N, 5, 1)
        
        #Create "batch" identity matrix of shape (N,2,2)
        I = np.stack([np.eye(2) for i in range(N)])

        #u should have shape (N,2,1)
        #Equation from DH/Du = 0
        u = -1/(2*alpha + 2*beta*m/v_e) * p[:,2:4,:]
        
        #Find the acceleration and big_ugly
        grav_accel = np.zeros((N,2,1))
        big_ugly = np.zeros((N,2,2))
        
        #rs shape (6n, N) -> (N, 6n) -> (N, 6n, 1)
        rs = np.expand_dims(nbody_sol.sol(tf[0]*t).T,2)
        for i in range(n - 1):
            r = rs[:,3*i: 3*i+2,:] #shape (N, 2, 1)
            rx = r - x #shape (N, 2, 1)
            dist_rx = np.linalg.norm(rx, axis=1, keepdims=True) #dist_rx shape (N,1,1)
            #if np.isclose(dist_rx, 0).any():
            #    print("i",i, dist_rx.squeeze(), x.squeeze())
            grav_accel += masses[i] * (rx)/dist_rx**3
        
            #we can't use np.outer() here for the outer product because it will flatten the input arrays
            #We want to do a batch outer product that uses batch matrix multiplication of the shapes
            #(N,2,1) and (N,1,2) returns a matrix of shape (N,2,2)
            rx_outer = rx@np.transpose(rx, (0,2,1))
            big_ugly += masses[i] * (-I + 3*rx_outer/dist_rx**2)/dist_rx**3
        
        u_accel = v_e*u/m #Shape (N, 2, 1)
        accel = u_accel + grav_accel #accel should have shape (N, 2, 1)
        
        #costate_evo shape = (N,5,5)@(N,5,1) = (N,5,1)
        costate_evo = -np.block([
            [np.zeros((N,2,2)), I, np.zeros((N,2,1))],
            [big_ugly, np.zeros((N,2,2)), -u_accel/m],
            [np.zeros((N,1,5))]
        ])@p
        
        #Concatenate the arrays together on axis=1 and squeeze axis 2 (which is just size 1) to get
        #an array of shape (N,10). Finally we need to transpose this to match the input size (10, N)
        return tf[0]*np.concatenate([
            v, #x' = v
            accel,
            -np.linalg.norm(u, axis=1, keepdims=True),
            costate_evo
        ], axis=1).squeeze().T
    
    #Since there are 11 unknowns, we need 11 boundary conditions:
    #5 unknown state variables, 5 unknown costate, and
    #1 unknown parameter (the final time t_f)
    def bc(ya, yb, tf):
        p = yb[5:]
        mf = yb[4]
        
        #Solve for control at t_f
        #u should have shape (2,)
        #Equation from DH/Du = 0
        u = -1/(2*alpha + 2*beta*mf/v_e) * p[2:4]
        
        #Hamiltonian at final time, H(t_f) = p dot f - L
        #find the acceleration due to control and gravity at t_f
        control_accel = v_e*u/mf
        
        grav_accel = np.zeros(2)
        rs = nbody_sol.sol(tf[0])
        for i in range(n - 1):
            dist = rs[3*i:3*i+2] - yb[:2]
            dist3norm = np.linalg.norm(dist)**3
            grav_accel += (masses[i]*dist/dist3norm)
        
        u2 = np.sum(u*u)
        pf =  yb[5:].dot(np.concatenate([
            yb[2:4], #y_2 = x'
            control_accel + grav_accel,
            [-np.sqrt(u2)]
        ]))
        
        #Lagrangian at t_f
        L = 1 + (alpha*v_e/mf + beta) * u2
        
        return np.concatenate([
            ya[:5] - y0, #x(0) = 0, v(0) = x'(0) = v0, m(0) = m0
            #This condition comes from our Bolza cost functional
            #which places a cost on x(t_f)'s distance from our desired target
            yb[5:7] + delta*2*(yb[:2] - xf), #p(t_f) = - Dphi/Dy(t_f)
            yb[7:9], #x'(t_f) is free (we could do the condition above)
            [(yb[9]), #p5(t_f) = 0 since m(t_f) is free
            (pf - L)], #H(t_f) = p(t_f) dot f(x(t_f), u(t_f), th(t_f)) - L(t_f)
        ])
    
    #Initialize our final time guess and our guess for the state/costate evolution over time
    #We just assume that u(t) = 0 for all time as a good first guess.
    #This allows the solution to converge to a reasonably close target
    t = np.linspace(0, 1, t_steps)
    state_guess = nbody_sol.sol(tf_guess*t)
    
    #There are 3 options for guessing p1 and p2 that work reasonably well
    if p12_guess_strat == "random":
        p12_guess = np.random.random((2,t_steps))
    #p1 and p2 would be zero if we happened to end up exactly on the target in the initial guess
    elif p12_guess_strat == "zeros": 
        p12_guess = np.zeros((2, t_steps))
    #This causes solve_bvp to converge very quickly, and is consistent with the other states and costates
    elif p12_guess_strat == "bolza":
        #when the control u=0, p1 and p2 (the integrals of p3 and p4) are constant and given by
        #the Bolza cost endpoint condition p_12(t_f) = -Dphi/D_x(t_f) = -2delta(x(t_f) - x_f)
        p12 = (-delta*2*(state_guess[3*(n-1): 3*(n-1)+2, -1] - xf)).reshape(2,1)
        p12_guess = p12*np.ones((1, t_steps)) #Shapes: (2,1)*(1,t_steps) = (2,t_steps)
    else:
        raise ValueError("p12_guess_strat must be 'random', 'zeros', or 'bolza'!")
    
    guess = np.concatenate([
        state_guess[3*(n-1): 3*(n-1)+2, :], #Guess positions from nbody solution with no control
        state_guess[3*n + 3*(n-1): 3*n + 3*(n-1)+2, :], #Guess velocities from nbody solution with no control
        m0*np.ones((1,t_steps)), #Guess mass as not changing due to u = 0
        p12_guess,
        np.zeros((3,t_steps)) #p5 = 0 for all t. p3 and p4 are 0 when the control u = 0
    ], axis=0)
    
    sol = solve_bvp(ode, bc, t, guess, p=[tf_guess], **kwargs)
    
    #Find optimal control u(t) from solution and return both
    m = sol.y[4,:]
    p = sol.y[5:,:]
    u = -1/(2*alpha + 2*beta*m/v_e) * p[2:4, :]
    return sol, u