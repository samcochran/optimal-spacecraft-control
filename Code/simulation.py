# Import packages
import numpy as np
from scipy.integrate import solve_ivp

def gravity_acceleration(t, x, m1=1, m2=1):
    """
    
    Parameters:
        x (ndarray, length 18) xyz coordinates of 3 bodies, followed by their velocities
    """
    # Extract coordinates
    v = x[9:]
    x1, x2, x3 = x[:3], x[3:6], x[6:9]
    
    # Get body distances
    sqdist12 = np.sum(np.square(x2-x1))
    sqdist13 = np.sum(np.square(x3-x1))
    sqdist23 = np.sum(np.square(x3-x2))
    
    # Construct the acceleration due to gravity
    a = np.zeros(9)
    
    a[:3] = m2*(x2-x1)/np.power(sqdist12, 1.5)
    a[3:6] = m1*(x1-x2)/np.power(sqdist12, 1.5)
    a[6:9] = m2*(x2-x3)/np.power(sqdist23, 1.5) + m1*(x1-x3)/np.power(sqdist13, 1.5)
    
    # Return the result
    return np.concatenate((v,a))
    
def quiver_acceleration(u12, u3, m1=1, m2=2):
    """This is similar to the method above except it specializes in calculating an acceleration vector field for the 3rd mass
    in the system where the positions of the 1st and 2nd mass have been precalculated and we investigate the acceleration 
    experienced by potential 3rd masses at all positions in a grid. We also assume that z=0 for everything in the system.
    
    Parameters:
        u12 (ndarray, (6,T)) xyz coordinates of the first 2 bodies (we ignore the z coordinates)
        u3 (ndarray, (2, N, N)) xy coordinate grid array of the 3rd body
        m1 (float) the mass of the first body
        m2 (float) the mass of the second body
    Returns:
        a_x (ndarray (2,T,N,N)) the acceleration (x,y) for each 3rd body point in the grid over all time
    """
    u1 = u12[:2,:].reshape(2, -1, 1, 1) # (2,T,1,1)
    u2 = u12[4:6, :].reshape(2, -1, 1, 1) # (2,T,1,1)
    d, N, _N = u3.shape #using d to make it easier to generalize to include z-coord
    u3 = u3.reshape(d, 1, N, N) #(2, 1, N, N)
    sqdist13 = np.sum(np.square(u3-u1)) # (2,1,N,N) - (2,T,1,1) = (2,T,N,N) using array broadcasting
    sqdist23 = np.sum(np.square(u3-u2)) # (2,1,N,N) - (2,T,1,1) = (2,T,N,N)
    a = m2*(u2-u3)/np.power(sqdist23, 1.5) + m1*(u1-u3)/np.power(sqdist13, 1.5) #(2,T,N,N)
    return a
    
def simulate_mechanics(ic, t_span, t_eval, mass_ratio=1):
    """
    Uses solve_ivp to simulate the time evolution of the system with given 
    initial conditions under gravity.
    
    Parameters:
        ic (ndarray, (18,)): initial conditions 
        t_span (tuple, 2): start and end of time interval 
        t_eval (ndarray): evaluation times for solution
        mass_ratio (float): ratio m2/m1 (m1=1 by default, then m2 = mass_ratio)
        
    Returns:
        sol (ndarray, (18, L)): an array where each column is the state of the 
            system at the given time step
    """
    # Construct a function for use in solve_ivp
    f = lambda t, y: gravity_acceleration(t, y, m1=1, m2=mass_ratio)
    
    # Numerically simulate
    sol = solve_ivp(fun=f, t_span=t_span, y0=ic, t_eval=t_eval)
    
    # Return the solution
    return sol.y

def gravity_acceleration_general(x, m):
    """
    
    Parameters:
        x (ndarray, length 6*n) xyz coordinates of 3 bodies (first 3*n entries), followed by their velocities (last 3*n entries)
        m (iterable, length n) masses of bodies.
        
    Returns:
        a (ndarray, length 6*n) derivative of x (velocities, followed by acceleration)
    """
    assert len(x)%6 == 0 
    n = len(x) // 6
    
    # Extract velocities
    v = x[3*n:]
    
    # Extract positions
    x = [ x[3*i:3*(i+1)] for i in range(n) ] 

    # Construct acceleration due to gravity
    a = np.concatenate(tuple([sum( m[j] * (x[j] - x[i])/np.power(np.sum((x[j]-x[i])**2), 1.5) for j in range(n) if j != i) for i in range(n)]))
    
    # Return the result
    return np.concatenate((v,a))

def simulate_mechanics_general(ic, t_span, t_eval, m):
    """
    Uses solve_ivp to simulate the time evolution of the system with given 
    initial conditions under gravity.
    
    Parameters:
        ic (ndarray, (6*n,)): initial conditions 
        t_span (tuple, 2): start and end of time interval 
        t_eval (ndarray): evaluation times for solution
        m (iterable): list of masses
        
    Returns:
        sol (ndarray, (6*n, L)): an array where each column is the state of the 
            system at the given time step
    """
    # Construct a function for use in solve_ivp
    f = lambda t, y: gravity_acceleration_general(y, m=m)
    
    # Numerically simulate
    sol = solve_ivp(fun=f, t_span=t_span, y0=ic, t_eval=t_eval)
    
    # Return the solution
    return sol.y

def massless_energy(ic, m):
    """
    Returns an array representing the massless energy of the ith body. 
    """
    n = len(ic)//6
    
    # Extract positions and velocities
    x = np.array([ ic[3*i:3*(i+1)] for i in range(n) ])
    v = np.array([ ic[3*(n+i):3*(n+i+1)] for i in range(n) ])
    print(x)
    print(v)
    
    # Get potential and kinetic energies
    k = 0.5*np.sum(v**2, axis=1)
    print(k)
    p = -0.5*np.array([sum( m[j]/np.sqrt(np.sum((x[i]-x[j])**2)) for j in range(n) if j != i) for i in range(n)])
    print(p)
    
    return k + p 

def total_energy(ic, m): return np.array(m) * massless_energy(ic)
    
    


    