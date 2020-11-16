# Import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style, rcParams
import matplotlib.animation as animation
#might want 3D stuff later
#from mpl_toolkits.mplot3d import Axes3D

#import project code
from simulation import quiver_acceleration

def plot_solution(sol, title):
    # Plot the solutions, assuming we have
    fig, ax = plt.subplots()

    # First body
    first = ax.plot(sol[0, :], sol[1, :], color='steelblue', label='First Body')
    ax.plot(sol[0, -1], sol[1, -1], color='steelblue', marker='o')

    # Second body
    second = ax.plot(sol[3, :], sol[4, :], color='seagreen', label='Second Body')
    ax.plot(sol[3, -1], sol[4, -1], color='seagreen', marker='o')

    # Third body
    third = ax.plot(sol[6, :], sol[7, :], color='indigo', label='Third Body')
    ax.plot(sol[6, -1], sol[7, -1], color='indigo', marker='o')
    
    #quivers
    u3x, u3y, a_x, a_y = get_acc_quivers(sol)
    ax.quiver(u3x, u3y, a_x[-1], a_y[-1], label='Acceleration field')

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    ax.legend(fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plt.show()

def get_acc_quivers(sol, grid_size=25):
    """Get acceleration vector field on the 3rd body given the current solution (only using the posiitions of the 1st and 
    2nd body) over time. For the specs below, N is the grid_size parameter and T is the number of time values in sol
    
    Inputs:
        sol (ndarray (18, T)) only uses the first 6 parts of sol: xyz coords for the 1st and 2nd bodies over time
        grid_size (int): how many evenly dispersed points on the grid to evaluate over time
    Returns:
        u3x (ndarray (N, N)): the x meshgrid for plotting the quivers
        u3y (ndarray (N, N)): the y meshgrid for plotting the quivers
        a_x (ndarray (T, N, N)): the x acceleration at each point of the grid for all time
        a_y (ndarray (T, N, N)): the y acceleration at each point of the grid for all time
    """
    u3x, u3y = np.meshgrid(np.linspace(-4,4,grid_size), np.linspace(-4,4,grid_size)) #(N,N)
    u3 = np.stack((u3x, u3y), axis=0) #(2,N,N)
    a_x, a_y = quiver_acceleration(sol[:7], u3) #sol[:7] == u12
    return u3x, u3y, a_x, a_y
    
def animate_solution(sol, title, filename, skip=1, interval=30, show_quivers=True):
    fig, ax = plt.subplots()
    
    u3x, u3y, a_x, a_y = get_acc_quivers(sol)
    
    first, = ax.plot([], [], color='steelblue', ls='--', label='First Body')
    first_pt, = ax.plot(sol[0, 0], sol[1, 0], color='steelblue', marker='o')

    # Second body
    second, = ax.plot([], [], color='seagreen', ls='--', label='Second Body')
    second_pt, = ax.plot(sol[3, 0], sol[4, 0], color='seagreen', marker='o')

    # Third body
    third, = ax.plot([], [], color='indigo', ls='--', label='Third Body')
    third_pt, = ax.plot(sol[6, 0], sol[7, 0], color='indigo', marker='o')
    
    #vector field
    if show_quivers:
        quiver = ax.quiver(u3x, u3y, a_x[0,:,:], a_y[0,:,:], alpha=0.5, label='Acceleration field')
        
    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    ax.legend(loc="upper right", fontsize=12, bbox_to_anchor=(1, 0.5))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    
    #limit animation frames
    N = sol.shape[1]
    frames = N // skip
    offset = N % skip
    
    def update(i):
        j = i*skip+offset
        first.set_data(sol[0, :j+1], sol[1, :j+1])
        second.set_data(sol[3, :j+1], sol[4, :j+1])
        third.set_data(sol[6, :j+1], sol[7, :j+1])
        
        first_pt.set_data(sol[0, j], sol[1, j])
        second_pt.set_data(sol[3, j], sol[4, j])
        third_pt.set_data(sol[6, j], sol[7, j])
        
        if show_quivers:
            quiver.set_UVC(a_x[j], a_y[j])
            return first, first_pt, second, second_pt, third, third_pt, quiver
        else:
            return first, first_pt, second, second_pt, third, third_pt
    
    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}.mp4".format(filename))
    plt.show()