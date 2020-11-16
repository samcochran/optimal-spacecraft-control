# Import packages
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#import project code
from simulation import quiver_acceleration

def plot_solution(sol, title, show_quivers=False, show_speed=True, ax=None):
    """ Plots in 2d a solution to the 3-body problem. Note: z-coordinates are ignored

    Inputs:
        sol (ndarray): an (18 x N) array containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
        show_quivers (bool): an optional parameter (default=True) for whether or not to plot the acceleration vector field at t_N
        show_speed (int): show the initial and final speed of the 3rd body (useful for evaluating slingshot effects)
    """

    if ax == None:
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
    if show_quivers:
        u3x, u3y, a_x, a_y = get_acc_quivers(sol)
        ax.quiver(u3x, u3y, a_x[-1], a_y[-1], label='Acceleration field')

    #speed text for satellite
    if show_speed:
        v0 = float(np.linalg.norm(sol[15:, 0]))
        vf = float(np.linalg.norm(sol[15:,-1]))
        vtext = "3rd body speed\n$v_0 = {:.4f}$\n$v_f = {:.4f}$".format(v0, vf)
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        ax.text(0.05, 0.17, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    if ax == None:
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

def animate_solution(sol, title, filename, skip=1, interval=30., show_quivers=True, show_speed=True):
    """ Animates in 2d a solution to the 3-body problem. Note: z-coordinates are ignored

    Inputs:
        sol (ndarray): an (18 x N) array (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
        filename (string): the name for the file the animation is saved to (NB: ".mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended (i.e. only every 40 xy values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many miliseconds to include between each frame
        show_quivers (bool): an optional parameter (default=True) for whether or not to include the acceleration vector field
            in the animation
    """
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

    #speed text for satellite
    if show_speed:
        v0 = float(np.linalg.norm(sol[15:, 0]))
        vtext = "3rd body speed\n$v_0 = {:.4f}$\n$v_n = {:.4f}$".format(v0, v0)
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        speed_text = ax.text(0.05, 0.17, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

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
        if show_speed:
            vn = float(np.linalg.norm(sol[15:, j]))
            vtext = "3rd body speed\n$v_0 = {:.4f}$\n$v_n = {:.4f}$".format(v0, vn)
            speed_text.set_text(vtext)

        return first, first_pt, second, second_pt, third, third_pt, quiver, speed_text

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}.mp4".format(filename))
    plt.show()

def plot_sol3d(sol, title):
    """ Plots in 3d a solution to the 3-body problem.

    Inputs:
        sol (ndarray): an (18 x N) array (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # First body
    first = ax.plot(sol[0, :], sol[1, :], sol[2, :], color='steelblue', label='First Body')
    ax.plot(sol[0, -1], sol[1, -1], sol[2, -1], color='steelblue', marker='o')

    # Second body
    second = ax.plot(sol[3, :], sol[4, :], sol[5, :], color='seagreen', label='Second Body')
    ax.plot(sol[3, -1], sol[4, -1], sol[5, -1], color='seagreen', marker='o')

    # Third body
    third = ax.plot(sol[6, :], sol[7, :], sol[8, :], color='indigo', label='Third Body')
    ax.plot(sol[6, -1], sol[7, -1], sol[8, -1], color='indigo', marker='o')

    #quivers Not sure how to do them in 3d and not sure if they're useful
    #u3x, u3y, a_x, a_y = get_acc_quivers(sol)
    #ax.quiver(u3x, u3y, a_x[-1], a_y[-1], label='Acceleration field')

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    plt.show()

def animate_sol3d(sol, title, filename, skip=1, interval=30):
    """ Animates in 3d a solution to the 3-body problem.

    Inputs:
        sol (ndarray): an (18 x N) array (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for the animation
        filename (string): the name for the file the animation is saved to (NB: "_3d.mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended. (i.e. only every 40 xyz values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many miliseconds to include between each frame
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #u3x, u3y, a_x, a_y = get_acc_quivers(sol)

    first, = ax.plot([], [], color='steelblue', ls='--', label='First Body')
    first_pt, = ax.plot(sol[0, 0], sol[1, 0], color='steelblue', marker='o')

    # Second body
    second, = ax.plot([], [], color='seagreen', ls='--', label='Second Body')
    second_pt, = ax.plot(sol[3, 0], sol[4, 0], color='seagreen', marker='o')

    # Third body
    third, = ax.plot([], [], color='indigo', ls='--', label='Third Body')
    third_pt, = ax.plot(sol[6, 0], sol[7, 0], color='indigo', marker='o')

    #vector field
    #if show_quivers:
    #    quiver = ax.quiver(u3x, u3y, a_x[0,:,:], a_y[0,:,:], alpha=0.5, label='Acceleration field')

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, bbox_to_anchor=(1, 0.5)) #loc="upper right",
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    #limit animation frames
    N = sol.shape[1]
    frames = N // skip
    offset = N % skip

    def update(i):
        j = i*skip+offset

        first.set_data(sol[0, :j+1], sol[1, :j+1])
        first.set_3d_properties(sol[2, :j+1])

        second.set_data(sol[3, :j+1], sol[4, :j+1])
        second.set_3d_properties(sol[5, :j+1])

        third.set_data(sol[6, :j+1], sol[7, :j+1])
        third.set_3d_properties(sol[8, :j+1])

        first_pt.set_data(sol[0, j], sol[1, j])
        first_pt.set_3d_properties(sol[2, j])

        second_pt.set_data(sol[3, j], sol[4, j])
        second_pt.set_3d_properties(sol[5, j])

        third_pt.set_data(sol[6, j], sol[7, j])
        third_pt.set_3d_properties(sol[8, j])

        #if show_quivers:
        #    quiver.set_UVC(a_x[j], a_y[j])
        #    return first, first_pt, second, second_pt, third, third_pt, quiver
        #else:
        return first, first_pt, second, second_pt, third, third_pt

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}_3d.mp4".format(filename))
    plt.show()
