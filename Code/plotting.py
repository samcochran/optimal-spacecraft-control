# Import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style, rcParams
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#import project code
from simulation import quiver_acceleration

style.use('seaborn')
rcParams['figure.figsize'] = 10, 10

def plot_solution(sol, title, ms=(1,1), show_quivers=False, show_speed=False, ax=None, lim=(-5,5), no_grid=True, savefile=None):
    """ Plots in 2d a solution to the 3-body problem. Note: z-coordinates are ignored

    Inputs:
        sol (ndarray): an (18 x N) array containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
        show_quivers (bool): an optional parameter (default=False) for whether or not to plot the acceleration vector field at t_N
        show_speed (int): show the initial and final speed of the 3rd body (useful for evaluating slingshot effects)
        ax (pyplot.axis) the axis to plot the solution on. If None (default), a new figure and axis will be created
        no_grid (bool): whether to turn the grid off or not using ax.grid() [Default: True]
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x and y) or
            a 4-entry tuple (xmin, xmax, ymin, ymax) setting different limits for each axis
        savefile (string) if not None, this plot will be saved to the path "../Plots/{savefile}" using plt.savefig
    """

    if ax == None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        show_plot = False

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
        u3x, u3y, a_x, a_y = get_acc_quivers(sol, ms, lim[:2], lim[2:])
        scale = max(np.max(np.abs(a_x[-1])), np.max(np.abs(a_y[-1])))
        ax.quiver(u3x, u3y, a_x[-1], a_y[-1], scale=scale, scale_units='xy', label='Acceleration field')

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
    ax.legend(fontsize=12)
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    elif len(lim) ==4:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        raise ValueError("lim must either have 2 entries or 4 entries!")

    if no_grid:
        ax.grid()

    if savefile is not None:
        plt.savefig(f"../Plots/{savefile}")
    if show_plot:
        plt.show()

def plot_nbody(sol, title, lim=(-5,5), plot_third=True, colors=None, ax=None, energies=None, no_grid=True, savefile=None):
    """ Plots in 2d a solution to the n-body problem. Note: z-coordinates are ignored

    Inputs:
        sol (ndarray): an (6n x M) array containing the xyz coordinates
            for the n bodies over a timespan {t_0, ..., t_M}
        title (string): the title for plot
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x and y) or
            a 4-entry tuple (xmin, xmax, ymin, ymax) setting different limits for each axis
        colors (list): a list of matplotlib colors to color each body by. If there are less colors than bodies, it will
            cycle through the list again. Default: None leaves the colors up to matplotlib
        ax (pyplot.axis) the axis to plot the solution on. If None (default), a new figure and axis will be created,
            and the plot will be shown with plt.show(). If the axis is supplied plt.show() will not be called, in case
            the user is plotting a bunch of subplots
        energies (ndarray): a (n x M) array containing the total energy (kinetic + potential) for each body in the system.
            Using this, text is added displaying the change in energy Delta-E given by E(t_M) - E(t_0).
            Default: None, no text will be displayed.
        no_grid (bool): whether to turn the grid off or not using ax.grid() [Default: True]
        savefile (string) if not None, this plot will be saved to the path "../Plots/{savefile}" using plt.savefig
    """
    if ax == None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        show_plot = False

    n = len(sol)//6

    for i in range(n):
        j = i*3
        if i == 2 and not plot_third:
            continue
        if colors is None:
            line, = ax.plot(sol[j, :], sol[j+1, :], label=f'Body {i+1}')
            ax.plot(sol[j, -1], sol[j+1, -1], color=line.get_color(), marker='o')
        else:
            color = colors[i % len(color)]
            ax.plot(sol[j, :], sol[j+1, :], color=color, label=f'Body {i+1}')
            ax.plot(sol[j, -1], sol[j+1, -1], color=color, marker='o')

    #energy text
    if energies is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        vtext = ""
        for i, energy in enumerate(energies):
            vtext += "$\Delta E_{} = {:.4f}$\n".format(i+1, energy[-1]-energy[0])
        vtext = vtext[:-1]
        ax.text(0.05, 0.25, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')
    ax.legend(loc="upper right", fontsize=12)
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    elif len(lim) ==4:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        raise ValueError("lim must either have 2 entries or 4 entries!")

    if no_grid:
        ax.grid()

    if savefile is not None:
        plt.savefig(f"../Plots/{savefile}")

    if show_plot:
        plt.show()


def get_acc_quivers(sol, ms, xlim, ylim, grid_size=25):
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
    u3x, u3y = np.meshgrid(np.linspace(*xlim,grid_size), np.linspace(*ylim,grid_size)) #(N,N)
    u3 = np.stack((u3x, u3y), axis=0) #(2,N,N)
    a = quiver_acceleration(sol[:7], u3, *ms) #sol[:7] == u12
    #log(1+a) acceleration values for better plotting
    scaled_a = np.log1p(np.log1p(np.abs(a)))
    #print(scaled_a.max(), scaled_a.min(), scaled_a.mean())
    a = np.sign(a)*scaled_a

    a_x, a_y = a
    return u3x, u3y, a_x, a_y

def animate_solution(sols, title, filename, skip=40, interval=30., ms=(1,1), lim=(-5,5), show_quivers=True, show_speed=True):
    """ Animates in 2d a solution to the 3-body problem. Note: z-coordinates are ignored

    Inputs:
        sols (ndarray or list of ndarrays): one array or a list of (18 x N) arrays (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
        filename (string): the name for the file the animation is saved to (NB: ".mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended (i.e. only every 40 xy values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many miliseconds to include
            between each frame
        ms (tuple): the masses of the first and second bodies (used to calculate the acceleration vector field)
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x and y) or
            a 4-entry tuple (xmin, xmax, ymin, ymax) setting different limits for each axis
        show_quivers (bool): an optional parameter (default=True) for whether or not to include the acceleration
            vector field in the animation
        show_speed (bool): whether or not to display a text box with the speed of the 3rd body, default: True
    """
    if (type(sols) == list): #if we have a list of solutions
        plot_multiple = (len(sols) > 1)
        show_speed = (not plot_multiple) and show_speed
        sol = sols[0]
    else:
        plot_multiple = False
        sol = sols
        sols = [sol]

    fig, ax = plt.subplots()

    #First body
    first, = ax.plot([], [], color='steelblue', ls=':', label='First Body')
    first_pt, = ax.plot(sol[0, 0], sol[1, 0], color='steelblue', marker='o')

    # Second body
    second, = ax.plot([], [], color='seagreen', ls=':', label='Second Body')
    second_pt, = ax.plot(sol[3, 0], sol[4, 0], color='seagreen', marker='o')

    # Third body/bodies
    if plot_multiple:
        thirds = []
        third_pts = []
        for solution in sols:
            if len(thirds) == 0:
                #Only label the first one
                third, = ax.plot([], [], color='indigo', ls=':', label='Third Body')
            else:
                third, = ax.plot([], [], color='indigo', ls=':')
            third_pt, = ax.plot(solution[6, 0], solution[7, 0], color='indigo', marker='o')

            thirds.append(third)
            third_pts.append(third_pt)
    else:
        third, = ax.plot([], [], color='indigo', ls=':', label='Third Body')
        third_pt, = ax.plot(sol[6, 0], sol[7, 0], color='indigo', marker='o')
        thirds = [third]
        third_pts = [third_pt]

    #axis limits
    if len(lim) == 2:
        xlim = lim
        ylim = lim
    elif len(lim) ==4:
        xlim = lim[:2]
        ylim = lim[2:]
    else:
        raise ValueError("lim must either have 2 entries or 4 entries!")

    #vector field
    if show_quivers:
        u3x, u3y, a_x, a_y = get_acc_quivers(sol, ms, xlim, ylim)
        scale = max(np.max(np.abs(a_x)), np.max(np.abs(a_y)))
        quiver = ax.quiver(u3x, u3y, a_x[0,:,:], a_y[0,:,:], alpha=0.5, scale=scale, scale_units='xy', label='Acceleration field')

    #speed text for satellite
    if show_speed:
        v0 = float(np.linalg.norm(sol[15:, 0]))
        vtext = "3rd body speed\n$v_0 = {:.4f}$\n$v_n = {:.4f}$".format(v0, v0)
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        speed_text = ax.text(0.05, 0.17, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Set plot parameters and labels
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=12, bbox_to_anchor=(1, 1.05))
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')

    #limit animation frames
    N = sol.shape[1]
    frames = N // skip
    offset = N % skip

    def update(i):
        j = i*skip+offset
        first.set_data(sol[0, :j+1], sol[1, :j+1])
        second.set_data(sol[3, :j+1], sol[4, :j+1])

        first_pt.set_data(sol[0, j], sol[1, j])
        second_pt.set_data(sol[3, j], sol[4, j])

        for i, solu in enumerate(sols):
            thirds[i].set_data(solu[6, :j+1], solu[7, :j+1])
            third_pts[i].set_data(solu[6, j], solu[7, j])

        returning = [first, first_pt, second, second_pt] + thirds + third_pts

        if show_quivers:
            quiver.set_UVC(a_x[j], a_y[j])
            returning.append(quiver)
        if show_speed:
            vn = float(np.linalg.norm(sol[15:, j]))
            vtext = "3rd body speed\n$v_0 = {:.4f}$\n$v_n = {:.4f}$".format(v0, vn)
            speed_text.set_text(vtext)
            returning.append(speed_text)

        return tuple(returning)

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}.mp4".format(filename))
    # plt.show()

def animate_nbody(sol, title, filename, skip=40, interval=30., lim=(-5,5), colors=None, energies=None):
    """ Animates in 2d a solution to the n-body problem. Note: z-coordinates are ignored

    Inputs:
        sol (ndarray): a (6n x M) array containing the xyz coordinates for the n bodies over a timespan {t_0, ..., t_M}
        title (string): the title for plot
        filename (string): the name for the file the animation is saved to (NB: ".mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended (i.e. only every 40 xy values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many milliseconds to include between each frame
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x and y) or
            a 4-entry tuple (xmin, xmax, ymin, ymax) setting different limits for each axis
        colors (list): a list of matplotlib colors to color each body by. If there are less colors than bodies, it will
            cycle through the list again. Default: None leaves the colors up to matplotlib
        energies (ndarray): a (n x M) array containing the total energy (kinetic + potential) for each body in the system.
            Using this, text is added displaying the change in energy Delta-E given by E(t_M) - E(t_0).
            Default: None, no text will be displayed.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    n = len(sol)//6

    #Set up lines
    paths, points = [], []
    for i in range(n):
        if colors is None:
            path, = ax.plot([], [], ls=':', label=f'Body {i+1}')
            point, = ax.plot(sol[3*i, 0], sol[3*i+1, 0], color=path.get_color(), marker='o')
        else:
            color = colors[i % len(colors)]
            path, = ax.plot([], [], color=color, ls=':', label='First Body')
            point, = ax.plot(sol[3*i, 0], sol[3*i+1, 0], color=color, marker='o')
        paths.append(path)
        points.append(point)

    #energy text
    if energies is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        text = ""
        for i, energy in enumerate(energies):
            text += "$\Delta E_{} = {:.4f}$\n".format(i+1, 0)
        text = text[:-1]
        energy_text = ax.text(0.05, 0.25, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    #axis limits
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    elif len(lim) ==4:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        raise ValueError("lim must either have 2 entries or 4 entries!")

    # Set plot parameters and labels
    ax.legend(loc="upper right", fontsize=12, bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')

    #limit animation frames
    N = sol.shape[1]
    frames = N // skip
    offset = N % skip

    def update(i):
        j = i*skip+offset
        for k in range(n):
            paths[k].set_data(sol[3*k, :j+1], sol[3*k+1, :j+1])
            points[k].set_data(sol[3*k, j], sol[3*k+1, j])

        returning = paths + points

        if energies is not None:
            text = ""
            for i, energy in enumerate(energies):
                text += "$\Delta E_{} = {:.4f}$\n".format(i+1, energy[j]-energy[0])
            text = text[:-1]
            energy_text.set_text(text)
            returning.append(energy_text)

        return tuple(returning)

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}.mp4".format(filename))
    plt.show()

def plot_sol3d(sol, title, lim=(-5,5), savefile=None):
    """ Plots in 3d a solution to the 3-body problem.

    Inputs:
        sol (ndarray): an (18 x N) array (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for plot
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x, y, and z) or
            a 6-entry tuple (xmin, xmax, ymin, ymax, zmin, zmax) setting different limits for each axis
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
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
    elif len(lim) == 6:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_zlim(lim[4], lim[5])
    else:
        raise ValueError("lim tuple must either have 2 elements or 6 elements!")
    if savefile is not None:
        plt.savefig(f"../Plots/{savefile}")
    plt.show()

def plot_nbody3d(sol, title, lim=(-5,5), colors=None, energies=None, savefile=None):
    """ Plots in 3d a solution to the n-body problem.

    Inputs:
        sol (ndarray): a (6n x M) array (but only the first 3n columns are needed) containing the xyz coordinates
            for the n bodies over a timespan {t_0, ..., t_M}
        title (string): the title for plot
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x, y, and z) or
            a 6-entry tuple (xmin, xmax, ymin, ymax, zmin, zmax) setting different limits for each axis
        colors (list): an optional list of colors to plot each body in.
        energies (ndarray): a (n x M) array containing the total energy (kinetic + potential) for each body in the system.
            Using this, text is added displaying the change in energy Delta-E given by E(t_M) - E(t_0).
            Default: None, no text will be displayed.
    """
    n = len(sol)//6

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n):
        j = i*3
        if colors is None:
            line, = ax.plot(sol[j, :], sol[j+1, :], sol[j+2, :], label=f'Body {i+1}')
            ax.plot(sol[j, -1], sol[j+1, -1], sol[j+2, -1], color=line.get_color(), marker='o')
        else:
            color = colors[i % len(colors)]
            ax.plot(sol[j, :], sol[j+1, :], sol[j+2, :], color=color, label=f'Body {i+1}')
            ax.plot(sol[j, -1], sol[j+1, -1], sol[j+2, -1], color=color, marker='o')

    #energy text
    if energies is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        vtext = ""
        for i, energy in enumerate(energies):
            vtext += "$\Delta E_{} = {:.4f}$\n".format(i+1, energy[-1]-energy[0])
        vtext = vtext[:-1]
        ax.text2D(0.05, 0.95, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
    elif len(lim) == 6:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_zlim(lim[4], lim[5])
    else:
        raise ValueError("lim tuple must either have 2 elements or 6 elements!")

    if savefile is not None:
        plt.savefig(f"../Plots/{savefile}")
    plt.show()

def animate_sol3d(sol, title, filename, skip=40, interval=30, lim=(-5,5)):
    """ Animates in 3d a solution to the 3-body problem.

    Inputs:
        sol (ndarray): an (18 x N) array (but only the first 9 columns are needed) containing the xyz coordinates
            for the 3 bodies over a timespan {t_0, ..., t_N}
        title (string): the title for the animation
        filename (string): the name for the file the animation is saved to (NB: "_3d.mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended. (i.e. only every 40 xyz values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many miliseconds to include between each frame
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x, y, and z) or
            a 6-entry tuple (xmin, xmax, ymin, ymax, zmin, zmax) setting different limits for each axis
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
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
    elif len(lim) == 6:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_zlim(lim[4], lim[5])
    else:
        raise ValueError("lim tuple must either have 2 elements or 6 elements!")

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

def animate_nbody3d(sol, title, filename, skip=40, interval=30, lim=(-5,5), colors=None, energies=None):
    """ Animates in 3d a solution to the n-body problem.

    Inputs:
        sol (ndarray): an (6n x M) array (but only the first 3n columns are needed) containing the xyz coordinates
            for the n bodies over a timespan {t_0, ..., t_M}
        title (string): the title for the animation
        filename (string): the name for the file the animation is saved to (NB: "_3d.mp4" is always appended to this)
        skip (int): the number of t values to skip for each frame of animation. At the current solution resolution,
            skipping 40 points is recommended. (i.e. only every 40 xyz values will be plotted)
        interval (float): an argument passed onto the FuncAnimation class for how many miliseconds to include between each frame
        lim (tuple): either a tuple with 2 entries for using the same (min, max) limits for each axis (x, y, and z) or
            a 6-entry tuple (xmin, xmax, ymin, ymax, zmin, zmax) setting different limits for each axis
        colors (list): an optional list of colors to plot each body in.
        energies (ndarray): a (n x M) array containing the total energy (kinetic + potential) for each body in the system.
            Using this, text is added displaying the change in energy Delta-E given by E(t_M) - E(t_0).
            Default: None, no text will be displayed.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = len(sol)//6

    paths = []
    points = []

    for i in range(n):
        j = i*3
        if colors is None:
            path, = ax.plot([], [], ls='--', label=f'Body {i+1}')
            point, = ax.plot(sol[j, 0], sol[j+1, 0], color=path.get_color(), marker='o')
        else:
            color = colors[i % len(colors)]
            path, = ax.plot([], [], ls='--', label=f'Body {i+1}')
            point, = ax.plot(sol[j, 0], sol[j+1, 0], marker='o')
        paths.append(path)
        points.append(point)

    # Set plot parameters and labels
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, bbox_to_anchor=(1, 0.5)) #loc="upper right",
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
    elif len(lim) == 6:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_zlim(lim[4], lim[5])
    else:
        raise ValueError("lim tuple must either have 2 elements or 6 elements!")

    #energy text
    if energies is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        vtext = ""
        for i, energy in enumerate(energies):
            vtext += "$\Delta E_{} = {:.4f}$\n".format(i+1, 0)
        vtext = vtext[:-1]
        energy_text = ax.text2D(0.05, 0.95, vtext, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    #limit animation frames
    M = sol.shape[1]
    frames = M // skip
    offset = M % skip

    def update(i):
        j = i*skip+offset

        for k in range(n):
            paths[k].set_data(sol[3*k, :j+1], sol[3*k+1, :j+1])
            paths[k].set_3d_properties(sol[3*k+2, :j+1])

            points[k].set_data(sol[3*k, j], sol[3*k+1, j])
            points[k].set_3d_properties(sol[3*k+2, j])

            returning = paths + points
            if energies is not None:
                vtext = ""
                for i, energy in enumerate(energies):
                    vtext += "$\Delta E_{} = {:.4f}$\n".format(i+1, energy[j]-energy[0])
                vtext = vtext[:-1]
                energy_text.set_text(vtext)
                returning.append(energy_text)

        return tuple(returning)

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}_3d.mp4".format(filename))
    plt.show()

def animate_control(sol, sol2, u, target, title, filename, skip=40, interval=30., lim=(-5,5), energies=None):

    # Concatenate/combine to plot 3 bodies, where the third body includeds control thrusts
    opt_tf = sol2.p[0]
    control_sol = sol2.y
    n_points = control_sol.shape[1]
    grid = np.linspace(0, opt_tf, n_points)
    orig_sol = sol.sol(grid)
    solution = np.vstack((orig_sol[:6, :], # Primary body positions
               control_sol[:2, :], np.zeros((1, n_points)), # Spacecraft positions
               orig_sol[9:15], # Primary body velocities
               control_sol[2:4, :], np.zeros((1, n_points)))) # Spacecraft velocities

    fig, ax = plt.subplots(figsize=(8, 8))

    n = len(solution)//6

    # Set up the stuff for the spacecraft
    paths, points = [], []
    path, = ax.plot([], [], ls='-', color='tomato', label='Scaled Thrust')
    point, = ax.plot(solution[3*(n-1), 0], solution[3*(n-1)+1, 0] , marker='o', label='Spacecraft')
    paths.append(path)
    points.append(point)

    #Set up lines for the primaries
    for i in range(n - 1):
        path, = ax.plot([], [], ls=':')
        point, = ax.plot(solution[3*i, 0], solution[3*i+1, 0], color=path.get_color(), marker='o', label=f'Body {i+1}')
        paths.append(path)
        points.append(point)

    # Plot the target
    ax.scatter(target[0], target[1], color='lightsalmon', label='Target')

    #energy text
    if energies is not None:
        props = dict(boxstyle='round', facecolor='white', alpha=1, zorder=2)
        text = ""
        for i, energy in enumerate(energies):
            text += "$\Delta E_{} = {:.4f}$\n".format(i+1, 0)
        text = text[:-1]
        energy_text = ax.text(0.05, 0.25, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    #axis limits
    if len(lim) == 2:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    elif len(lim) ==4:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        raise ValueError("lim must either have 2 entries or 4 entries!")

    # Set plot parameters and labels
    ax.legend(loc="upper right", fontsize=12, bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')

    #limit animation frames
    N = solution.shape[1]
    frames = N // skip
    offset = N % skip
    scale = np.max(u)*4

    def update(i):
        j = i*skip+offset
        thrust = u[:, j] / scale # note that this is scaled by scale
        paths[0].set_data([solution[3*(n-1), j], solution[3*(n-1), j] - thrust[0]], [solution[3*(n-1)+1, j], solution[3*(n-1)+1, j+1] - thrust[1]])
        points[0].set_data(solution[3*(n-1), j], solution[3*(n-1)+1, j])
        for k in range(n - 1):
            paths[k+1].set_data(solution[3*k, :j+1], solution[3*k+1, :j+1])
            points[k+1].set_data(solution[3*k, j], solution[3*k+1, j])

        returning = paths + points

        if energies is not None:
            text = ""
            for i, energy in enumerate(energies):
                text += "$\Delta E_{} = {:.4f}$\n".format(i+1, energy[j]-energy[0])
            text = text[:-1]
            energy_text.set_text(text)
            returning.append(energy_text)

        return tuple(returning)

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=interval)
    ani.save("../Animations/{}.mp4".format(filename))
    plt.show()
