#%% Euler-Maruyama method for eigenvalue SDE solutions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def dyson(dt=0.1,time=10,n=5,starting=np.linspace(start=0, stop=5, num=5)):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += 1/diff
            simulation[i+1,j] = simulation[i,j] + dt*mu + sqrtime*np.random.randn()
    return simulation

def deterministic_dyson(dt=0.1,time=10,n=5,starting=np.linspace(start=0, stop=5, num=5)):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += 1/diff
            simulation[i+1,j] = simulation[i,j] + dt*mu
    return simulation

def wishart(dt=0.1,time=10,n=5,starting=np.linspace(start=0, stop=5, num=5)):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += (simulation[i,k]+simulation[i,j])/diff
            simulation[i+1,j] = simulation[i,j] + dt*(mu+n) + sqrtime*np.random.randn()*np.sqrt(simulation[i,j])
    return simulation

def deterministic_wishart(dt=0.1,time=10,n=5,starting=np.linspace(start=0, stop=5, num=5)):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += (simulation[i,k]+simulation[i,j])/diff
            simulation[i+1,j] = simulation[i,j] + dt*(mu+n)
    return simulation

def jacobi(dt=0.1,time=10,n=5,starting=np.linspace(start=0.1, stop=0.9, num=5),n_1=5,n_2=5):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += (simulation[i,j]*(1-simulation[i,k]) + simulation[i,k]*(1-simulation[i,j]))/diff
            simulation[i+1,j] = simulation[i,j] + dt*(mu+n_2-(n_1+n_2)*simulation[i,j]) + 1.5*sqrtime*np.random.randn()*np.sqrt(simulation[i,j]*(1-simulation[i,j]))
    return simulation

def deterministic_jacobi(dt=0.1,time=10,n=5,starting=np.linspace(start=0.1, stop=0.9, num=5),n_1=5,n_2=5):
    num_points = int(time / dt)
    simulation = np.zeros((num_points,n))
    for l in range(n):
        simulation[0,l] = starting[l]
    sqrtime = np.sqrt(dt)
    for i in range(num_points-1):
        for j in range(n):
            mu = 0
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += (simulation[i,j]*(1-simulation[i,k]) + simulation[i,k]*(1-simulation[i,j]))/diff
            simulation[i+1,j] = simulation[i,j] + dt*(mu+n_2-(n_1+n_2)*simulation[i,j]) 
    return simulation

if __name__ == "__main__":
    # Style for matplotlib
    matplotlib.style.use("seaborn-v0_8")
    # plt.style.use('tableau-colorblind10')
    
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)
    # Code for LaTeX
    matplotlib.use("pgf")  # LaTeX
    plt.rcParams.update({   # LaTeX
    "pgf.texsystem": "pdflatex",  # LaTeX
    'font.family': 'serif',   # LaTeX
    'text.usetex': True,    # LaTeX
    'pgf.rcfonts': False,})   # LaTeX

    # Set random seed 
    np.random.seed(57)

    # Examples

    # Dyson and deterministic Dyson comparison
    plt.figure(figsize=(6,2.5))
    n = 9
    time_dyson = 20
    dt = 0.01
    time_mesh = np.linspace(start=0,stop=time_dyson,num=int(time_dyson/dt))
    starting = np.linspace(start=-3, stop=3, num=n)
    sims = dyson(dt=dt,time=time_dyson,n=n,starting=starting)
    det_sims = deterministic_dyson(dt=dt,time=time_dyson,n=n,starting=starting)
    for i in range(n):
        plt.plot(time_mesh,det_sims[:,i],linewidth=0.8)
    for i in range(n):
        plt.plot(time_mesh,sims[:,i],linewidth=1)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.savefig("img/dyson_comparison.pgf")

    # Examples of Dyson Brownian motion
    dys_fig, dys_ax = plt.subplots(2,2)
    dys_fig.set_size_inches(6.5,4)
    n_ex2 = 4
    dt_ex2 =0.01
    time_ex2 = 10
    starting_ex2 = np.array([-1.1,-1,2,2.1])
    
    # First experiment
    time_mesh_ex2 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex2))
    sims_ex2 = dyson(dt=dt_ex2,time=time_ex2,n=n_ex2,starting=starting_ex2)
    for i in range(n_ex2):
        dys_ax[0,0].plot(time_mesh_ex2,sims_ex2[:,i],linewidth=0.6)

    # Second experiment
    dt_ex3 = 0.01
    n_ex3 = 5
    starting_ex3 = np.linspace(start=-1,stop=1,num=n_ex3)
    time_mesh_ex3 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex3))
    sims_ex3 = dyson(dt=dt_ex3,time=time_ex2,n=n_ex3,starting=starting_ex3)
    for i in range(n_ex3):
        dys_ax[0,1].plot(time_mesh_ex3,sims_ex3[:,i],linewidth=0.6)

    # Third experiment
    dt_ex4 = 0.01
    n_ex4 = 7
    starting_ex4 = np.linspace(start=-2.5,stop=2.5,num=n_ex4)
    time_mesh_ex4 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex4))
    sims_ex4 = dyson(dt=dt_ex4,time=time_ex2,n=n_ex4,starting=starting_ex4)
    for i in range(n_ex4):
        dys_ax[1,0].plot(time_mesh_ex4,sims_ex4[:,i],linewidth=0.6)

    # Fourth experiment
    dt_ex5 = 0.001
    n_ex5 = 9
    starting_ex5 = np.linspace(start=-2.5,stop=2.5,num=n_ex5)
    time_mesh_ex5 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex5))
    sims_ex5 = dyson(dt=dt_ex5,time=time_ex2,n=n_ex5,starting=starting_ex5)
    for i in range(n_ex5):
        dys_ax[1,1].plot(time_mesh_ex5,sims_ex5[:,i],linewidth=0.6)
    # Figure parameters
    for ax in dys_ax.flat:
        ax.set(xlabel=r'time ($t$)', ylabel=r'Position')
    dys_fig.savefig("img/four_dysons.pgf")
    
    # deterministic Dyson
    # A distance that does not grow
    det_dys_fig, det_dys_ax = plt.subplots(2,2)
    det_dys_fig.set_size_inches(6.5,2.5)
    n_ex2 = 4
    dt_ex2 =0.01
    time_ex2 = 10
    starting_ex2 = np.array( [-1.1,-1,2,2.1] )
    time_mesh_ex2 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex2))
    sims_ex2 = deterministic_dyson(dt=dt_ex2,time=time_ex2,n=n_ex2,starting=starting_ex2)
    for i in range(n_ex2):
        det_dys_ax[0,0].plot(time_mesh_ex2,sims_ex2[:,i])
    # Second experiment
    dt_ex3 = 0.01
    n_ex3 = 5
    starting_ex3 = np.linspace(start=-1,stop=1,num=n_ex3)
    time_mesh_ex3 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex3))
    sims_ex3 = deterministic_dyson(dt=dt_ex3,time=time_ex2,n=n_ex3,starting=starting_ex3)
    for i in range(n_ex3):
        det_dys_ax[0,1].plot(time_mesh_ex3,sims_ex3[:,i])

    # Third experiment
    dt_ex4 = 0.01
    n_ex4 = 7
    starting_ex4 = np.linspace(start=-2.5,stop=2.5,num=n_ex4)
    time_mesh_ex4 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex4))
    sims_ex4 = deterministic_dyson(dt=dt_ex4,time=time_ex2,n=n_ex4,starting=starting_ex4)
    for i in range(n_ex4):
        det_dys_ax[1,0].plot(time_mesh_ex4,sims_ex4[:,i])

    # Fourth experiment
    dt_ex5 = 0.001
    n_ex5 = 9
    starting_ex5 = np.linspace(start=-2.5,stop=2.5,num=n_ex5)
    time_mesh_ex5 = np.linspace(start=0,stop=time_ex2,num=int(time_ex2/dt_ex5))
    sims_ex5 = deterministic_dyson(dt=dt_ex5,time=time_ex2,n=n_ex5,starting=starting_ex5)
    for i in range(n_ex5):
        det_dys_ax[1,1].plot(time_mesh_ex5,sims_ex5[:,i])
    # Figure parameters
    for ax in det_dys_ax.flat:
        ax.set(xlabel=r'time ($t$)', ylabel=r'Position')
    # for ax in det_dys_ax.flat:
    #     ax.label_outer()
    det_dys_fig.savefig("img/four_det_dysons.pgf")

    # Wishart process
    # # Wishart process
    plt.figure(figsize=(6,2.5))
    n_wis = 9
    dt_wis = 0.01
    time_wis = 2
    time_mesh_wis = np.linspace(start=0,stop=time_wis,num=int(time_wis/dt_wis))
    starting_wis = np.linspace(start=0.1, stop=20, num=n_wis)
    sims_wis = wishart(dt=dt_wis,time=2,n=n_wis,starting=starting_wis)
    sims_det_wis = deterministic_wishart(dt=dt_wis,time=2,n=n_wis,starting=starting_wis)
    for i in range(n_wis):
        plt.plot(time_mesh_wis,sims_wis[:,i],linewidth=1)
    for i in range(n_wis):
        plt.plot(time_mesh_wis,sims_det_wis[:,i],linewidth=1)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.savefig("img/wishart_comparison.pgf")

    # Deterministic Wishart process
    plt.figure(figsize=(6,2.5))
    sims_det_wis = deterministic_wishart(dt=0.01,time=2,n=n_wis,starting=starting_wis)
    for i in range(n_wis):
        plt.plot(time_mesh_wis,sims_det_wis[:,i])
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.savefig("img/deterministic_wishart.pgf")

    # Jacobi process
    plt.figure(figsize=(6,2.5))
    n_jacobi = 9
    dt_jacobi=0.000001
    time_jacobi=0.05
    starting_jacobi = np.linspace(start=0.3, stop=0.5, num=n_jacobi)
    time_mesh_jacobi = np.linspace(start=0,stop=time_jacobi,num=int(time_jacobi/dt_jacobi))
    sims_jacobi = jacobi(dt=dt_jacobi,time=time_jacobi,n=n_jacobi,starting=starting_jacobi,n_1=n_jacobi+1,n_2=n_jacobi+2)
    for i in range(n_jacobi):
        plt.plot(time_mesh_jacobi,sims_jacobi[:,i],linewidth=0.6)
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.savefig("img/jacobi.pdf")

    # Deterministic Jacobi
    plt.figure(figsize=(6,2.5))
    n_det_jacobi = 9
    dt_det_jacobi=0.001
    time_det_jacobi=0.05
    starting_det_jacobi = np.linspace(start=0.3, stop=0.5, num=n_jacobi)
    time_mesh_det_jacobi = np.linspace(start=0,stop=time_jacobi,num=int(time_det_jacobi/dt_det_jacobi))
    sims_det_jacobi = deterministic_jacobi(dt=dt_det_jacobi,time=time_det_jacobi,n=n_det_jacobi,starting=starting_det_jacobi,n_1=n_det_jacobi+1,n_2=n_det_jacobi+2)
    for i in range(n_det_jacobi):
        plt.plot(time_mesh_det_jacobi,sims_det_jacobi[:,i])
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.savefig("img/deterministic_jacobi.pgf")

    
    
# %%
