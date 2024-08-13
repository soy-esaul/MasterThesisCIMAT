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
    # i index is the time discretization
    for i in range(num_points-1):
        # with j we iterate over every particle
        for j in range(n):
            mu = 0
            # with k we iterate over the remaining particles (k \neq j)
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
    # i index is the time discretization
    for i in range(num_points-1):
        # with j we iterate over every particle
        for j in range(n):
            mu = 0
            # with k we iterate over the remaining particles (k \neq j)
            for k in range(n):
                diff = simulation[i,j] - simulation[i,k]
                if diff != 0:
                    mu += (simulation[i,j]*(1-simulation[i,k]) + simulation[i,k]*(1-simulation[i,j]))/diff
            simulation[i+1,j] = simulation[i,j] + dt*(mu+n_2-(n_1+n_2)*simulation[i,j]) 
    return simulation

if __name__ == "__main__":
    # Style for matplotlib
    # matplotlib.style.use("seaborn-v0_8")
    plt.style.use('tableau-colorblind10')
    plt.figure(figsize=(7,4))
    
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)
    # Code for LaTeX
    # matplotlib.use("pgf")  # LaTeX
    # plt.rcParams.update({   # LaTeX
    # "pgf.texsystem": "pdflatex",  # LaTeX
    # 'font.family': 'serif',   # LaTeX
    # 'text.usetex': True,    # LaTeX
    # 'pgf.rcfonts': False,})   # LaTeX

    # Set random seed 
    np.random.seed(57)

    # Examples
    n = 9
    time_dyson = 20
    dt = 0.01
    time_mesh = np.linspace(start=0,stop=time_dyson,num=int(time_dyson/dt))
    starting = np.linspace(start=-3, stop=3, num=n)
    sims = dyson(dt=dt,time=time_dyson,n=n,starting=starting)
    det_sims = deterministic_dyson(dt=dt,time=time_dyson,n=n,starting=starting)
    for i in range(n):
        plt.plot(time_mesh,det_sims[:,i])
    for i in range(n):
        plt.plot(time_mesh,sims[:,i])
    plt.xlabel(r"Time ($t$)")
    plt.ylabel(r"Position in space ($\lambda_i$)")
    plt.show()
    
    # A distance does not grow
    # plt.figure(figsize=(7,4))
    # n_ex2 = 4
    # starting_ex2 = np.array( [-1.1,-1,2,2.1] )
    # sims_ex2 = deterministic_dyson(dt=0.01,time=10,n=n_ex2,starting=starting_ex2)
    # for i in range(n_ex2):
    #     plt.plot(sims_ex2[:,i])
    # plt.show()


    # # Wishart process
    # plt.figure(figsize=(7,4))
    # n_wis = 10
    # starting_wis = np.linspace(start=0.1, stop=20, num=n_wis)
    # sims_wis = wishart(dt=0.01,time=10,n=n_wis,starting=starting_wis)
    # for i in range(n_wis):
    #     plt.plot(sims_wis[:,i])
    # plt.show()

    # # Deterministic Wishart process
    # plt.figure(figsize=(7,4))
    # sims_det_wis = deterministic_wishart(dt=0.01,time=2,n=n_wis,starting=starting_wis)
    # for i in range(n_wis):
    #     plt.plot(sims_det_wis[:,i])
    # plt.show()

    # Jacobi process
    # plt.figure(figsize=(7,4))
    # sims_jacobi = jacobi(dt=0.000001,time=0.05,n=n_jacobi,starting=starting_jacobi,n_1=n_jacobi+1,n_2=n_jacobi+2)
    # for i in range(n_jacobi):
    #     plt.plot(sims_jacobi[:,i])
    # plt.show()

    # Deterministic Jacobi
    # plt.figure(figsize=(7,4))
    # n_jacobi = 9
    # starting_jacobi = np.linspace(start=0.3, stop=0.5, num=n_jacobi)
    # sims_det_jacobi = deterministic_jacobi(dt=0.001,time=0.05,n=n_jacobi,starting=starting_jacobi,n_1=n_jacobi+1,n_2=n_jacobi+2)
    # for i in range(n_jacobi):
    #     plt.plot(sims_det_jacobi[:,i])
    # plt.show()

    
    
# %%
