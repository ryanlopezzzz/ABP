import numpy as np

def get_flocking_factors(exp_data, v0):
    #vicsek_param = | \sum \vec{n}_i / N |
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    
    Np = nx_data.shape[1] #number of particles
    
    vicsek_param_nx = np.sum(nx_data, axis=1) / Np
    vicsek_param_ny = np.sum(ny_data, axis=1) / Np
    
    vicsek_param = vicsek_param_nx**2 + vicsek_param_ny**2
    
    #vel_param = | \sum \vec{v}_i / (N * v0) |
    
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    
    vel_param_nx = np.sum(vx_data, axis=1) / (Np * v0)
    vel_param_ny = np.sum(vy_data, axis=1) / (Np * v0)
    
    vel_param = vel_param_nx**2 + vel_param_ny**2
    
    return vicsek_param, vel_param