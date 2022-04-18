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

def get_velocity_vicsek_param(exp_data):
    #velocity vicsek_param = | \sum \hat{v}_i / N |
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    v_mag_data = np.sqrt(vx_data**2 + vy_data**2)
    
    #eliminate samples with zero velocity
    vx_data = vx_data[np.all(v_mag_data!=0, axis=1)]
    vy_data = vy_data[np.all(v_mag_data!=0, axis=1)]
    v_mag_data = v_mag_data[np.all(v_mag_data!=0, axis=1)]

    #compute order parameter
    vx_norm_data = vx_data / v_mag_data
    vy_norm_data = vy_data / v_mag_data
    sum_norm_vel = np.sqrt(np.mean(vx_norm_data, 1)**2 + np.mean(vy_norm_data, 1)**2) #shape: [num of time snapshots]
    velocity_vicsek_param = np.mean(sum_norm_vel)
    return velocity_vicsek_param