import numpy as np

def get_dir_cross_vel(exp_data):
    """
    Computes the average across particles of $\vec{v}_i \cross \vec{n}_i$
    """
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    Np = nx_data.shape[1] #number of particles
        
    dir_cross_vel = np.sum(nx_data*vy_data-ny_data*vx_data, axis=1) / Np
    
    return dir_cross_vel

def get_dir_cross_vel_norm(exp_data):
    """
    Computes the average across particles of $\vec{v}_i \cross \vec{n}_i / ||\vec{v}_i||$
    """
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    Np = nx_data.shape[1] #number of particles

    dir_cross_vel_norm = np.sum( (nx_data*vy_data-ny_data*vx_data)/np.sqrt(vx_data**2 + vy_data**2) , axis=1) / Np
    
    return dir_cross_vel_norm