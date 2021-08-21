import numpy as np

def get_dir_dot_vel(exp_data):
    """
    Computes the average of $\vec{v}_i \dot \vec{n}_i$ and $\vec{v}_i \dot \vec{n}_i / ||\vec{v}_i||$
    """
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    
    Np = nx_data.shape[1] #number of particles
    
    dir_dot_vel = np.sum(nx_data*vx_data+ny_data*vy_data, axis=1) / Np
    dir_dot_vel_norm = np.sum((nx_data*vx_data+ny_data*vy_data) / np.sqrt(vx_data**2 + vy_data**2),axis=1) / Np
    
    return dir_dot_vel, dir_dot_vel_norm

def get_dir_cross_vel(exp_data):
    """
    Computes the average of $\vec{v}_i \cross \vec{n}_i$ and $\vec{v}_i \cross \vec{n}_i / ||\vec{v}_i||$
    """
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    
    Np = nx_data.shape[1] #number of particles
        
    dir_cross_vel = np.sum(nx_data*vy_data-ny_data*vx_data, axis=1) / Np
    dir_cross_vel_norm = np.sum((nx_data*vy_data-ny_data*vx_data) / np.sqrt(vx_data**2 + vy_data**2),axis=1) / Np
    
    return dir_cross_vel, dir_cross_vel_norm


def get_vel_mag_distr(exp_data):
    """
    Computes the velocity magnitude distribution at each time frame and averaged over time frames. To make a distribution,
    velocities in same interval are grouped together.
    """
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    
    v_mag_data = np.sqrt(vx_data**2 + vy_data**2)
    
    v_mag_data_avg = np.sum(v_mag_data, axis=1)
    
    #To Graph:
    #ax.hist(v_mag_data_avg, bins=n_bins)
    
    return v_mag_data_avg, v_mag_data
    
    

    