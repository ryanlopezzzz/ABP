import numpy as np

def get_speed_data(exp_data):
    """
    Computes the speed of particles
    """
    speed_data = np.sqrt(exp_data['vx']**2 + exp_data['vy']**2) #shape: [num of time snapshots, num of particles]
    return speed_data

def get_vel_mag_rel_var(exp_data):
    """
    Computes the relative variance of speed for particles averaged over time. Relative variance (X) = Var(X) / Mean(X)^2.
    """
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    speed_data = np.sqrt(vx_data**2 + vy_data**2)
    speed_var = np.var(speed_data, axis=1) #variance across particles
    speed_mean = np.average(speed_data, axis=1)
    speed_rel_var = speed_var / (speed_mean**2)
    return speed_rel_var