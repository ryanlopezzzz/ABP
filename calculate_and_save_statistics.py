"""
Used to compute and save statistics for different simulations concurrently
"""

import os
import numpy as np
import json
import read_data as rd #reads snapshot text data
from collections import OrderedDict
import argparse
from Utils.graphing_helpers import edges_from_centers_linear
from Analysis_Calculations.msd import get_msd
import directories
import multiprocessing
import pickle
from sys import exit
import time

def compute_director_vicsek_parameter(exp_data: dict) -> float:
    # director vicsek param = | \sum \vec{n}_i / N | averaged over particles for each time step
    nx_data = exp_data['nx'] #shape: [num of time snapshots, num of particles]
    ny_data = exp_data['ny']
    Np = nx_data.shape[1] #number of particles

    director_vicsek_param_nx = np.sum(nx_data, axis=1) / Np
    director_vicsek_param_ny = np.sum(ny_data, axis=1) / Np
    director_vicsek_param = director_vicsek_param_nx**2 + director_vicsek_param_ny**2
    return director_vicsek_param

def compute_velocity_vicsek_parameter(exp_data: dict, v0: float) -> float:
    #velocity vicsek param = | \sum \vec{v}_i / (N * v0) |
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    Np = vx_data.shape[1] #number of particles
    
    velocity_vicsek_param_nx = np.sum(vx_data, axis=1) / (Np * v0)
    velocity_vicsek_param_ny = np.sum(vy_data, axis=1) / (Np * v0)
    velocity_vicsek_param = velocity_vicsek_param_nx**2 + velocity_vicsek_param_ny**2
    return velocity_vicsek_param

def compute_velocity_magnitude_histogram(exp_data: dict) -> tuple:
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    vy_data = exp_data['vy']
    v_mag_data = np.sqrt(vx_data**2 + vy_data**2)
    histogram_values, bin_edges = np.histogram(v_mag_data, bins=20, density=True)
    return histogram_values, bin_edges

def compute_velocity_x_histogram(exp_data: dict) -> tuple:
    vx_data = exp_data['vx'] #shape: [num of time snapshots, num of particles]
    histogram_values, bin_edges = np.histogram(vx_data, bins=20, density=True)
    return histogram_values, bin_edges

def compute_velocity_y_histogram(exp_data: dict) -> tuple:
    vy_data = exp_data['vy'] #shape: [num of time snapshots, num of particles]
    histogram_values, bin_edges = np.histogram(vy_data, bins=20, density=True)
    return histogram_values, bin_edges

def compute_local_packing_fractions(exp_data: dict, num_bins_along_dim: int, L: float, particle_area: float) -> tuple:
    """
    Calculate local packing fraction by splitting simulation box into bins and counting number of paricles in each bin.
    Returns local packing fraction values at each time / space coordinate, does not give historgram.

    :param num_bins_along_dim: Number of bins along x and y dimensions, so total bins is squared of this value
    :param L: Length of the box
    :param particle_area: Area of a single particle (polydispersity not supported)

    :return local_packing_fraction: First index indicates time snapshots, the next index indicates which y box it is,
        the third index indicates which x box it is
    """
    exp_data_x = exp_data['x'] #shape: [num of time snapshots, num of particles]
    exp_data_y = exp_data['y']
    num_time_snapshots = exp_data_x.shape[0]
    box_range = np.array([[-L/2, L/2], [-L/2, L/2]]) #simulation box range
    bin_area = (L / num_bins_along_dim)**2

    local_packing_fraction = np.zeros((num_time_snapshots, num_bins_along_dim, num_bins_along_dim))
    for snapshot in range(num_time_snapshots):
        histogram2d, x_edges, y_edges = np.histogram2d(exp_data_x[snapshot], exp_data_y[snapshot], bins=num_bins_along_dim, range=box_range)
        histogram2d = histogram2d.T #transpose switches from (x,y) to (y,x) convention. NOTE: Used for plotting.
        local_packing_fraction_snapshot = (particle_area/bin_area)*histogram2d
        local_packing_fraction[snapshot] = local_packing_fraction_snapshot
    return local_packing_fraction, x_edges, y_edges

def get_packing_bins_for_histogram(local_packing_fraction: np.ndarray, num_bins_along_dim: int, L: float, particle_area: float) -> np.ndarray:
    # Due to discretization of simulation box, not all packing fractions are possible (only discrete number, since 
    # discrete number of particles). To avoid gaps in histogram, must make each bin correspond to adding a single
    # particle to the smaller box.
    bin_area = (L/num_bins_along_dim)**2
    delta_packing_fraction = particle_area / bin_area
    bin_centers = delta_packing_fraction*np.arange(0, np.ceil(np.max(local_packing_fraction)/delta_packing_fraction)+1)
    bin_edges = edges_from_centers_linear(bin_centers)
    return bin_edges

def compute_local_packing_fraction_histogram(exp_data: dict, num_bins_along_dim: int, L: float, particle_area: float) -> tuple:
    local_packing_fractions, _, _ = compute_local_packing_fractions(exp_data, num_bins_along_dim, L, particle_area)
    bin_edges = get_packing_bins_for_histogram(local_packing_fractions, num_bins_along_dim, L, particle_area)
    packing_fraction_histogram, _ = np.histogram(local_packing_fractions.flatten(), bins=bin_edges, density=True)
    return packing_fraction_histogram, bin_edges

def compute_dir_cross_vel(exp_data: dict) -> np.ndarray:
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

def compute_dir_cross_vel_norm(exp_data: dict) -> np.ndarray:
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

def save_statistics(experiment_folder_name: str, run_folder_name: str, snapshot_dir: str, v0: float, L: float, num_bins_along_dim: int, particle_area: float):
    start_saving_time = time.time()
    save_base_filepath = os.path.join('saved_data', experiment_folder_name, run_folder_name)
    os.makedirs(save_base_filepath)
    exp_data = rd.get_exp_data(snapshot_dir)
    load_data_time = time.time()
    print(f'Loaded data in {(load_data_time-start_saving_time):.1f}')

    # Director vicsek parameter
    director_vicsek_parameter = compute_director_vicsek_parameter(exp_data)
    director_vicsek_parameter_dict = {
        'director_vicsek_parameter' : director_vicsek_parameter
        }
    director_vicsek_parameter_filepath = os.path.join(save_base_filepath, 'director_vicsek_parameter.pickle')
    with open(director_vicsek_parameter_filepath, 'wb') as file:
        pickle.dump(director_vicsek_parameter_dict, file)

    # Velocity vicsek parameter
    velocity_vicsek_parameter = compute_velocity_vicsek_parameter(exp_data, v0)
    velocity_vicsek_parameter_dict = {
        'velocity_vicsek_parameter' : velocity_vicsek_parameter
        }
    velocity_vicsek_parameter_filepath = os.path.join(save_base_filepath, 'velocity_vicsek_parameter.pickle')
    with open(velocity_vicsek_parameter_filepath, 'wb') as file:
        pickle.dump(velocity_vicsek_parameter_dict, file)
    vicsek_param_time = time.time()
    print(f'Vicsek Param in {(vicsek_param_time - load_data_time):.1f}')

    # Velocity magnitude histogram
    velocity_magnitude_histogram, velocity_magnitude_histogram_bin_edges = compute_velocity_magnitude_histogram(exp_data)
    velocity_magnitude_histogram_dict = {
        'velocity_magnitude_histogram' : velocity_magnitude_histogram,
        'bin_edges' : velocity_magnitude_histogram_bin_edges
        }
    velocity_magnitude_histogram_parameter_filepath = os.path.join(save_base_filepath, 'velocity_magnitude_histogram.pickle')
    with open(velocity_magnitude_histogram_parameter_filepath, 'wb') as file:
        pickle.dump(velocity_magnitude_histogram_dict, file)

    # Velocity x component histogram
    velocity_x_histogram, velocity_x_histogram_bin_edges = compute_velocity_x_histogram(exp_data)
    velocity_x_histogram_dict = {
        'velocity_x_histogram' : velocity_x_histogram,
        'bin_edges' : velocity_x_histogram_bin_edges
        }
    velocity_x_histogram_parameter_filepath = os.path.join(save_base_filepath, 'velocity_x_histogram.pickle')
    with open(velocity_x_histogram_parameter_filepath, 'wb') as file:
        pickle.dump(velocity_x_histogram_dict, file)

    # Velocity y component histogram
    velocity_y_histogram, velocity_y_histogram_bin_edges = compute_velocity_y_histogram(exp_data)
    velocity_y_histogram_dict = {
        'velocity_y_histogram' : velocity_y_histogram,
        'bin_edges' : velocity_y_histogram_bin_edges
        }
    velocity_y_histogram_parameter_filepath = os.path.join(save_base_filepath, 'velocity_y_histogram.pickle')
    with open(velocity_y_histogram_parameter_filepath, 'wb') as file:
        pickle.dump(velocity_y_histogram_dict, file) 
    velocity_hist_param_time = time.time()
    print(f'Velocity histograms in {(velocity_hist_param_time - vicsek_param_time):.1f}')  

    # Save density histogram
    packing_fraction_histogram, packing_fraction_bin_edges = compute_local_packing_fraction_histogram(exp_data, num_bins_along_dim, L, particle_area)
    packing_fraction_histogram_dict = {
        'packing_fraction_histogram' : packing_fraction_histogram,
        'bin_edges' : packing_fraction_bin_edges
    }
    packing_fraction_histogram_parameter_filepath = os.path.join(save_base_filepath, 'packing_fraction_histogram.pickle')
    with open(packing_fraction_histogram_parameter_filepath, 'wb') as file:
        pickle.dump(packing_fraction_histogram_dict, file)
    packing_fraction_param_time = time.time()
    print(f'Packing Fractions in {(packing_fraction_param_time - velocity_hist_param_time):.1f}')  

    # Save MSD with and without flocking
    msd_normal = get_msd(exp_data, L, msd_type='normal')
    msd_minus_flocking = get_msd(exp_data, L, msd_type='normal_minus_avg')
    msd_dict = {
        'msd_normal' : msd_normal,
        'msd_minus_flocking' : msd_minus_flocking
    }
    msd_parameter_filepath = os.path.join(save_base_filepath, 'msd.pickle')
    with open(msd_parameter_filepath, 'wb') as file:
        pickle.dump(msd_dict, file)
    msd_param_time = time.time()
    print(f'MSDs in {(msd_param_time - packing_fraction_param_time):.1f}')  

    # Save director cross velocity
    dir_cross_vel = compute_dir_cross_vel(exp_data)
    dir_cross_vel_norm = compute_dir_cross_vel_norm(exp_data)
    dir_cross_vel_dict = {
        'dir_cross_vel' : dir_cross_vel,
        'dir_cross_vel_norm' : dir_cross_vel_norm
    }
    dir_cross_vel_parameter_filepath = os.path.join(save_base_filepath, 'dir_cross_vel.pickle')
    with open(dir_cross_vel_parameter_filepath, 'wb') as file:
        pickle.dump(dir_cross_vel_dict, file)

if __name__ == '__main__':
    processes = []
    """
    # Polar align statistics
    save_folder_name = os.path.join("/home/ryanlopez", f'Polar_Align_Box_L=200_July_8')
    packing_frac = 0.6
    v0 = 0.03
    J_vals = np.logspace(-3, 0, num=13)
    D_r_vals = np.logspace(-3, 0, num=13)[:-2] #excluding top two rows of Dr
    box_length = 200
    particle_area = np.pi
    num_bins_along_dim = int(box_length / 14) # for computing local packing fraction, ~ number of particle radii long each box is
    for J in J_vals:
        for D_r in D_r_vals:
            exp_folder_name = "phi=%.4f_and_v0=%.4f"%(packing_frac, v0)
            run_folder_name = "J=%.4f_and_Dr=%.4f"%(J, D_r)
            exp_dir, run_dir, snapshot_dir = directories.get_dir_names(save_folder_name, exp_folder_name, run_folder_name)
            p = multiprocessing.Process(target=save_statistics, 
                                        args=('Polar_Align_Box_L=200_July_8', run_folder_name, snapshot_dir, v0, box_length, num_bins_along_dim, particle_area)
                                        )
            processes.append(p)
            p.start()
            if len(processes) == 20: #if reach max allowed threads 
                for process in processes: #wait for all processes to end
                    process.join()
                processes = [] #reset processes after they all end
                print('Finished with set of threads')
    """
    # Force align statistics
    save_folder_name = os.path.join("/home/ryanlopez", f'Force_Align_Box_L=200_July_8')
    packing_frac = 0.6
    v0 = 0.03
    Jv_vals = np.logspace(-4, 0, num=17)
    D_r_vals = np.logspace(-4, -1, num=13)[:-2] #excluding top two rows of Dr
    box_length = 200
    particle_area = np.pi
    num_bins_along_dim = int(box_length / 14) # for computing local packing fraction, ~ number of particle radii long each box is
    for Jv_val in Jv_vals:
        for D_r in D_r_vals:
            J = Jv_val / v0
            exp_folder_name = "phi=%.4f_and_v0=%.4f"%(packing_frac, v0)
            run_folder_name = "J=%.4f_and_Dr=%.4f"%(J, D_r)
            exp_dir, run_dir, snapshot_dir = directories.get_dir_names(save_folder_name, exp_folder_name, run_folder_name)
            p = multiprocessing.Process(target=save_statistics, 
                                        args=('Force_Align_Box_L=200_July_8', run_folder_name, snapshot_dir, v0, box_length, num_bins_along_dim, particle_area)
                                        )
            processes.append(p)
            p.start()
            if len(processes) == 20: #if reach max allowed threads 
                for process in processes: #wait for all processes to end
                    process.join()
                processes = [] #reset processes after they all end
                print('Finished with set of threads')