import json
import multiprocessing
import numpy as np
import os
import json
import read_data as rd
from Analysis_Calculations.msd import get_msd
from Analysis_Calculations.flocking_factors import get_velocity_vicsek_param
from Analysis_Calculations.dir_cross_vel import get_dir_cross_vel_norm
from Analysis_Calculations.local_packing_fraction import get_local_packing_fraction, get_packing_hist_peak_distance, get_packing_mean_and_std_dev
from Utils.graphing_helpers import save_fig_pdf, edges_from_centers_linear, edges_from_centers_log

def create_new_single_stats(run_dir):
    """
    Copies single stats file to new single stats file if doesnt exist
    """
    single_stats_filename = os.path.join(run_dir, 'single_stats.json')
    new_single_stats_filename = os.path.join(run_dir, 'new_single_stats.json')
    if not os.path.isfile(new_single_stats_filename):
        with open(single_stats_filename, 'r') as single_stats_file:
            single_stats = json.loads(single_stats_file.read())
        with open(new_single_stats_filename, 'w') as new_single_stats_file:
            new_single_stats_file.write(json.dumps(single_stats))

def save_local_packing_fraction(run_dir):
    num_bins_along_dim = 18

    #Load data
    snapshot_dir = os.path.join(run_dir, 'snapshot_data')
    run_desc_filename = os.path.join(run_dir, 'run_desc.json')
    analysis_data_dir = os.path.join(run_dir, 'analysis_data')
    local_packing_hist_filename = os.path.join(analysis_data_dir, 'local_packing_hist.npy')
    with open(run_desc_filename, 'r') as run_desc_file:
        run_desc = json.loads(run_desc_file.read())
    box_length = run_desc['L']
    particle_area = np.pi*run_desc['radius']**2
    exp_data = rd.get_exp_data(snapshot_dir)

    #Calculate local packing histogram
    local_packing, _, _ = get_local_packing_fraction(exp_data, num_bins_along_dim, box_length, particle_area)

    #Save to file
    if not os.path.isdir(analysis_data_dir):
        os.mkdir(analysis_data_dir)
    with open(local_packing_hist_filename, 'wb') as local_packing_hist_file:
        np.save(local_packing_hist_file, local_packing) 
    print(f'Finished {run_dir}')

def save_packing_fraction_std_dev(run_dir):
    num_bins_along_dim = 5

    #Load data
    local_packing_hist_filename = os.path.join(run_dir, 'analysis_data', 'local_packing_hist.npy')
    run_desc_filename = os.path.join(run_dir, 'run_desc.json')
    single_stats_filename = os.path.join(run_dir, 'new_single_stats.json')
    with open(local_packing_hist_filename, 'rb') as local_packing_file:
        local_packing_fraction = np.load(local_packing_file)
    with open(run_desc_filename, 'r') as run_desc_file:
        run_desc = json.loads(run_desc_file.read())
    with open(single_stats_filename, 'r') as single_stats_file:
        single_stats = json.loads(single_stats_file.read())
    box_length = run_desc['L']
    particle_area = np.pi*run_desc['radius']**2

    #Calculate std dev
    _, std_dev, _ = get_packing_mean_and_std_dev(local_packing_fraction, num_bins_along_dim, box_length, particle_area)

    #Save to new json file
    single_stats['packing_std_dev'] = std_dev
    with open(single_stats_filename, 'w') as single_stats_file:
        single_stats_file.write(json.dumps(single_stats))
    print(f'Finished {run_dir}')

def save_packing_peak_distance(run_dir):
    num_bins_along_dim = 5

    #Load data
    snapshot_dir = os.path.join(run_dir, 'snapshot_data')
    run_desc_filename = os.path.join(run_dir, 'run_desc.json')
    single_stats_filename = os.path.join(run_dir, 'single_stats.json')
    new_single_stats_filename = os.path.join(run_dir, 'new_single_stats.json')
    with open(run_desc_filename, 'r') as run_desc_file:
        run_desc = json.loads(run_desc_file.read())
    with open(single_stats_filename, 'r') as single_stats_file:
        single_stats = json.loads(single_stats_file.read())
    box_length = run_desc['L']
    particle_area = np.pi*run_desc['radius']**2
    exp_data = rd.get_exp_data(snapshot_dir)

    #Calculate peak distance
    peak_distance = get_packing_hist_peak_distance(exp_data, num_bins_along_dim, box_length, particle_area)

    #Save to new json file
    single_stats['packing_peak_distance'] = peak_distance
    with open(new_single_stats_filename, 'w') as new_single_stats_file:
        new_single_stats_file.write(json.dumps(single_stats))
    print(f'Finished {run_dir}')

def save_velocity_vicsek_param(run_dir):
    #Load data
    snapshot_dir = os.path.join(run_dir, 'snapshot_data')
    new_single_stats_filename = os.path.join(run_dir, 'new_single_stats.json')
    with open(new_single_stats_filename, 'r') as new_single_stats_file:
        new_single_stats = json.loads(new_single_stats_file.read())
    exp_data = rd.get_exp_data(snapshot_dir)

    #Calculate peak distance
    velocity_vicsek_param = get_velocity_vicsek_param(exp_data)

    #Save to new json file
    new_single_stats['velocity_vicsek_param'] = velocity_vicsek_param
    with open(new_single_stats_filename, 'w') as new_single_stats_file:
        new_single_stats_file.write(json.dumps(new_single_stats))
    print(f'Finished {run_dir}')

def save_msd(run_dir, msd_type):
    #Load data
    snapshot_dir = os.path.join(run_dir, 'snapshot_data')
    run_desc_filename = os.path.join(run_dir, 'run_desc.json')
    with open(run_desc_filename, 'r') as run_desc_file:
        run_desc = json.loads(run_desc_file.read())
    box_length = run_desc['L']
    v0 = run_desc['v0']
    final_time = run_desc['tf']
    total_snapshots = run_desc['total_snapshots']
    snapshot_delta_time = final_time / total_snapshots
    if v0*snapshot_delta_time > box_length/2: #can't calculate MSD because of boundary conditions
        return
    exp_data = rd.get_exp_data(snapshot_dir)
    #Calculate MSD
    msd = get_msd(exp_data, box_length, msd_type=msd_type)
    #Save to file
    analysis_data_dir = os.path.join(run_dir, 'analysis_data')  
    if msd_type == 'normal':
        msd_filename =  os.path.join(analysis_data_dir, 'msd_normal.npy')
    elif msd_type == 'normal_minus_avg':
        msd_filename = os.path.join(analysis_data_dir, 'msd_normal_minus_avg.npy')
    if not os.path.isdir(analysis_data_dir):
        os.mkdir(analysis_data_dir)
    with open(msd_filename, 'wb') as msd_normal_file:
        np.save(msd_normal_file, msd) 

def save_multiple_data(run_dir):
    #Save array data
    analysis_folder = os.path.join(run_dir, 'analysis_data')
    local_packing_filename = os.path.join(analysis_folder, 'local_packing_hist.npy')
    if not os.path.isfile(local_packing_filename):
        save_local_packing_fraction(run_dir)
    normal_msd_filename = os.path.join(analysis_folder, 'msd_normal.npy')
    if not os.path.isfile(normal_msd_filename):
        save_msd(run_dir, 'normal')
    normal_minus_avg_msd_filename = os.path.join(analysis_folder, 'msd_normal_minus_avg.npy')
    if not os.path.isfile(normal_minus_avg_msd_filename):
        save_msd(run_dir, 'normal_minus_avg')
    #Save single stats data
    create_new_single_stats(run_dir)
    new_single_stats_filename = os.path.join(run_dir, 'new_single_stats.json')
    with open(new_single_stats_filename, 'r') as single_stats_file:
        new_single_stats = json.loads(single_stats_file.read())
    if 'packing_std_dev' not in new_single_stats:
        save_packing_fraction_std_dev(run_dir)
    if 'velocity_vicsek_param' not in new_single_stats:
        save_velocity_vicsek_param(run_dir)

if __name__ == '__main__':
    processes = []
    save_dir = "/home/ryanlopez/Polar_Align_Vary_Phi_V_Saved_Data2"
    phi_vals = [0.6]
    v0_vals = [0.1]
    J_vals = np.logspace(-4, -1, num=7)
    Dr_vals = np.logspace(-4, -1, num=7)
    for phi in phi_vals:
        for v0 in v0_vals:
            for J in J_vals:
                for Dr in Dr_vals:
                    if (J in np.logspace(-3, 0, num=4)) and (Dr in np.logspace(-3, 0, num=4)):
                        continue
                    run_dir = os.path.join(save_dir, "phi=%.4f_and_v0=%.4f"%(phi, v0), "J=%.4f_and_Dr=%.4f"%(J, Dr))
                    p = multiprocessing.Process(target=save_multiple_data, args=(run_dir,))
                    processes.append(p)
                    p.start()
                    if len(processes) == 22: #if reach max allowed threads 
                        for process in processes: #wait for all processes to end
                            process.join()
                        processes = [] #reset processes after they all end
                        print('Finished with set of threads')