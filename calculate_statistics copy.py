import os
import numpy as np
import json
import read_data as rd #reads snapshot text data
from collections import OrderedDict
import argparse
from Analysis_Calculations.msd import get_msd
from Analysis_Calculations.local_packing_fraction import get_local_packing_fraction, get_packing_hist_peak_distance, get_packing_mean_and_std_dev
from Analysis_Calculations.flocking_factors import get_velocity_vicsek_param, get_flocking_factors
from Analysis_Calculations.dir_cross_vel import get_dir_cross_vel_norm, get_dir_cross_vel
from Analysis_Calculations.dir_dot_vel import get_dir_dot_vel_norm, get_dir_dot_vel
from Analysis_Calculations.speed_dist import get_speed_data

def save_local_packing_fraction(analysis_data_dir, exp_data, run_desc):
    local_packing_hist_filename = os.path.join(analysis_data_dir, 'local_packing_hist.npy')
    #Parameters
    box_length = run_desc['L']
    num_bins_along_dim = int(box_length/14)
    particle_area = np.pi*run_desc['radius']**2
    #Calculate and save local packing histogram
    local_packing, _, _ = get_local_packing_fraction(exp_data, num_bins_along_dim, box_length, particle_area)
    with open(local_packing_hist_filename, 'wb') as local_packing_hist_file:
        np.save(local_packing_hist_file, local_packing) 

def get_packing_fraction_std_dev(analysis_data_dir, run_desc):
    #Parameters
    box_length = run_desc['L']
    num_bins_along_dim = int(box_length/14)
    particle_area = np.pi*run_desc['radius']**2
    #Calculate local packing fraction std dev
    local_packing_hist_filename = os.path.join(analysis_data_dir, 'local_packing_hist.npy')
    with open(local_packing_hist_filename, 'rb') as local_packing_file:
        local_packing_fraction = np.load(local_packing_file)
    _, std_dev, _ = get_packing_mean_and_std_dev(local_packing_fraction, num_bins_along_dim, box_length, particle_area)
    return std_dev

def get_packing_peak_distance(analysis_data_dir, run_desc):
    #Parameters
    box_length = run_desc['L']
    num_bins_along_dim = int(box_length/14)
    particle_area = np.pi*run_desc['radius']**2
    local_packing_hist_filename = os.path.join(analysis_data_dir, 'local_packing_hist.npy')
    with open(local_packing_hist_filename, 'rb') as local_packing_file:
        local_packing_fraction = np.load(local_packing_file)
    peak_distance = get_packing_hist_peak_distance(local_packing_fraction, num_bins_along_dim, box_length, particle_area)
    return peak_distance

def save_msd(analysis_data_dir, exp_data, run_desc, msd_type):
    msd_filename = os.path.join(analysis_data_dir, f'msd_{msd_type}.npy')
    #Parameters
    box_length = run_desc['L']
    v0 = run_desc['v0']
    final_time = run_desc['tf']
    total_snapshots = run_desc['total_snapshots']
    snapshot_delta_time = final_time / total_snapshots
    if v0*snapshot_delta_time > box_length/2: #can't calculate MSD because of boundary conditions
        return
    #Calculate and save MSD
    if not os.path.isfile(msd_filename):
        msd = get_msd(exp_data, box_length, msd_type=msd_type)
        with open(msd_filename, 'wb') as msd_normal_file:
            np.save(msd_normal_file, msd) 

"""
Calculates and saves statistics on data
"""
#Get simulation directory to calculate statistics on from command line
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', help='Run directory')
args = parser.parse_args()
run_dir = args.run_dir

#Load necessary data
run_desc_file = open(os.path.join(run_dir, 'run_desc.json'))
run_desc = json.loads(run_desc_file.read())
run_desc_file.close()
snapshot_dir = os.path.join(run_dir, "snapshot_data/")
exp_data = rd.get_exp_data(snapshot_dir)

#Save array data
analysis_data_dir = os.path.join(run_dir, 'analysis_data')
if not os.path.isdir(analysis_data_dir):
    os.mkdir(analysis_data_dir)
save_local_packing_fraction(analysis_data_dir, exp_data, run_desc)
#save_msd(analysis_data_dir, exp_data, run_desc, 'normal')
#save_msd(analysis_data_dir, exp_data, run_desc, 'normal_minus_avg')

#Save single statistic values
#velocity_vicsek_param = get_velocity_vicsek_param(exp_data)
vicsek_param, vel_param = get_flocking_factors(exp_data, run_desc['v0'])
dir_dot_vel_val = get_dir_dot_vel(exp_data)
dir_dot_vel_val_norm = get_dir_dot_vel_norm(exp_data)
dir_dot_vel_norm_val = get_dir_dot_vel_norm(exp_data)
dir_cross_vel_val = get_dir_cross_vel(exp_data)
dir_cross_vel_norm_val = get_dir_cross_vel_norm(exp_data)

single_stats = OrderedDict()
#single_stats['velocity_vicsek_param'] = velocity_vicsek_param
single_stats['packing_std_dev'] = get_packing_fraction_std_dev(analysis_data_dir, run_desc)
single_stats['vicsek_param'] = np.average(vicsek_param)
single_stats['dir_dot_vel'] = np.average(dir_dot_vel_val / run_desc['v0'])
single_stats['dir_dot_vel_norm'] = np.average(dir_dot_vel_norm_val)
single_stats['dir_cross_vel'] = np.average(np.abs(dir_cross_vel_val) / run_desc['v0'])
single_stats['dir_cross_vel_norm'] = np.average(np.abs(dir_cross_vel_norm_val))

single_stats['vicsek_param_std_dev'] = np.std(vicsek_param)
single_stats['dir_dot_vel_std_dev'] = np.std(dir_dot_vel_val / run_desc['v0'])
single_stats['dir_dot_vel_norm_std_dev'] = np.std(dir_dot_vel_norm_val)
single_stats['dir_cross_vel_std_dev'] = np.std(dir_cross_vel_val / run_desc['v0'])
single_stats['dir_cross_vel_norm_std_dev'] = np.std(dir_cross_vel_norm_val)

single_stats_file = open(os.path.join(run_dir, "single_stats.json"), 'w')
single_stats_file.write(json.dumps(single_stats))
single_stats_file.close()

print('Finished calculating statistics on simulation.')