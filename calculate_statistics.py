import os
import numpy as np
import json
import read_data as rd #reads snapshot text data
from collections import OrderedDict
import argparse
from Analysis_Calculations import (flocking_factors, dir_cross_vel, dir_dot_vel, msd, speed_dist)

"""
Calculates the following statistics (and some more) and saves:

vicsek_param : | \sum n_i | / N
dir_dot_vel : \sum (n_i \dot v_i) / N
dir_dot_vel_norm : \sum (n_i \dot v_i / |v_i|) / N
dir_cross_vel : \sum (n_i \cross v_i) / N
dir_cross_vel_norm : \sum (n_i \cross v_i / |v_i|) / N
"""
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', help='Run directory')
args = parser.parse_args()
run_dir = args.run_dir
    
snapshot_dir = os.path.join(run_dir, "snapshot_data/")
exp_data = rd.get_exp_data(snapshot_dir)
print('Finished loading data files. Calculating Statistics.')

run_desc_file = open(os.path.join(run_dir, 'run_desc.json'))
run_desc = json.loads(run_desc_file.read())
run_desc_file.close()

vicsek_param, vel_param = flocking_factors.get_flocking_factors(exp_data, run_desc['v0'])
dir_dot_vel_val = dir_dot_vel.get_dir_dot_vel(exp_data)
dir_dot_vel_norm_val = dir_dot_vel.get_dir_dot_vel_norm(exp_data)
dir_cross_vel_val = dir_cross_vel.get_dir_cross_vel(exp_data)
dir_cross_vel_norm_val = dir_cross_vel.get_dir_cross_vel_norm(exp_data)

single_stats = OrderedDict()
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

#np_msd = msd.new_MSD_sim_data(exp_data, run_desc['L'])
#np_msd_minus_flock = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True)
#np_msd_parallel = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True, msd_type='parallel')
#np_msd_perp = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True, msd_type='perpendicular')

#np.save(os.path.join(f, 'msd.npy'), np_msd)
#np.save(os.path.join(f, 'msd_minus_flock.npy'), np_msd_minus_flock)
#np.save(os.path.join(f, 'msd_parallel.npy'), np_msd_parallel)
#np.save(os.path.join(f, 'msd_perp.npy'), np_msd_perp)

#print('Finished ', i+1, ' run.')