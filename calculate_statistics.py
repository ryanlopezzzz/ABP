import resource
gigabyte = int(1e9)
resource.setrlimit(resource.RLIMIT_AS, (49*gigabyte, 50*gigabyte))

"""
Calculates the following statistics (and some more) and saves to single_stats.json for each run in exp_dir:

vicsek_param : | \sum n_i | / N
dir_dot_vel : \sum (n_i \dot v_i) / N
dir_dot_vel_norm : \sum (n_i \dot v_i / |v_i|) / N
dir_cross_vel : \sum (n_i \cross v_i) / N
dir_cross_vel_norm : \sum (n_i \cross v_i / |v_i|) / N
"""
save_dir = "/home/ryanlopez/Velocity_Align_Saved_Data"
exp_folder_name = input("Enter exp directory name: ") #Folder name of experiment directory

import os
import numpy as np
import json
import read_data as rd #reads snapshot text data
from collections import OrderedDict
import Physical_Quantities.flocking_factors as flocking_factors
import Physical_Quantities.various as various
import Physical_Quantities.MSD as msd
import Physical_Quantities.local_flocking_R as local_flocking_R

#Directory where all data is saved

exp_dir = os.path.join(save_dir, exp_folder_name)

for i, f in enumerate(os.scandir(exp_dir)): #runs through all immediate subdirectories
    if os.path.isdir(os.path.join(f, "snapshot_data/")):            
        snapshot_dir = os.path.join(f, "snapshot_data/")
        exp_data = rd.get_exp_data(snapshot_dir)
        print('Finished loading data files. Calculating Statistics.')

        run_desc_file = open(os.path.join(f, 'run_desc.json'))
        run_desc = json.loads(run_desc_file.read())
        run_desc_file.close()

        vicsek_param, vel_param = flocking_factors.get_flocking_factors(exp_data, run_desc['v0'])
        dir_dot_vel, dir_dot_vel_norm = various.get_dir_dot_vel(exp_data)
        dir_cross_vel, dir_cross_vel_norm = various.get_dir_cross_vel(exp_data)
        vel_mag_rel_var, v_mag_data = various.get_vel_mag_rel_var(exp_data)
        R_avg, Q = local_flocking_R.get_local_flocking_R(exp_data, 2*run_desc['radius'], run_desc['L'])
        np.save(os.path.join(f, 'Q.npy'), Q)

        single_stats = OrderedDict()
        single_stats['vicsek_param'] = np.average(vicsek_param)
        single_stats['dir_dot_vel'] = np.average(dir_dot_vel / run_desc['v0'])
        single_stats['dir_dot_vel_norm'] = np.average(dir_dot_vel_norm)
        single_stats['dir_cross_vel'] = np.average(np.abs(dir_cross_vel) / run_desc['v0'])
        single_stats['dir_cross_vel_norm'] = np.average(np.abs(dir_cross_vel_norm))
        single_stats['vel_mag_rel_var'] = np.average(vel_mag_rel_var)
        single_stats['R_avg'] = R_avg
        
        single_stats['vicsek_param_std_dev'] = np.std(vicsek_param)
        single_stats['dir_dot_vel_std_dev'] = np.std(dir_dot_vel / run_desc['v0'])
        single_stats['dir_dot_vel_norm_std_dev'] = np.std(dir_dot_vel_norm)
        single_stats['dir_cross_vel_std_dev'] = np.std(dir_cross_vel / run_desc['v0'])
        single_stats['dir_cross_vel_norm_std_dev'] = np.std(dir_cross_vel_norm)

        single_stats_file = open(os.path.join(f, "single_stats.json"), 'w')
        single_stats_file.write(json.dumps(single_stats))
        single_stats_file.close()
        
        np_msd = msd.new_MSD_sim_data(exp_data, run_desc['L'])
        np_msd_minus_flock = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True)
        np_msd_parallel = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True, msd_type='parallel')
        np_msd_perp = msd.new_MSD_sim_data(exp_data, run_desc['L'], subtract_avg_disp=True, msd_type='perpendicular')
        
        
        np.save(os.path.join(f, 'msd.npy'), np_msd)
        np.save(os.path.join(f, 'msd_minus_flock.npy'), np_msd_minus_flock)
        np.save(os.path.join(f, 'msd_parallel.npy'), np_msd_parallel)
        np.save(os.path.join(f, 'msd_perp.npy'), np_msd_perp)
        
        print('Finished ', i+1, ' run.')