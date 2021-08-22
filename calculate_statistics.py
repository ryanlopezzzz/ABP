"""
Calculates the following statistics and saves to single_stats.json for each run in exp_dir:

vicsek_param : | \sum n_i | / N
dir_dot_vel : \sum (n_i \dot v_i) / N
dir_dot_vel_norm : \sum (n_i \dot v_i / |v_i|) / N
dir_cross_vel : \sum (n_i \cross v_i) / N
dir_cross_vel_norm : \sum (n_i \cross v_i / |v_i|) / N
"""
save_dir = "/Users/ryanlopez/Desktop/Python_Programs/Dr_Marchetti_Research/Server_Data"
exp_folder_name = "liquid_vary_J_Dr" #Folder name of experiment directory

import os
import numpy as np
import json
import read_data as rd #reads snapshot text data
from collections import OrderedDict
import Physical_Quantities.flocking_factors as flocking_factors
import Physical_Quantities.various as various

#Directory where all data is saved

exp_dir = os.path.join(save_dir, exp_folder_name)

for i, f in enumerate(os.scandir(exp_dir)): #runs through all immediate subdirectories
    if f.is_dir() and f.path != os.path.join(exp_dir, ".ipynb_checkpoints"):            
        snapshot_dir = os.path.join(f, "snapshot_data/")
        exp_data = rd.get_exp_data(snapshot_dir)

        run_desc_file = open(os.path.join(f, 'run_desc.json'))
        run_desc = json.loads(run_desc_file.read())
        run_desc_file.close()

        vicsek_param, vel_param = flocking_factors.get_flocking_factors(exp_data, run_desc['v0'])
        dir_dot_vel, dir_dot_vel_norm = various.get_dir_dot_vel(exp_data)
        dir_cross_vel, dir_cross_vel_norm = various.get_dir_cross_vel(exp_data)

        single_stats = OrderedDict()
        single_stats['vicsek_param'] = np.average(vicsek_param)
        single_stats['dir_dot_vel'] = np.average(dir_dot_vel / run_desc['v0'])
        single_stats['dir_dot_vel_norm'] = np.average(dir_dot_vel_norm)
        single_stats['dir_cross_vel'] = np.average(dir_cross_vel / run_desc['v0'])
        single_stats['dir_cross_vel_norm'] = np.average(dir_cross_vel_norm)
        
        single_stats['vicsek_param_std_dev'] = np.std(vicsek_param)
        single_stats['dir_dot_vel_std_dev'] = np.std(dir_dot_vel / run_desc['v0'])
        single_stats['dir_dot_vel_norm_std_dev'] = np.std(dir_dot_vel_norm)
        single_stats['dir_cross_vel_std_dev'] = np.std(dir_cross_vel / run_desc['v0'])
        single_stats['dir_cross_vel_norm_std_dev'] = np.std(dir_cross_vel_norm)

        single_stats_file = open(os.path.join(f, "single_stats.json"), 'w')
        single_stats_file.write(json.dumps(single_stats))
        single_stats_file.close()
        print('Finished ', i, ' run.')