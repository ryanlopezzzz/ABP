import resource
gigabyte = int(1e9)
resource.setrlimit(resource.RLIMIT_AS, (3*gigabyte, 10*gigabyte)) #sets memory limit to avoid large calculations

import multiprocessing
import subprocess
import numpy as np
import os
import json
import time
import random
from collections import OrderedDict
import copy
import directories

"""
Script for running multiple jobs.
"""
def save_run_desc(run_desc, run_desc_filename):
    # Saves a file with all the parameters for the run
    with open(run_desc_filename, 'w') as run_desc_file:
        run_desc_file.write(json.dumps(run_desc))

def run_simulation(run_dir):
    # Runs the simulation and saves the particle positions, velocities, etc to a folder (not included here).
    run_desc_filename = os.path.join(run_dir, "run_desc.json")
    run_simulation_command = ['python3', 'ABP_Simulation.py', '--paramsFilename', run_desc_filename]
    with open(os.path.join(run_dir, 'stdout.txt'), 'w') as stdout_file:
        subprocess.run(run_simulation_command, stdout = stdout_file)

def calculate_statistics(run_dir):
    # Computes statistics on simulation data and saves it (not included here)
    calculate_statistics_command = ['python3', 'calculate_statistics.py', '--run_dir', run_dir]
    subprocess.run(calculate_statistics_command)

def run_on_thread(run_dir):
    # This function is run in parallel on different threads
    totalstart_time = time.time()
    run_simulation(run_dir)
    calculate_statistics(run_dir)
    print('Total took {} seconds'.format(time.time() - totalstart_time))

default_run_desc = OrderedDict({
    'J': 0.1, 
    'D_r': 0.1, 
    'v0': 0.03, #fix at this value
    'packing_frac': 0.6, #fix at this value
    'gamma_t': 1,
    'gamma_r': 1,
    'kT': 0,
    'radius': 1,
    'poly': 0,
    'k': 1,
    'L': 200,
    'warm_up_time': 1e5,
    'tf': 2.5e5,
    'tstep': 0.01,
    'rand_seed': None,
    'vel_align_norm': False,
    'velocity_align': False,
    'polar_align': True,
    'total_snapshots': 1000
})

if __name__ == '__main__':
    processes = []
    save_folder_name = os.path.join("/home/ryanlopez", f'Polar_Align_Box_L=200_July_8')
    if not os.path.isdir(save_folder_name):
        os.mkdir(save_folder_name)
    starttime = time.time()
    folder_run_desc = copy.deepcopy(default_run_desc)
    J_vals = np.logspace(-3, 0, num=13)
    D_r_vals = np.logspace(-3, 0, num=13)[:-2] #excluding top two rows of Dr
    for J_val in J_vals:
        for D_r_val in D_r_vals:
            run_desc = copy.deepcopy(folder_run_desc)
            J = J_val
            D_r = D_r_val
            run_desc['J'] = J
            run_desc['D_r'] = D_r
            run_desc['rand_seed'] = random.randint(1,10000)
            exp_folder_name = "phi=%.4f_and_v0=%.4f"%(run_desc['packing_frac'], run_desc['v0'])
            run_folder_name = "J=%.4f_and_Dr=%.4f"%(J, D_r)
            exp_dir, run_dir, snapshot_dir = directories.get_dir_names(save_folder_name, exp_folder_name, run_folder_name)
            run_desc['exp_dir'] = exp_dir
            run_desc['run_dir'] = run_dir
            run_desc['snapshot_dir'] = snapshot_dir
            directories.create(exp_dir, run_dir, snapshot_dir)
            run_desc_filename = os.path.join(run_dir, "run_desc.json")
            save_run_desc(run_desc, run_desc_filename)
            p = multiprocessing.Process(target=run_on_thread, args=(run_dir,))
            processes.append(p)
            p.start()
            if len(processes) == 20: #if reach max allowed threads 
                for process in processes: #wait for all processes to end
                    process.join()
                processes = [] #reset processes after they all end
                print('Finished with set of threads')