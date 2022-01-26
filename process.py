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
    with open(run_desc_filename, 'w') as run_desc_file:
        run_desc_file.write(json.dumps(run_desc))

def run_simulation(run_dir):
    run_desc_filename = os.path.join(run_dir, "run_desc.json")
    run_simulation_command = ['python3', 'ABP_Simulation.py', '--paramsFilename', run_desc_filename]
    with open(os.path.join(run_dir, 'stdout.txt'), 'w') as stdout_file:
        subprocess.run(run_simulation_command, stdout = stdout_file)

def calculate_statistics(run_dir):
    calculate_statistics_command = ['python3', 'calculate_statistics.py', '--run_dir', run_dir]
    subprocess.run(calculate_statistics_command)

def run_on_thread(run_dir):
    run_simulation(run_dir)
    calculate_statistics(run_dir)

default_run_desc = OrderedDict({
    'J': 0.1, 
    'D_r': 0.1,
    'v0': 0.01,
    'packing_frac': 0.8,
    'gamma_t': 1,
    'gamma_r': 1,
    'kT': 0,
    'radius': 1,
    'poly': 0,
    'k': 1,
    'L': 70,
    'warm_up_time': 1e5,
    'tf': 5e5,
    'tstep': 1e-1,
    'rand_seed': None,
    'vel_align_norm': False,
    'velocity_align': True,
    'polar_align': False,
    'total_snapshots': 1000
})

if __name__ == '__main__':
    processes = []
    for phi in [0.4, 0.6, 0.8, 1]:
        for v0 in [0.01, 0.03,0.1]:
            starttime = time.time()

            folder_run_desc = copy.deepcopy(default_run_desc)
            folder_run_desc['packing_frac'] = phi
            folder_run_desc['v0'] = v0
            for Jv in np.logspace(-3,0,num=4):
                for D_r in np.logspace(-3,0,num=4):
                    J=(Jv/v0)
                    run_desc = copy.deepcopy(folder_run_desc)
                    run_desc['J'] = J
                    run_desc['D_r'] = D_r
                    run_desc['rand_seed'] = random.randint(1,10000)
                    save_folder_name = os.path.join("/home/ryanlopez", 'Velocity_Align_Vary_Phi_V_Saved_Data2')
                    exp_folder_name = "phi=%.4f_and_v0=%.4f"%(phi, v0)
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
                    if len(processes) == 16: #if reach max allowed threads 
                        for process in processes: #wait for all processes to end
                            process.join()
                        processes = [] #reset processes after they all end
                        print('Finished with set of 16 threads')

            print('Finished with ' + exp_folder_name)
            print('That took {} seconds'.format(time.time() - starttime))


"""def create_directories_and_update_run_desc(run_desc):
    #Create directories for saving data and update run_desc in place with new directory paths, return path
    #for saved run_desc
    base_dir = '/home/ryanlopez'
    save_dir = os.path.join(base_dir, run_desc['save_folder_name'])
    exp_dir, run_dir, snapshot_dir = directories.create(
        save_dir, run_desc['exp_folder_name'], run_desc['run_folder_name']
        )
    run_desc.pop('save_folder_name')
    run_desc.pop('exp_folder_name')
    run_desc.pop('run_folder_name')
    run_desc['exp_dir'] = exp_dir
    run_desc['run_dir'] = run_dir
    run_desc['snapshot_dir'] = snapshot_dir
    run_desc_filename = os.path.join(run_dir, "run_desc.json")
    with open(run_desc_filename, 'w') as run_desc_file:
        run_desc_file.write(json.dumps(run_desc))

def run_simulation(run_desc):
    with open(run_desc_filename, 'w') as run_desc_file:
        run_desc_file.write(json.dumps(run_desc))
    #Run simulation
    run_simulation_command = ['python3', 'ABP_Simulation.py', '--paramsFilename', run_desc_filename]
    with open(os.path.join(run_dir, 'stdout.txt'), 'w') as stdout_file:
        subprocess.run(run_simulation_command, stdout = stdout_file)

default_run_desc = OrderedDict({
    'J': 0.1, 
    'D_r': 0.1,
    'v0': 0.01,
    'packing_frac': 0.8,
    'gamma_t': 1,
    'gamma_r': 1,
    'kT': 0,
    'radius': 1,
    'poly': 0,
    'k': 1,
    'L': 70,
    'warm_up_time': 1e5,
    'tf': 5e5,
    'tstep': 1e-1,
    'rand_seed': None,
    'vel_align_norm': False,
    'velocity_align': False,
    'polar_align': True,
    'total_snapshots': 100
})

if __name__ == '__main__':
    for phi in [0.4, 0.6, 0.8, 1]:
        for v0 in [0.1, 0.03]:
            starttime = time.time()
            processes = []

            folder_run_desc = copy.deepcopy(default_run_desc)
            folder_run_desc['packing_frac'] = phi
            folder_run_desc['v0'] = v0
            for J in np.logspace(-3,0,num=4):
                for D_r in np.logspace(-3,0,num=4):
                    run_desc = copy.deepcopy(folder_run_desc)
                    run_desc['J'] = J
                    run_desc['D_r'] = D_r
                    run_desc['save_folder_name'] = 'Polar_Align_Vary_Phi_V_Saved_Data'
                    run_desc['exp_folder_name'] = "phi=%.4f_and_v0=%.4f"%(phi, v0)
                    run_desc['run_folder_name'] = "J=%.4f_and_Dr=%.4f"%(J, D_r)
                    run_desc['rand_seed'] = random.randint(1,10000)

                    p = multiprocessing.Process(target=run_simulation, args=(run_desc,))
                    processes.append(p)
                    p.start()
            
            assert(len(processes) <= 16) #set maximum CPU core usage
            for process in processes:
                process.join()

            print('Finished with ' + run_desc['exp_folder_name'])
            print('That took {} seconds'.format(time.time() - starttime))""" 