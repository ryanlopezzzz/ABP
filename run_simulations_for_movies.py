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
from Utils import convert_txt_to_json
convert_txt_to_json = convert_txt_to_json.convert_txt_to_json

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

if __name__ == '__main__':
    processes = []
    save_folder_name = "Velocity_Align_Big_Phase_Diagrams"
    exp_folder_name = 'phi=0.6000_and_v0=0.0300'
    run_folder_name = 'J=0.0333_and_Dr=0.0010'
    simulate_time = 10000
    total_snapshots = 1000
    #Create new movie directory
    load_run_folder_name = os.path.join("/home/ryanlopez", save_folder_name, exp_folder_name, run_folder_name)
    save_exp_folder_name = os.path.join("/home/ryanlopez", save_folder_name + '_movie', exp_folder_name)
    save_run_folder_name = os.path.join(save_exp_folder_name, run_folder_name)
    snapshot_folder_name = os.path.join(save_run_folder_name, 'snapshot_data')
    if not os.path.isdir(save_run_folder_name):
        os.makedirs(save_run_folder_name)
    if not os.path.isdir(snapshot_folder_name):
        os.mkdir(os.path.join(snapshot_folder_name))
    #Load parameters from existing simulation and save to new directory
    run_desc_filename = os.path.join(load_run_folder_name, 'run_desc.json')
    with open(run_desc_filename, 'r') as run_desc_file:
        run_desc = json.load(run_desc_file)
    total_existing_snapshots = run_desc['total_snapshots']
    box_length = run_desc['L']
    #Edit Run Description
    run_desc['warm_up_time'] = 0
    run_desc['tf'] = simulate_time
    run_desc['total_snapshots'] = total_snapshots
    run_desc['exp_dir'] = save_exp_folder_name
    run_desc['run_dir'] = save_run_folder_name
    run_desc['snapshot_dir'] = snapshot_folder_name
    init_json_filename = os.path.join(save_run_folder_name, 'load_state.json')
    run_desc['init_filename'] = init_json_filename
    #Save Run Description
    run_desc_filename = os.path.join(save_run_folder_name, "run_desc.json")
    save_run_desc(run_desc, run_desc_filename)
    #Convert existing snapshot data to json format
    load_snapshot_filename = os.path.join(load_run_folder_name, 'snapshot_data', f'snapshot_{total_existing_snapshots-1:05d}.txt')
    convert_txt_to_json(load_snapshot_filename, init_json_filename, box_length)
    #Run Simulation
    starttime = time.time()
    run_simulation(save_run_folder_name)
    calculate_statistics(save_run_folder_name)
    print('Finished with ' + exp_folder_name)
    print('That took {} seconds'.format(time.time() - starttime))