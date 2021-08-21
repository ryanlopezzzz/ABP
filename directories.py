#Creates directories for saving experiment data

import os
import datetime

def create(save_dir, exp_folder_name, run_folder_name):

    exp_dir = os.path.join(save_dir, exp_folder_name + "/") #Directory where all data for particular experiment type is stored

    if not os.path.isdir(exp_dir): #Check if exp_dir is already a directory
        try:
            os.mkdir(exp_dir)
            print ("Successfully created the directory %s " % exp_dir)
        except OSError:
            print ("Creation of the directory %s failed" % exp_dir)

    #Create folder to store run data
    run_dir = os.path.join(exp_dir, run_folder_name) #folder where this run data is stored
    try:
        os.mkdir(run_dir)
        print ("Successfully created the directory %s " % run_dir)
    except OSError:
        print ("Creation of the directory %s failed" % run_dir)

    snapshot_dir = os.path.join(run_dir, "snapshot_data/") #creates directory to store snapshot data of run
    try:
        os.mkdir(snapshot_dir)
        print ("Successfully created the directory %s " % snapshot_dir)
    except OSError:
        print ("Creation of the directory %s failed" % snapshot_dir)
        
    return exp_dir, run_dir, snapshot_dir