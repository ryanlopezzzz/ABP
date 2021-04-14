#Creates directories for saving experiment data

import os
import datetime

def create(save_dir, exp_folder_name, load_date, save_python_file = False):

    exp_dir = os.path.join(save_dir, exp_folder_name + "/") #Directory where all data for particular experiment type is stored

    if not os.path.isdir(exp_dir): #Check if exp_dir is already a directory
        try:
            os.mkdir(exp_dir)
            print ("Successfully created the directory %s " % exp_dir)
        except OSError:
            print ("Creation of the directory %s failed" % exp_dir)

    #Create folder or connect to old folder to store run data

    if load_date is None: #creates new directory with date as folder name to save run information
        currentDT = datetime.datetime.now() 
        run_folder_name = "%d-%d-%d--%d-%02d-%02d"%(currentDT.month, currentDT.day, currentDT.year, currentDT.hour, currentDT.minute, currentDT.second)

        run_dir = os.path.join(exp_dir, run_folder_name) #folder where this run data is stored

        try:
            os.mkdir(run_dir)
            print ("Successfully created the directory %s " % run_dir)
        except OSError:
            print ("Creation of the directory %s failed" % run_dir)

    else: #connects to old run's path if specified in load_date
        run_dir = os.path.join(exp_dir, load_date)

        if os.path.isdir(run_dir):
            print("Successfully connected to previous run %s" % run_dir)
        else:
            print("Connection to previous run %s failed" % run_dir)

    snapshot_dir = os.path.join(run_dir, "snapshot_data/") #creates directory to store snapshot data of run

    if load_date is None:
        try:
            os.mkdir(snapshot_dir)
            print ("Successfully created the directory %s " % snapshot_dir)
        except OSError:
            print ("Creation of the directory %s failed" % snapshot_dir)

    save_python_file = False #Makes copy of jupyter notebook and saves in run_dir if True. Useful if you want to look back at code and have the space

    if save_python_file:
        cur_dir = os.getcwd()

        src = cur_dir + '/ABP_Simulation.ipynb'
        dst = run_dir + '/ABP_Simulation.ipynb'

        shutil.copyfile(src, dst)
        
    return exp_dir, run_dir, snapshot_dir