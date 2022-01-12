#Creates directories for saving experiment data
import os

def get_dir_names(save_dir, exp_folder_name, run_folder_name):
    exp_dir = os.path.join(save_dir, exp_folder_name + "/") #Directory where all data for particular experiment type is stored
    run_dir = os.path.join(exp_dir, run_folder_name) #folder where this run data is stored
    snapshot_dir = os.path.join(run_dir, "snapshot_data/") #creates directory to store snapshot data of run
    return exp_dir, run_dir, snapshot_dir

def create(exp_dir, run_dir, snapshot_dir):
    if not os.path.isdir(exp_dir):
        try: #create exp_dir
            os.mkdir(exp_dir)
            print ("Successfully created the directory %s " % exp_dir)
        except OSError:
            print ("Creation of the directory %s failed" % exp_dir)
    try: #create run_dir
        os.mkdir(run_dir)
        print ("Successfully created the directory %s " % run_dir)
    except OSError:
        print ("Creation of the directory %s failed" % run_dir)
    try: #create snapshot_dir
        os.mkdir(snapshot_dir)
        print ("Successfully created the directory %s " % snapshot_dir)
    except OSError:
        print ("Creation of the directory %s failed" % snapshot_dir)