"""
Reads saved simulation data in .txt format and convers to dictionary of numpy arrays.
"""

import os
import numpy as np

#returns exp_data, a dictionary with keys ['id', 'x', 'y', 'ipx', 'ipy', 'nx', 'ny', 'vx', 'vy']
#exp_data dict values are arrays of shape [num of time snapshots, num of particles]
def get_exp_data(snapshot_dir):

    #Get snapshot data file names:
    file_names = []
    for f in os.listdir(snapshot_dir):
        if f.endswith(".txt"):
            file_names.append(os.path.join(snapshot_dir, f))
    file_names = sorted(file_names) #sort in temporal order
    
    #Gets names of observable quantities saved:
    first_file = open(file_names[0],'r')
    var_names = first_file.readline().split() #Names of quantities listed in first line of .txt document with spaces
    var_names = var_names[1:] #There is a "#" at beginning of first line, not an observable
    
    
    total_snapshots = len(file_names)
    Np = np.loadtxt(file_names[0]).shape[0] #number of particles
    data = np.zeros((len(var_names) , total_snapshots , Np)) #Size: [num of obervables, num of snapshots, num of particles]

    for (t,name) in enumerate(file_names): #each different file corresponds to different time
        var_vals = np.loadtxt(name).T #shape: [num of observables, num of particles]
        data[:,t,:] = var_vals
    
    exp_data = {} #reorganize into python dictionary
    for obs in range(len(var_names)):
        exp_data[var_names[obs]] = data[obs]
    
    return exp_data

def get_snapshot_data(snapshot_filename):
    #returns a dictionary with keys ['id', 'x', 'y', 'ipx', 'ipy', 'nx', 'ny', 'vx', 'vy']
    #exp_data dict values are of shape [num of particles]
    with open(snapshot_filename, 'r') as snapshot_file:
        variable_names = snapshot_file.readline().split()
        variable_names = variable_names[1:] #There is a "#" at beginning of first line, not an observable
    data = np.loadtxt(snapshot_filename).T #shape: num variables, num_particles
    snapshot_data = {} #reorganize into python dictionary
    for variable_index, variable_name in enumerate(variable_names):
        snapshot_data[variable_name] = data[variable_index,:]
    return snapshot_data