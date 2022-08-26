import sys
import os
import numpy as np
from scipy.optimize import curve_fit
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import pdb #python debugger
from timeit import default_timer as timer #timer
from collections import OrderedDict
import importlib
sys.path.insert(1,'/home/ryanlopez/ABPTutorial/c++') #Connects to ABP Folder github.com/ryanlopezzzz/ABPTutorial
from cppmd.builder import *
import cppmd as md
import read_data as rd #reads snapshot text data
import directories #used to create directories for saving data

def save_fig_pdf(base_filename, extra_label):
    save_filename = os.path.join(base_filename, extra_label)
    plt.savefig(save_filename, format='pdf', bbox_inches='tight')

def edges_from_centers_linear(centers):
    """
    Returns bin edges for histogram given array of centers spaced linearly
    
    Input centers: [a, a+1*b, a+2*b, ..., a+ (n-1)*b] (linearly spaced centers)
    Output edges: [a-b/2, a+b/2, a+3b/2, ..., a+(n-1/2)*b] (linearly spaced bin edges which surround centers)
    """
    a = centers[0]
    b = centers[1]-centers[0]
    n = len(centers)
    
    edges_first_value = a-b/2
    edges_last_value = a+(n-1/2)*b
    edges_length = n+1
    
    edges = np.linspace(edges_first_value, edges_last_value, edges_length)
    return edges

def edges_from_centers_log(centers):
    """
    Returns bin edges for histogram given array of center spaced linearly
    """
    centers_log = np.log(centers)
    edges_log = edges_from_centers_linear(centers_log)
    edges = np.exp(edges_log)
    
    return edges

Dr_vals = np.logspace(-3,0,num=4)
Jv_vals = np.logspace(-3,0,num=4)
Dr_bins = edges_from_centers_log(Dr_vals)
Jv_bins = edges_from_centers_log(Jv_vals)

folder_names = ['phi=0.4000_and_v0=0.0100', 'phi=0.4000_and_v0=0.0300', 'phi=0.4000_and_v0=0.1000', 
    'phi=0.6000_and_v0=0.0100', 'phi=0.6000_and_v0=0.0300', 'phi=0.6000_and_v0=0.1000', 
    'phi=0.8000_and_v0=0.0100', 'phi=0.8000_and_v0=0.1000', 'phi=0.8000_and_v0=0.0300',
    'phi=1.0000_and_v0=0.0100', 'phi=1.0000_and_v0=0.0300', 'phi=1.0000_and_v0=0.1000'
                ]
save_dir = "/home/ryanlopez/Polar_Align_Vary_Phi_V_Saved_Data2/"
phase_diagram_dir = os.path.join(save_dir, 'phase_diagrams')
vicsek_image_dir = os.path.join(save_dir, 'phase_diagrams', 'vicsek_order_param')
dir_cross_vel_image_dir = os.path.join(save_dir, 'phase_diagrams', 'dir_cross_vel_norm')
dir_cross_vel_same_image_dir = os.path.join(save_dir, 'phase_diagrams', 'dir_cross_vel_norm_same_color_bar')
if not os.path.isdir(phase_diagram_dir):
    os.mkdir(phase_diagram_dir)
if not os.path.isdir(vicsek_image_dir):
    os.mkdir(vicsek_image_dir)
if not os.path.isdir(dir_cross_vel_image_dir):
    os.mkdir(dir_cross_vel_image_dir)
if not os.path.isdir(dir_cross_vel_same_image_dir):
    os.mkdir(dir_cross_vel_same_image_dir)

for folder_name in folder_names:
    phi = float(folder_name[4:10])
    v0 = float(folder_name[18:24])
    vicsek_values = np.full([len(Dr_vals), len(Jv_vals)], 100, dtype = float) #fill with 100 so can see if an index not replaced
    dir_cross_vel_norm_values = np.full([len(Dr_vals), len(Jv_vals)], 100, dtype = float) 
    
    exp_dir = os.path.join(save_dir, folder_name)
    for f in os.scandir(exp_dir): #runs through all immediate subdirectories
        if os.path.isdir(os.path.join(f, "snapshot_data/")):
            run_desc_file = open(os.path.join(f, 'run_desc.json'))
            run_desc = json.loads(run_desc_file.read())
            run_desc_file.close()
            single_stats_file = open(os.path.join(f, 'single_stats.json'))
            single_stats = json.loads(single_stats_file.read())
            single_stats_file.close()
            Jv_index = np.digitize(run_desc['J'], Jv_bins)-1
            Dr_index = np.digitize(run_desc['D_r'], Dr_bins)-1
            vicsek_values[Dr_index, Jv_index] = single_stats['vicsek_param']
            dir_cross_vel_norm_values[Dr_index, Jv_index] = single_stats['dir_cross_vel_norm']
   
    simulation_desc = r"$\phi=%.2f$, $v_0=%.1e$, k=1, L=70, Simulation Time = $5 \times 10^{4}$"%(phi, v0)
    plt_xlabel = r'$J$'
    plt_ylabel = r'$D_r$'

    #plot vicsek param
    fig, ax = plt.subplots()
    xedges, yedges = np.meshgrid(Jv_bins,Dr_bins)
    plot = ax.pcolormesh(xedges,yedges,vicsek_values, vmin=0, vmax=1)
    plt.colorbar(plot)
    plt.title("Vicsek Parameter Phase Diagram")
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.xscale('log')
    plt.yscale('log')
    fig.text(.5, -0.05, r"Vicsek Order Parameter = $\left| \sum \vec{n}_i \right| / N$", ha='center', fontsize=12)
    fig.text(.5, -0.13, simulation_desc, ha='center', fontsize=12)
    save_fig_pdf(vicsek_image_dir, '%s.pdf'%folder_name)

    #plot dir cross vel norm
    fig, ax = plt.subplots()
    xedges, yedges = np.meshgrid(Jv_bins,Dr_bins)
    plot = ax.pcolormesh(xedges,yedges,dir_cross_vel_norm_values)
    plt.colorbar(plot)
    plt.title("Dir cross Vel Norm Phase Diagram")
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.xscale('log')
    plt.yscale('log')
    fig.text(.5, -0.05, r"Director Cross Velocity Norm = $\sum \left|\vec{n}_i \times \hat{v}_i \right| / N$", ha='center', fontsize=12)
    fig.text(.5, -0.13, simulation_desc, ha='center', fontsize=12)
    save_fig_pdf(dir_cross_vel_image_dir, '%s.pdf'%folder_name)

    #plot with same color range
    fig, ax = plt.subplots()
    xedges, yedges = np.meshgrid(Jv_bins,Dr_bins)
    plot = ax.pcolormesh(xedges,yedges,dir_cross_vel_norm_values, vmin=0, vmax=0.15)
    plt.colorbar(plot)
    plt.title("Dir cross Vel Norm Phase Diagram")
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.xscale('log')
    plt.yscale('log')
    fig.text(.5, -0.05, r"Director Cross Velocity Norm = $\sum \left|\vec{n}_i \times \hat{v}_i \right| / N$", ha='center', fontsize=12)
    fig.text(.5, -0.13, simulation_desc, ha='center', fontsize=12)
    save_fig_pdf(dir_cross_vel_same_image_dir, '%s.pdf'%folder_name)

    plt.close('all')
print('Finished saving all diagrams')