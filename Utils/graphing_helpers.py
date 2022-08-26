import os
import matplotlib.pyplot as plt
import numpy as np

def save_fig_pdf(save_filename):
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

def plot_phase_diagram(x_values, y_values, plot_values, save_filename = None, plt_xlabel = '', plt_ylabel = '', title = '', 
                        vmin=0, vmax=1, caption_line_1 = '', caption_line_2 = '', log_scale = True):
    fig, ax = plt.subplots()
    if log_scale:
        x_bins = edges_from_centers_log(x_values)
        y_bins = edges_from_centers_log(y_values)
        plt.xscale('log')
        plt.yscale('log')
    else:
        x_bins = edges_from_centers_linear(x_values)
        y_bins = edges_from_centers_linear(y_values)
    xedges, yedges = np.meshgrid(x_bins,y_bins)
    plot = ax.pcolormesh(xedges,yedges, plot_values, vmin=vmin, vmax=vmax, edgecolors='k')
    plt.colorbar(plot)
    plt.title(title)
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    fig.text(.5, -0.05, caption_line_1, ha='center', fontsize=12)
    fig.text(.5, -0.13, caption_line_2, ha='center', fontsize=12)
    if save_filename != None:
        save_fig_pdf(save_filename)

def plot_vicsek_phase_diagram(x_values, y_values, plot_values, save_filename = None, plt_xlabel = '', plt_ylabel = '',
                                title = 'Vicsek Phase Diagram', simulation_desc = '', log_scale = True):
    caption_line_1 = r"Vicsek Order Parameter = $\left| \sum \vec{n}_i \right| / N$"
    plot_phase_diagram(x_values, y_values, plot_values, save_filename = save_filename, plt_xlabel = plt_xlabel, 
                        plt_ylabel = plt_ylabel, title = title, vmin=0, vmax=1, caption_line_1 = caption_line_1, 
                        caption_line_2 = simulation_desc, log_scale = log_scale)

def plot_dir_cross_vel_phase_diagram(x_values, y_values, plot_values, save_filename = None, plt_xlabel = '', plt_ylabel = '',
                                title = 'Dir Cross Vel Phase Diagram', simulation_desc = '', log_scale = True):
    caption_line_1 = r"Director Cross Velocity = $\sum \left|\vec{n}_i \times \vec{v}_i \right| / (v_0 N)$"
    plot_phase_diagram(x_values, y_values, plot_values, save_filename = save_filename, plt_xlabel = plt_xlabel, 
                        plt_ylabel = plt_ylabel, title = title, vmin=0, vmax=1, caption_line_1 = caption_line_1, 
                        caption_line_2 = simulation_desc, log_scale = log_scale)

def plot_dir_cross_vel_norm_phase_diagram(x_values, y_values, plot_values, save_filename = None, plt_xlabel = '', plt_ylabel = '',
                                title = 'Dir Cross Vel Norm Phase Diagram', simulation_desc = '', vmin = 0, vmax = 1, log_scale = True):
    caption_line_1 = r"Director Cross Velocity Norm = $\sum \left|\vec{n}_i \times \hat{v}_i \right| / N$"
    plot_phase_diagram(x_values, y_values, plot_values, save_filename = save_filename, plt_xlabel = plt_xlabel, 
                        plt_ylabel = plt_ylabel, title = title, vmin=vmin, vmax=vmax, caption_line_1 = caption_line_1, 
                        caption_line_2 = simulation_desc, log_scale = log_scale)


