import numpy as np
from scipy.signal import find_peaks, peak_prominences
import sys
sys.path.append('../../')
from Utils.graphing_helpers import edges_from_centers_linear

def get_local_packing_fraction(exp_data, num_bins_along_dim, L, particle_area):
    """
    Calculate local packing fraction by splitting simulation box into bins and counting number of paricles in each bin.

    :param num_bins_along_dim: Number of bins along x and y dimensions, so total bins is squared of this value
    :param L: Length of the box
    :param particle_ara: Area of a single particle (polydispersity not supported)
    """
    exp_data_x = exp_data['x'] #shape: [num of time snapshots, num of particles]
    exp_data_y = exp_data['y']
    num_time_snapshots = exp_data_x.shape[0]
    box_range = np.array([[-L/2, L/2], [-L/2, L/2]]) #simulation box range
    bin_area = (L/ num_bins_along_dim)**2

    local_packing_fraction = np.zeros((num_time_snapshots, num_bins_along_dim, num_bins_along_dim))
    for snapshot in range(num_time_snapshots):
        histogram2d, x_edges, y_edges = np.histogram2d(exp_data_x[snapshot], exp_data_y[snapshot], bins=num_bins_along_dim, range=box_range)
        histogram2d = histogram2d.T #transpose since doesn't follow x-y axis convention
        local_packing_fraction_snapshot = (particle_area/bin_area)*histogram2d
        local_packing_fraction[snapshot] = local_packing_fraction_snapshot
    return local_packing_fraction, x_edges, y_edges

def get_packing_bins_for_hist(local_packing_fraction, num_bins_along_dim, L, particle_area):
    #Each bin in hist corresponds to adding single particle to box
    bin_area = (L/num_bins_along_dim)**2
    delta_packing_fraction = particle_area / bin_area
    bin_centers = delta_packing_fraction*np.arange(0, np.ceil(np.max(local_packing_fraction)/delta_packing_fraction)+1)
    return bin_centers

def get_peaks_info(local_packing_fraction, num_bins_along_dim, L, particle_area):    
    """
    Calculates one or two peaks of local packing histogram.
    """
    bin_centers = get_packing_bins_for_hist(local_packing_fraction, num_bins_along_dim, L, particle_area)
    bin_edges = edges_from_centers_linear(bin_centers)
    #Calculate histogram
    hist_values, _ = np.histogram(local_packing_fraction.flatten(), bins=bin_edges, density=True)
    peaks_indices, _ = find_peaks(hist_values, prominence=0.05)
    prominences, _, _ = peak_prominences(hist_values, peaks_indices)
    largest_peaks_indices = peaks_indices[np.argsort(-prominences)[:2]]
    return largest_peaks_indices, bin_centers

def get_packing_hist_peak_distance(local_packing_fraction, num_bins_along_dim, L, particle_area):
    largest_peaks_indices, bin_centers = get_peaks_info(local_packing_fraction, num_bins_along_dim, L, particle_area)
    if len(largest_peaks_indices) == 1:
        peak_distance = 0
    else:
        peak_distance = np.abs(bin_centers[largest_peaks_indices[1]]-bin_centers[largest_peaks_indices[0]])
    return peak_distance

def get_packing_mean_and_std_dev(local_packing_fraction, num_bins_along_dim, L, particle_area):
    """
    Calculates one or two peaks of local packing histogram.
    """
    bin_centers = get_packing_bins_for_hist(local_packing_fraction, num_bins_along_dim, L, particle_area)
    bin_edges = edges_from_centers_linear(bin_centers)
    #Calculate histogram
    hist_values, _ = np.histogram(local_packing_fraction.flatten(), bins=bin_edges, density=True)
    #Compute mean and std dev
    mean = np.average(bin_centers, weights=hist_values)
    variance = np.average((mean-bin_centers)**2, weights=hist_values)
    std_dev = np.sqrt(variance)
    return mean, std_dev, bin_centers