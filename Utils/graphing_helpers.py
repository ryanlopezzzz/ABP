"""
Useful functions for making histograms, since matplotlib requires bin edges, not centers.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

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