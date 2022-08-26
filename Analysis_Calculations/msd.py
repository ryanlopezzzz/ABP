"""
Computes MSD and MSD minus flocking, still need to implement so this data is saved.
"""

import numpy as np

"""
Simulation MSD can be calculated from displacement of particles, so consider the difference in positions at each time step. The periodic boundary conditions must be accounted for. Make the assumption that in 1 time step the particle will never traverse more than half of the boxes length in any dimension (or else actual displacement becomes unclear). The formula used for displacement in the x dimension $D_x$ going from position $x_i$ to $x_f$ is (accounting for periodic boundary conditions):

$D_x = \left[ \left(x_f-x_i+\frac{3L_x}{2} \right)\% L_x \right] - \frac{L_x}{2} $

Where $\%$ denotes the modulo operator. This formula can be verified by checking all cases. Same formula for y dimension.
"""
 
def get_msd(exp_data, L, msd_type = "normal"):
    position_data = np.stack((exp_data['x'],exp_data['y']), axis=2) #shape: [time steps, num of particles, 2 for x and y]
    nsteps = position_data.shape[0] #number of time steps recorded
    total_MSD = np.zeros(nsteps-1)
    
    for start_t_index in range(nsteps-1): #Use this for shifted time statistics
        """
        Roll and shrink roll_position_data with each iterations of for loop. If ndim=1 of position data:
        
        position_data = [0 1 2 3 4 5]
        
        After each iteration: 
        roll_position_data = [0 1 2 3 4 5]    (start_t_index=0)
        roll_position_data = [1 2 3 4 5]
        roll_position_data = [2 3 4 5]
        roll_position_data = [3 4 5]
        roll_position_data = [4 5]            (start_t_index=nsteps-1)
        """
        roll_position_data = np.roll(position_data,-1*start_t_index, axis=0)
        roll_position_data = roll_position_data[:nsteps-start_t_index,:,:] #rolls position data back and deletes excess
                
        """
        MSD calculation
        """
        x_f_minus_x_i = np.diff(roll_position_data, axis=0) #takes difference of position elements at different times    
        actual_disp = ((x_f_minus_x_i + (3*L/2)) % L) - (L/2) #account for periodic boundary conditions
        msd_disp = get_disp_for_msd_type(actual_disp, msd_type)    
        net_disp = np.cumsum(msd_disp, axis=0) #net vector displacement is the sum of the displacements at all previous times
        MSD_sim = np.sum(np.square(net_disp),axis=2) #Pythagorean theorem: MSD = \sum (\delta x_i^2)
        MSD_sim_ensemble = np.average(MSD_sim, axis=1) #Average MSD across all particles in ensemble
        
        padding = (0,start_t_index) #want to pad zeros to end of MSD calculation to make them all the same shape
        MSD_sim_ensemble = np.pad(MSD_sim_ensemble, padding, constant_values=0)
        total_MSD += MSD_sim_ensemble
    
    prefactor = np.divide(1,np.flip(np.arange(1,nsteps))) #1/num of time statistics for each time diff, [1/(N-1),...,1]
    total_MSD = prefactor * total_MSD
    
    return total_MSD

def get_disp_for_msd_type(actual_disp, msd_type):
    if msd_type == 'normal':
        return actual_disp
    
    elif msd_type == 'normal_minus_avg':
        avg_disp = np.average(actual_disp, axis=1)[:,None,:] #average over particles, shape: [time_steps, 1, 2 (x/y)]
        disp_minus_avg = actual_disp - avg_disp
        return disp_minus_avg
    
    elif msd_type == 'parallel':
        """
        Calculate MSD for displacement parallel to flocking
        """
        avg_disp = np.average(actual_disp, axis=1)[:,None,:] #average over particles, shape: [time_steps, 1, 2 (x/y)]
        disp_minus_avg = actual_disp - avg_disp
        disp_minus_avg_dot_avg_disp = dot_product(disp_minus_avg, avg_disp) #shape: [time_steps, num_particles, 1]
        avg_disp_dot_avg_disp = dot_product(avg_disp, avg_disp) #shape: [time_steps, 1, 1]
        parallel_disp = np.divide(disp_minus_avg_dot_avg_disp, np.sqrt(avg_disp_dot_avg_disp)) #shape:[time_steps,num_particles,1]
        return parallel_disp
    
    elif msd_type == 'perpendicular':
        """
        Calculate MSD for displacement perpendicular to flocking
        """
        avg_disp = np.average(actual_disp, axis=1)[:,None,:] #average over particles, shape: [time_steps, 1, 2 (x/y)]
        disp_minus_avg = actual_disp - avg_disp
        disp_minus_avg_cross_avg_disp = cross_product(disp_minus_avg, avg_disp) #shape: [time_steps, num_particles, 1]
        avg_disp_dot_avg_disp = dot_product(avg_disp, avg_disp) #shape: [time_steps, 1, 1]
        perp_disp = np.divide(disp_minus_avg_cross_avg_disp, np.sqrt(avg_disp_dot_avg_disp)) #shape:[time_steps, num_particles, 1]
        return perp_disp
    
    else:
        raise ValueError('msd_type given is not an allowed option.')

def dot_product(a, b):
    """
    Takes dot product along last axis of a and b and retains shape of first input (except last dimension becomes 1)
    """
    dot_product = np.sum(a[...,:] * b[...,:], axis=-1, keepdims = True)
    return dot_product

def cross_product(a, b):
    """
    Takes cross product along last axis of a and b, dot with z hat, keep last dimension.
    """
    cross_product = np.cross(a, b)[...,None]
    return cross_product