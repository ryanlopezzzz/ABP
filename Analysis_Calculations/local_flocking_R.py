import resource
gigabyte = int(1e9)
resource.setrlimit(resource.RLIMIT_AS, (49*gigabyte, 50*gigabyte))
import numpy as np

def get_local_flocking_R(exp_data, particle_diameter, box_length):
    """
    Calculates R as defined in Lorenzo Caprini "Hidden velocity ordering in dense suspensions of self-propelled disks,
    except it measures the alignment of direction instead of velocity.

    Note: sigma is defined as average particle diameter
    """

    x_data = exp_data['x'] #shape: [num of time snapshots, num of particles]
    y_data = exp_data['y']
    nx_data = exp_data['nx']
    ny_data = exp_data['ny']
    theta_data = np.arctan2(nx_data, ny_data) #director angle of each particle
    num_time_snaps = x_data.shape[0]
    num_particles = x_data.shape[1]

    """
    To utilize numpy vectorization, I will make all 'basic' quantities into np arrays whose shape is broadcastable with
    [num of time snapshots (t), num of particles (i), num of k values (k), num of particles (j)]
    """

    # We consider circular crowns of mean radius k * particle_diameter and thickness particle_diameter
    # We limit the max circular crown radius to be half the boxes length due to periodic boundary conditions
    min_k = 1
    max_k = int(np.floor(box_length / (2*particle_diameter)))
    k_values = np.linspace(min_k, max_k, num=max_k) #k_values = {1, ..., max_k}
    num_k = max_k
    
    crown_centers = particle_diameter * k_values
    crown_centers = np.reshape(crown_centers, (1,1,num_k,1)) 
    crown_inner_radii = crown_centers - particle_diameter / 2
    crown_outer_radii = crown_centers + particle_diameter / 2

    #Get the distance between particles with periodic boundary conditions
    xi_data = np.reshape(x_data, (num_time_snaps,num_particles,1,1))
    yi_data = np.reshape(y_data, (num_time_snaps,num_particles,1,1))
    xj_data = np.reshape(x_data, (num_time_snaps,1,1,num_particles))
    yj_data = np.reshape(y_data, (num_time_snaps,1,1,num_particles))

    xi_minus_xj = xi_data - xj_data # xi_minus_xj[t][i][k][j] = x_data[t][i] - x_data[t][j]
    xi_minus_xj = (  ( xi_minus_xj+(3*box_length/2) ) % box_length  )-(box_length/2) #accounts for periodic boundary conditions, see MSD calculation
    yi_minus_yj = yi_data - yj_data
    yi_minus_yj = (  ( yi_minus_yj+(3*box_length/2) ) % box_length  )-(box_length/2) 
    
    mag_ri_minus_rj = np.sqrt(xi_minus_xj**2 + yi_minus_yj**2) #mag_ri_minus_rj[t][i][k][j] = | \vec{r}[t][i] - \vec{r}[t][j] |

    #Define a variable called is_in_crown[t][i][k][j] = True if at time t, the particle j is in the crown of mean radius k * particle_diameter centered at particle i
    #and i does not equal j, and it equals False otherwise
    same_ij_index = np.reshape(np.identity(num_particles), (1,num_particles,1,num_particles)) #same_ij_index[t][i][k][j] = True (1) if i=j, False (0) otherwise
    is_in_crown = (mag_ri_minus_rj < crown_outer_radii) & (mag_ri_minus_rj > crown_inner_radii) & np.logical_not(same_ij_index)

    #Define variable d_ij[t][i][k][j] = min{ |\theta_i(t)-\theta_j(t)|, 2pi -  |\theta_i(t)-\theta_j(t)|} if is_in_crown[t][i][k][j]=True, 0 otherwise
    thetai = np.reshape(theta_data, (num_time_snaps,num_particles,1,1))
    thetaj = np.reshape(theta_data, (num_time_snaps,1,1,num_particles))
    mag_thetai_minus_thetaj = np.abs(thetai - thetaj)
    d_ij = np.minimum(mag_thetai_minus_thetaj, 2*np.pi - mag_thetai_minus_thetaj)
    d_ij = d_ij * is_in_crown

    #Now we calculate R
    N_k = np.sum(is_in_crown, axis=3) #sum all particles j in the crown
    with np.errstate(divide='ignore',invalid='ignore'): #when N_k=0, should set Q_i to zero
        Q_i = 1 - (2/(np.pi*N_k)) *  np.sum(d_ij, axis=3) #shape: [num of time steps, num of particles, num of k vals]
        Q_i = np.nan_to_num(Q_i)
    Q = np.sum(Q_i, axis=1) / num_particles #shape: [num of time steps, num of k vals]
    R = np.sum(Q, axis=1) * particle_diameter #shape: [num of time steps]
    R_avg = np.average(R) 

    return R_avg, Q


