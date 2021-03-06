import numpy as np

def calculate_fourier(quantity, exp_data, L=70, min_wave_length=5, num_bins=12):
    """
    Computes the average power for different wavelengths and sorts into bins. Follows equation: $F(q_x, q_y) = \frac{1}{N} \sum_{i=1}^N exp(i q_x x_i + i q_y y_i) ) \cdot Quantity_i$, $P(q_x, q_y) = F(q_x, q_y) F(q_x, q_y)^*$, and returns $\Braket{P(q_x, q_y) }_{\sqrt{q_x^2 + q_y^2}=q}$.
    
    :param quantity: numpy array of shape [num of time snapshots, num of particles] to take fourier transform of.
    :param L: Length of simulation box.
    :param min_wave_length: Should be less than or equal to the mean radius of particles.
    """
    q_vals = (2*np.pi/L) * np.arange(0, np.floor(L/min_wave_length)) #q is wavevector, largest value should be less than 2pi/ min_wave_length
    q_x, q_y = np.meshgrid(q_vals, q_vals)
    q_mag = np.sqrt(q_x**2 + q_y**2)
    x_data = exp_data['x'] #shape: [num of time snapshots, num of particles]
    y_data = exp_data['y']
    Np = x_data.shape[-1] #Number of particles
    print(Np)
    
    """
    Reshape all arrays to [num of q_x, num of q_y, num of time snapshots, num of particles] (or dim size is 1).
    """
    q_x = np.reshape(q_x, (*q_x.shape, 1, 1))
    q_y = np.reshape(q_y, (*q_y.shape, 1, 1))
    x_data = np.reshape(x_data, (1, 1, *x_data.shape))
    y_data = np.reshape(y_data, (1, 1, *y_data.shape))
    quantity = np.reshape(quantity, (1, 1, *quantity.shape))
    
    """
    Calculate fourier transform of quantity and get average power of each wave vector.
    """
    # \tilde{Quantity} = \frac{1}{\sqrt{N}} \sum_{i=1}^N exp(i q_x x_i + i q_y y_i) ) \cdot Quantity_i
    quantity_fourier = (1/np.sqrt(Np)) * np.sum( np.exp(1j*(q_x*x_data+q_y*y_data)) * quantity , axis=-1) #shape: [num of q_x, num of q_y, num of time snapshots]
    quantity_fourier = np.abs(quantity_fourier)**2 #Get square of fourier coefficients
    #\Braket{ \tilde{Quantity} \cdot \tilde{Quantity}^* }_{time}
    quantity_fourier = np.average(quantity_fourier, axis=-1) #Average over time
    
    """
    Sort into bins based on wavelength, and take average of each bin.
    """
    bins = np.linspace(np.amin(q_mag.flatten()[1:]), np.amax(q_mag), num_bins+1) #minimum of q_mag should exclude (q_x,q_y)=(0,0)
    indices_of_bins = np.digitize(q_mag, bins) #Gets index of bin that each element of q_mag belongs to
    indices_of_bins[0,0] = -1 #Says zero fourier mode belongs to bin -1 (doesn't exist), so it's not counted
    avg_fourier_powers = []
    for i in range(num_bins+1):
        if i == 0:
            continue #By contruction, never any wavevectors before first or after last bin
        indices_for_bin_i = np.where(indices_of_bins==i)
        average_for_bin_i = np.average(quantity_fourier[indices_for_bin_i])
        avg_fourier_powers.append(average_for_bin_i)
    
    return np.array(avg_fourier_powers), bins