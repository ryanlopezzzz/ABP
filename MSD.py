import numpy as np

def get_MSD_sim_data(position_data, L):
    nsteps = position_data.shape[0] #number of time steps recorded
    
    MSD_sim_data = np.zeros((nsteps)) #The simulation ensemble MSD at each time step

    x_i = position_data

    x_f = np.roll(x_i,-1,axis=0) #this rolls along time dimension so x_f[i,:,:] = x_i[i+1,:,:]

    x_f_minus_x_i = (x_f-x_i)[:-1] #we don't want to include the displacement at final time which is x(t=0) - x(t=tf)

    disp = ((x_f_minus_x_i + (3*L/2)) % L) - (L/2) #vector displacement at each time step for each particle, see formula above      

    net_disp = np.cumsum(disp, axis=0) #perform cumulative sum along time axis, so the net vector displacement is the sum of the displacements at all previous times

    MSD_sim = np.sum(np.square(net_disp),axis=2) #gets MSD for each particle by pythagorean theorem: MSD = \sum (\delta x_i^2)

    MSD_sim_ensemble = np.average(MSD_sim, axis=1) #Average MSD across all particles in ensemble

    return MSD_sim_ensemble, MSD_sim
    #Add in time average as well!