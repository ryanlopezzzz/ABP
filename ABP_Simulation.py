#!/usr/bin/env python
# coding: utf-8

# # Implementing a 2D Simulation of Active Brownian Particles

# In[1]:


#Imports

import sys
import os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import datetime
import pdb #python debugger
from timeit import default_timer as timer #timer
from collections import OrderedDict
import argparse

sys.path.insert(1, '/Users/ryanlopez/ABPTutorial/c++') #Connects to ABP Folder github.com/ryanlopezzzz/ABPTutorial
from cppmd.builder import *
import cppmd as md

import read_data as rd #reads snapshot text data
import directories #used to create directories for saving data
import Physical_Quantities.MSD as MSD
import Physical_Quantities.flocking_factors as flocking_factors
import Physical_Quantities.various as various


# ## ABP Physics
# 
# ABPs are described by the coupled Langevin Equations:
# <br>
# <br>
# $\dot{r}_i = v_0 \hat{n} + \mu \sum_j F_{ij} + \xi_i^t$
# <br>
# $\dot{\theta}_i = \xi_i^r$
# <br>
# <br>
# Where both $\xi^t$ and $\xi^r$ are Gaussian white noise which satisfy:
# <br>
# <br>
# $<\xi_i^t \xi_j^t> = 2D_t \delta_{ij}$
# <br>
# $<\xi_i^r \xi_j^r> = 2D_r \delta_{ij}$
# <br>
# <br>
# The relevant physics of the ABPs is determined through 4 parameters:
# 
# $Pe_r = \frac{v_0}{D_r a}$ (Peclet number, $a$ is the particle's radius)
# <br>
# $\mu = 1$ (Particle Mobility)
# 

# ### Dimensionless quantities I should use:
# Packing fraction
# 
# Peclet Number $Pe_r = \frac{v_0 \tau_r}{a}=\frac{v_0}{D_r a}$
# 
# Time scale $\tau_k = \frac{1}{\mu k}$
# 
# Mobility $\mu = \frac{1}{\gamma_t}$
# 
# Fluctuation-Dissipation Theorem: $D_t = k T \mu$

# In[2]:


#Physical parameters:

gamma_t = 1.0 #translational friction coefficient
gamma_r = 1.0 #rotational friction coefficient

alpha = 0.03 #Active self propulsion force, v_0 = alpha/gamma_t
kT = 0 #temperature of the system, typically set to zero

param_search = True

if not param_search:
    packing_frac = 1 #rotation diffusion coefficient
else:
    parser = argparse.ArgumentParser();
    parser.add_argument('--param', help='Varied parameter value for the run.', default=None)
    args = parser.parse_args()
    
    packing_frac = float(args.param)

#Calculate some parameters:
D_r = 0.2
v0 = alpha/gamma_t #self propulsion speed
mu = 1/gamma_t #mobility
D_t = kT * mu #translational diffusion coefficient #this comes from Fluctuation-Dissipation Theorem
kT_rot = D_r * gamma_r #rotational temperature, see brownian_rot_integrator.py

#Box parameters:
phi = packing_frac/np.pi #particle number density, phi = N/L^2  # divide by np.pi when r=1
L = 70 #Simulation box side length, makes square
radius = 1 #Radius of particles if poly=0 #In initial config, center of particles are not closer than a together?
poly = 0.3 #Implements polydispersity in particle, r_i = R * (1+ poly * uniform(-0.5, .0.5) )

Np = int(round(phi*L**2)) #number of particles
approx_packing = np.pi * radius**2 * Np / (L**2) #Note this is not true when poly =/= 0

#Optional Forces:
apply_soft_repulsive_force = True #applies soft harmonic force if particles are overlapping
k = 1 #spring constant for harmonic force
apply_velocity_alignment = True #Applies torque to each particle based on relative alignment with neighbors
J = 1

#Integration parameters:

warm_up_time = 1e4 #1e0 #Run simulation for this amount of time to reach steady state
tf = 1e4 #time to run simulation while logging physical quantities
tstep = 1e-1 #Time step size for integration
rand_seed = random.randint(1,10000) #random seed used for Brownian integration

warm_up_nsteps = int(warm_up_time / tstep)
nsteps = int(tf / tstep) #total number of time steps

print("Packing Fraction: " + str(approx_packing) + "\n")
print("Number of particles: " + str(Np) + "\n \n")
print("Orientational Correlation Time: " + str(1/D_r if D_r !=0 else "infinity") + "\n")
print("Interaction time: " + str(1/(mu*k))+"\n")
#print("Mean free time between collisions: " + str(L**2 /(2*radius*v0*Np)))


# # Folders
# 
# All research data is contained in a directory with path variable <em>save_dir</em>
# <br>
# <br>
# Within <em>save_dir</em>, there are different sub-directories corresponding to different types of experiments, with path variable <em>exp_dir</em>
# <br>
# <br>
# Within <em>exp_dir</em>, there are different sub-directories corresponding to different specific runs of the experiment, with path variable <em>run_dir</em>. These folders contain information about the specific run, and are named automatically by the date-time it was first run. Example: "2-24-2021--22-15-52" corresponds to 2/24/2021 at 10:15:52PM

# In[3]:


#Directory where all data is saved
save_dir = "/Users/ryanlopez/Desktop/Python_Programs/Dr_Marchetti_Research/Saved_Data"


# In[4]:


exp_folder_name = "Vary_phi_and_Dr=0.2" #Folder name of experiment directory, don't change inbetween runs unless studying something different

load_date = None #Enter date in format 2-24-2021--22-15-52 (2/24/2021 at 10:15:52PM) to connect to previous run
#If load_date = None, will start new experiment


# In[5]:


name = "D_r=%.4f_and_packing_frac=%.2f"%(D_r,approx_packing)
exp_dir, run_dir, snapshot_dir = directories.create(save_dir, exp_folder_name, load_date, name=name)


# In[6]:


run_desc = OrderedDict()

run_desc['gamma_t'] = gamma_t
run_desc['gamma_r'] = gamma_r
run_desc['alpha'] = alpha
run_desc['kT'] = kT
run_desc['D_r'] = D_r
run_desc['v0'] = v0
run_desc['mu'] = mu
run_desc['D_t'] = D_t
run_desc['kT_rot'] = kT_rot
run_desc['phi'] = phi
run_desc['L'] = L
run_desc['radius'] = radius
run_desc['poly'] = poly
run_desc['Np'] = Np
run_desc['approx_packing'] = approx_packing
run_desc['k'] = k
run_desc['J'] = J
run_desc['warm_up_time'] = warm_up_time
run_desc['tf'] = tf
run_desc['tstep'] = tstep
run_desc['rand_seed'] = rand_seed
run_desc['warm_up_nsteps'] = warm_up_nsteps
run_desc['nsteps'] = nsteps

def write_desc():
    run_desc_file = open(os.path.join(run_dir, "run_desc.json"), 'w')
    run_desc_file.write(json.dumps(run_desc))
    run_desc_file.close()
write_desc()


# In[7]:


#creates random initial configuration, saves config to outfile
random_init(phi, L, radius = radius, rcut=0, poly = poly, outfile=os.path.join(run_dir, 'init.json'))


# In[8]:


reader = md.fast_read_json(os.path.join(run_dir, 'init.json'))  #here we read the json file in c++
system = md.System(reader.particles, reader.box)

dump = md.Dump(system)          # Create a dump object

evolver = md.Evolver(system)    # Create a system evolver object

#add the forces and torques

# Create pairwise repulsive interactions with the spring contant k = 10 and range a = 2.0
if apply_soft_repulsive_force:
    evolver.add_force("Soft Repulsive Force", {"k": k})
    #evolver.add_force("Harmonic Force", {"k": k})
    
# Create self-propulsion, self-propulsion strength alpha
evolver.add_force("Self Propulsion", {"alpha": alpha})

# Create pairwise polar alignment with alignment strength J = 1.0 and range a = 2.0
if apply_velocity_alignment:
    evolver.add_torque("Velocity Align", {"k": J})

#Add integrators

# Integrator for updating particle position, friction gamma = 1.0 , "random seed" seed = 10203 and no thermal noise
evolver.add_integrator("Brownian Positions", {"T": kT, "gamma": gamma_t, "seed": rand_seed})

# Integrator for updating particle orientation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0, "random seed" seed = 10203
evolver.add_integrator("Brownian Rotation", {"T": kT_rot, "gamma": gamma_r, "seed": rand_seed})

evolver.set_time_step(tstep) # Set the time step for all the integrators


# In[9]:


#warms up simulation to reach steady state
for t in range(warm_up_nsteps):
    evolver.evolve()
print("Warm up time complete")


# In[10]:


total_snapshots = 100 #total number of snapshots to save
print("Saving observables every %s time steps"%(int(nsteps/total_snapshots)))

#simulation while logging quantities:
for t in range(nsteps):
    if t % int(nsteps/10) == 0:
        print("Time step : ", t)
    evolver.evolve()    # Evolve the system by one time step
    if t % int(nsteps/total_snapshots) == 0:     #Save snapshot of the observable data
        snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:08d}.txt'.format(t))
        dump.dump_data(snapshot_file_path) #Saves data in .txt file
        snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:08d}.vtp'.format(t))
        dump.dump_vtp(snapshot_file_path) #Saves data in .vtp file for visualization
print("done")


# In[11]:


exp_data = rd.get_exp_data(snapshot_dir)
position_data = rd.get_position_data(snapshot_dir)


# In[12]:


vicsek_param, vel_param = flocking_factors.get_flocking_factors(exp_data, v0)

#plt.plot(vicsek_param)
#plt.plot(vel_param)

MSD_sim_ensemble, _ = MSD.get_MSD_sim_data(position_data, L)
dir_dot_vel, dir_dot_vel_norm = various.get_dir_dot_vel(exp_data)
_, v_mag_data = various.get_vel_mag_distr(exp_data)


# In[15]:


run_desc['vicsek_param'] = np.average(vicsek_param)
run_desc['vel_param'] = np.average(vel_param)
np.save(os.path.join(run_dir, "MSD_sim_ensemble.npy"), MSD_sim_ensemble)
np.save(os.path.join(run_dir, "dir_dot_vel.npy"), dir_dot_vel)
np.save(os.path.join(run_dir, "dir_dot_vel_norm.npy"), dir_dot_vel_norm)
np.save(os.path.join(run_dir, "v_mag_data.npy"), v_mag_data)

write_desc()

