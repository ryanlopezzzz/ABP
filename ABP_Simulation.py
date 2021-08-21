#!/usr/bin/env python
# coding: utf-8

# # 2D Simulation of Active Brownian Particles with Velocity Alignment

# In[34]:


import sys
import os
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import datetime
import pdb #python debugger
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


# ## Active Brownian Particle (ABP) Physics
# 
# We consider a system of active colloids with velocity alignment in 2D. The dynamics are described by the coupled Langevin Equations:
# <br>
# <br>
# $\dot{\textbf{r}}_i = v_0 \hat{\textbf{n}} + \mu \sum_j \textbf{F}_{ij} +\xi_i^t$
# <br>
# $\dot{\theta}_i = J(\textbf{v}_i \times \hat{\textbf{n}}) \cdot \hat{\textbf{z}} +  \xi_i^r$
# <br>
# <br>
# Where both $\xi^t$ and $\xi^r$ are Gaussian white noise which satisfy:
# <br>
# <br>
# $<\xi_i^t(t) \xi_j^t (t')> = 2D_t \delta_{ij} \delta(t-t') $
# <br>
# $<\xi_i^r (t) \xi_j^r (t')> = 2D_r \delta_{ij} \delta(t-t')$
# <br>
# <br>
# We restrict our system to be at zero temperature ($D_t=0$) with unit mobility ($\mu = 1$) and harmonic pair repulsion forces when particles overlap $\textbf{F}_{ij} = -k(R_i+R_j-r_{ij})\hat{\textbf{r}}_{ij}$
# 
# 
# 

# In[52]:


"""
Throughout all simulations there are only four variables which are usually varied

    J            :   Orientational Torque Coefficient
    D_r          :   Rotational Diffusion Rate
    v0           :   Active Self Propulsion Speed
    packing_frac :   Approx. Packing Fraction (Defined as \pi * <R>^2 * Num particles / Area of Box)
"""
param_search = False #If True, params for run are determined from command line (Good for trying lots of params)

if param_search:
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', nargs='+', help='Varied parameter value for the run.', default=None)
    args = parser.parse_args()
    
    J = float(args.params[0]) #This variable determined by 
    D_r = float(args.params[1])

J=2
D_r=0.03
v0=0.15 #np.sqrt(2 * gamma_t * D_r * Teff)
packing_frac=0.6 


# In[55]:


"""
Assigned physical parameters:
"""
gamma_t = 1.0 #translational friction coefficient
gamma_r = 1.0 #rotational friction coefficient
kT = 0 #temperature of the system, typically set to zero
radius = 1 #Average radius of particles
poly = 0.3 #Implements polydispersity in particle, r_i = R * (1+ poly * uniform(-0.5, .0.5) )
k = 1 #spring constant for harmonic force
L = 70 #Simulation box side length

"""
Calculated physical paramters:
"""
Teff = 0.5 * gamma_t * v0**2 / D_r
alpha = v0 * gamma_t 
mu = 1/gamma_t #mobility
D_t = kT * mu #translational diffusion coefficient, this comes from Fluctuation-Dissipation Theorem
kT_rot = D_r * gamma_r #rotational temperature, see brownian_rot_integrator.py
phi = packing_frac / (np.pi*radius**2) #particle number density, phi = N/(L^2 * <R^2>)
Np = int(round(phi*L**2)) #number of particles

"""
Simulation computation parameters:
"""
warm_up_time = 8e3 #Run simulation for this amount of time to reach steady state
tf = 2e3 #Time to run simulation while logging physical quantities, after reach steady state
tstep = 1e-2 #Time step size for integration
warm_up_nsteps = int(warm_up_time / tstep) #number of integration steps for warming up
nsteps = int(tf / tstep) #number of int. steps for logging physical quantites
rand_seed = random.randint(1,10000) #random seed used for Brownian integration


# In[56]:


print("Orientational Correlation Time: " + str(1/D_r if D_r !=0 else "infinity") + "\n")
print("Harmonic interaction time: " + str(1/(mu*k))+"\n")
print("Mean free time between collisions from self propulsion: " + str(L**2 /(2*radius*v0*Np) if v0!=0 else "infinity"))


# # Folders
# 
# All research data is contained in a directory with path variable <em>save_dir</em>
# <br>
# <br>
# Within <em>save_dir</em>, there are different sub-directories corresponding to different types of experiments, with path variable <em>exp_dir</em>
# <br>
# <br>
# Within <em>exp_dir</em>, there are different sub-directories corresponding to different specific runs of the experiment, with path variable <em>run_dir</em>. These folders contain information about the specific run, and are typically named by the parameters of that particular run.

# In[63]:


save_dir = "/Users/ryanlopez/Desktop/Python_Programs/Dr_Marchetti_Research/Saved_Data"
exp_folder_name = "glass_vary_J_Dr"
run_folder_name = "D_r=%.4f_and_J=%.4f"%(D_r,J)

exp_dir, run_dir, snapshot_dir = directories.create(save_dir, exp_folder_name, run_folder_name)


# In[64]:


"""
run_desc is an Ordered Dictionary that contains all the data necessary to reproduce the simulation
"""
run_desc = OrderedDict({
    'J':J,
    'D_r':D_r,
    'v0':v0,
    'packing_frac':packing_frac,
    'gamma_t':gamma_t,
    'gamma_r':gamma_r,
    'kT':kT,
    'radius':radius,
    'poly':poly,
    'k':k,
    'L':L,
    'warm_up_time':warm_up_time,
    'tf':tf,
    'tstep':tstep,
    'rand_seed':rand_seed
})
def write_desc():
    run_desc_file = open(os.path.join(run_dir, "run_desc.json"), 'w')
    run_desc_file.write(json.dumps(run_desc))
    run_desc_file.close()
write_desc()


# In[65]:


#creates random initial configuration, saves config to outfile
random_init(phi, L, radius = radius, rcut=0, poly = poly, outfile=os.path.join(run_dir, 'init.json'))


# In[66]:


reader = md.fast_read_json(os.path.join(run_dir, 'init.json'))  #here we read the json file in c++
system = md.System(reader.particles, reader.box)

dump = md.Dump(system)          # Create a dump object

evolver = md.Evolver(system)    # Create a system evolver object

#add the forces and torques

# Create pairwise repulsive interactions with harmonic strength k
evolver.add_force("Soft Repulsive Force", {"k": k})

# Create self-propulsion, self-propulsion strength alpha
evolver.add_force("Self Propulsion", {"alpha": alpha})

# Create pairwise polar alignment with alignment strength J =
evolver.add_torque("Velocity Align", {"k": J})

#Add integrators

# Integrator for updating particle position, friction gamma = 1.0 , "random seed" seed = 10203 and no thermal noise
evolver.add_integrator("Brownian Positions", {"T": kT, "gamma": gamma_t, "seed": rand_seed})

# Integrator for updating particle orientation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0, "random seed" seed = 10203
evolver.add_integrator("Brownian Rotation", {"T": kT_rot, "gamma": gamma_r, "seed": rand_seed})

evolver.set_time_step(tstep) # Set the time step for all the integrators


# In[67]:


#warms up simulation to reach steady state
for t in range(warm_up_nsteps):
    evolver.evolve()
print("Warm up time complete")


# In[68]:


total_snapshots = 100 #total number of snapshots to save
print("Saving observables every %s time steps"%(int(nsteps/total_snapshots)))

#simulation while logging quantities:
for t in range(nsteps):
    if t % int(nsteps/10) == 0:
        print("Time step : ", t)
    evolver.evolve()    # Evolve the system by one time step
    if t % int(nsteps/total_snapshots) == 0:     #Save snapshot of the observable data
        snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:05d}.txt'.format(t))
        dump.dump_data(snapshot_file_path) #Saves data in .txt file
        snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:05d}.vtp'.format(t))
        dump.dump_vtp(snapshot_file_path) #Saves data in .vtp file for visualization
print("done")

