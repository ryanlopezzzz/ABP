import sys
import os
import json
import numpy as np
import argparse
sys.path.insert(1, '/Users/ryanlopez/ABPTutorial/c++') #Connects to ABP Folder github.com/ryanlopezzzz/ABPTutorial
from cppmd.builder import *
import cppmd as md

#Load parameters for simulation
parser = argparse.ArgumentParser()
parser.add_argument('--paramsFilename', help='Parameters for the run.')
args = parser.parse_args()
with open(args.paramsFilename, 'r') as params_file:
    run_desc = json.load(params_file)

J = run_desc['J']
D_r = run_desc['D_r']
v0 = run_desc['v0']
print('v0 is: ' + str(v0))
packing_frac = run_desc['packing_frac']
gamma_t = run_desc['gamma_t']
gamma_r = run_desc['gamma_r']
kT = run_desc['kT']
radius = run_desc['radius']
poly = run_desc['poly']
k = run_desc['k']
L = run_desc['L']
warm_up_time = run_desc['warm_up_time']
tf = run_desc['tf']
tstep = run_desc['tstep']
rand_seed = run_desc['rand_seed']
vel_align_norm = run_desc['vel_align_norm']
velocity_align = run_desc['velocity_align']
polar_align = run_desc['polar_align']
total_snapshots = run_desc['total_snapshots']
exp_dir = run_desc['exp_dir']
run_dir = run_desc['run_dir']
snapshot_dir = run_desc['snapshot_dir']

"""
Calculated physical parameters:
"""
Teff = 0.5 * gamma_t * v0**2 / D_r
alpha = v0 * gamma_t 
mu = 1/gamma_t #mobility
D_t = kT * mu #translational diffusion coefficient, this comes from Fluctuation-Dissipation Theorem
kT_rot = D_r * gamma_r #rotational temperature, see brownian_rot_integrator.py
phi = packing_frac / (np.pi*radius**2) #particle number density, phi = N/(L^2 * <R^2>)
Np = int(round(phi*L**2)) #number of particles
warm_up_nsteps = int(warm_up_time / tstep) #number of integration steps for warming up
nsteps = int(tf / tstep) #number of int. steps for logging physical quantites

print("Orientational Correlation Time: " + str(1/D_r if D_r !=0 else "infinity") + "\n")
print("Harmonic interaction time: " + str(1/(mu*k))+"\n")
print("Mean free time between collisions from self propulsion: " + str(L**2 /(2*radius*v0*Np) if v0!=0 else "infinity"))

#creates random initial configuration, saves config to outfile
random_init(phi, L, radius = radius, rcut=0, poly = poly, outfile=os.path.join(run_dir, 'init.json'))

reader = md.fast_read_json(os.path.join(run_dir, 'init.json'))  #here we read the json file in c++
system = md.System(reader.particles, reader.box)

dump = md.Dump(system)          # Create a dump object

evolver = md.Evolver(system)    # Create a system evolver object

#add the forces and torques

# Create pairwise repulsive interactions with harmonic strength k
evolver.add_force("Soft Repulsive Force", {"k": k})

# Create self-propulsion, self-propulsion strength alpha
evolver.add_force("Self Propulsion", {"alpha": alpha})

# Create pairwise alignment
if velocity_align == True:
    evolver.add_torque("Velocity Align", {"k": J, "norm": vel_align_norm})
elif polar_align == True:
    evolver.add_torque("Polar Align", {"k": J})

#Add integrators

# Integrator for updating particle position, friction gamma = 1.0 , "random seed" seed = 10203 and no thermal noise
evolver.add_integrator("Brownian Positions", {"T": kT, "gamma": gamma_t, "seed": rand_seed})

# Integrator for updating particle orientation, friction gamma = 1.0, "rotation" T = 0.1, D_r = 0.0, "random seed" seed = 10203
evolver.add_integrator("Brownian Rotation", {"T": kT_rot, "gamma": gamma_r, "seed": rand_seed})

evolver.set_time_step(tstep) # Set the time step for all the integrators

#warms up simulation to reach steady state
for t in range(warm_up_nsteps):
    evolver.evolve()
print("Warm up time complete")

print("Saving observables every %s time steps"%(int(nsteps/total_snapshots)))

#simulation while logging quantities:
for t in range(nsteps):
    if t % int(nsteps/10) == 0:
        print("Time step : ", t)
    evolver.evolve()    # Evolve the system by one time step
    if t % int(nsteps/total_snapshots) == 0:     #Save snapshot of the observable data
        snapshot_num = int(t*total_snapshots/nsteps)
        snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:05d}.txt'.format(snapshot_num))
        dump.dump_data(snapshot_file_path) #Saves data in .txt file
        #snapshot_file_path = os.path.join(snapshot_dir, 'snapshot_{:09d}.vtp'.format(t))
        #dump.dump_vtp(snapshot_file_path) #Saves data in .vtp file for visualization
print("done")