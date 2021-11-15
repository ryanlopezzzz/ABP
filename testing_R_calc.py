import resource
gigabyte = int(1e9)
resource.setrlimit(resource.RLIMIT_AS, (45*gigabyte, 50*gigabyte))


import os
import read_data as rd #reads snapshot text data
import Physical_Quantities.local_flocking_R as local_flocking_R

base_dir = "/home/ryanlopez/Velocity_Align_Saved_Data/"
folder_dir = "phi=0.60_and_v0=0.0010"
#folder_dir = "phi=1.00_and_v0=0.1000"
run_dir = "J=10000.0000_and_Dr=0.0001"
#run_dir = "J=10.0000_and_Dr=0.1000"
#run_dir = "J=100.0000_and_Dr=0.0001"

#folder_dir = ""
snapshot_name = "snapshot_data"

snapshot_dir = os.path.join(base_dir, folder_dir, run_dir, snapshot_name)
exp_data = rd.get_exp_data(snapshot_dir)
#R_avg = local_flocking_R.get_local_flocking_R(exp_data, 2, 70, snapshot=0)

#print(R_avg)

