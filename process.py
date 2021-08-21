import subprocess
import numpy as np

param_list1 = np.linspace(0,0.2,4) #J
param_list2 = np.linspace(0.03333333,0.2,4) #D_r
 
for i, param1 in enumerate(param_list1):
    for j, param2 in enumerate(param_list2):
        if [i,j] != [0,0] and [i,j] != [0,3] and [i,j] != [3,0] and [i,j] != [3,3]:
            subprocess.run(["python3", "ABP_Simulation.py", "--params",  "%.8e"%param1, "%.8e"%param2])