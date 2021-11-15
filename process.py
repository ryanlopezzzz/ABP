import subprocess
import numpy as np

param_list1 = np.logspace(-2,0,num=3) #J
param_list2 = np.logspace(-2,0,num=3) #D_r
 
for param1 in param_list1:
    for param2 in param_list2:
        subprocess.run(["python3","ABP_Simulation.py","--params","%.8e"%param1,"%.8e"%param2])