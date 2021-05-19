import subprocess

param_list = [0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]
 
for param in param_list:
    subprocess.run(["python3", "ABP_Simulation.py", "--param",  "%.4e"%param])