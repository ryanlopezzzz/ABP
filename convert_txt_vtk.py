"""
Converts .txt files to .vtk files
"""
import subprocess
import os
import numpy as np
import vtk
import read_data as rd #reads snapshot text data

#Directory where all data is saved
server_save_dir = "ryanlopez@128.111.18.245:~/Saved_Data"
local_save_dir = "/Users/ryanlopez/Desktop/Python_Programs/Dr_Marchetti_Research/Saved_Data"

exp_folder_name = input('Enter Exp Folder Name: ')
run_folder_name = input('Enter Run Folder Name: ')

server_snapshot_path = os.path.join(server_save_dir, exp_folder_name, run_folder_name, 'snapshot_data/') 
local_snapshot_path = os.path.join(local_save_dir, exp_folder_name, run_folder_name)

if not os.path.isdir(os.path.join(local_save_dir, exp_folder_name)):
    os.mkdir(os.path.join(local_save_dir, exp_folder_name))
if not os.path.isdir(local_snapshot_path):
    os.mkdir(local_snapshot_path)

subprocess.run(['rsync', '-a', server_snapshot_path, local_snapshot_path])

exp_data = rd.get_exp_data(local_snapshot_path)
for snapshot_index in range(exp_data['id'].shape[0]): #runs through snapshots at different times
    vtp_filename = os.path.join(local_snapshot_path, 'snapshot_{:05d}.vtp'.format(snapshot_index))
    points = vtk.vtkPoints()
    ids = vtk.vtkIntArray()
    n = vtk.vtkDoubleArray()
    v = vtk.vtkDoubleArray()
    ids.SetNumberOfComponents(1)
    n.SetNumberOfComponents(3)
    v.SetNumberOfComponents(3)
    ids.SetName("id")
    n.SetName("director")
    v.SetName("velocity")

    for p in range(exp_data['id'].shape[1]): #run through all the particles                
        points.InsertNextPoint([exp_data['x'][snapshot_index,p],exp_data['y'][snapshot_index,p], 0.0])
        ids.InsertNextValue(int(exp_data['id'][snapshot_index,p]))
        n.InsertNextTuple([exp_data['nx'][snapshot_index,p],exp_data['ny'][snapshot_index,p], 0.0])
        v.InsertNextTuple([exp_data['vx'][snapshot_index,p],exp_data['vy'][snapshot_index,p], 0.0])  
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().AddArray(ids)
    polyData.GetPointData().AddArray(n)
    polyData.GetPointData().AddArray(v)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polyData)
    else:
        writer.SetInputData(polyData)
    writer.SetDataModeToAscii()
    writer.Write()   