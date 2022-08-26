import sys
sys.path.insert(1, '../')
import json
import read_data as rd

def convert_txt_to_json(txt_filepath, json_filepath, box_length, radius=1):
    """
    Converts simulation state txt file to json for loading an already ran simulation.
    """
    snapshot_data = rd.get_snapshot_data(txt_filepath)
    num_particles = len(snapshot_data['id'])
    particles = []
    for particle_index in range(num_particles):
        particles.append({
           'id': int(snapshot_data['id'][particle_index]),
           'r': [snapshot_data['x'][particle_index], snapshot_data['y'][particle_index]],
           'n': [snapshot_data['nx'][particle_index], snapshot_data['ny'][particle_index]],
           'v': [snapshot_data['vx'][particle_index], snapshot_data['vy'][particle_index]],
           'f': [0.0,0.0], #computed based on positions
           'radius': radius
        })
    #Put into ABPTutorial format
    json_data = {}
    json_data['system'] = {}
    json_data['system']["box"] = {"Lx": box_length, "Ly": box_length}
    json_data['system']["particles"] = particles
    with open(json_filepath, 'w') as json_file:
        json.dump(json_data, json_file, indent = 4)
txt_file = '/home/ryanlopez/Velocity_Align_Norm_Tests/phi=0.6000_and_v0=0.0300/J=0.0010_and_Dr=0.0010/snapshot_data/snapshot_01999.txt'
json_file = '/home/ryanlopez/Velocity_Align_Norm_Tests/test_json_write.json'
convert_txt_to_json(txt_file, json_file, 70)