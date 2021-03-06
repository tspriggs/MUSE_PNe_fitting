import os
import yaml

# Open yaml file to get galaxy names from individual entries

with open("../galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
galaxy_names = [gal for gal in galaxy_info]

# data path
data_path = [f"galaxy_data/{gal}_data/" for gal in galaxy_names]

# Plots path

plots_path = [f"Plots/{gal}/" for gal in galaxy_names]

# exported data

export_path = [f"exported_data/{gal}/" for gal in galaxy_names]

# loop through galaxies and create directories, if they are not there, make them, if they are, continue

for i in range(0, len(galaxy_names)):
    try:
        os.mkdir(data_path[i])    
        print("Directory " , data_path[i] ,  " Created ")
    except FileExistsError:
        print("Directory " , data_path[i] ,  " already exists")  
    
    try:
        os.makedirs(plots_path[i])    
        print("Directory " , plots_path[i] ,  " Created ")
    except FileExistsError:
        print("Directory " , plots_path[i] ,  " already exists") 
        
    try:
        os.makedirs(export_path[i])    
        print("Directory " , export_path[i] ,  " Created ")
    except FileExistsError:
        print("Directory " , export_path[i] ,  " already exists") 
        
        
    
    
    