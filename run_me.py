import os
import yaml
import numpy as np

# First make a parent directory where all of the following will be stored

# Then setup file structures, as set out in yaml file

#with open("galaxy_info.yaml", "r") as yaml_data:
#    galaxy_info = yaml.load(yaml_data)
    
galaxy_names = ['FCC167', 'FCC170', 'FCC255', 'FCC277']#[gal for gal in galaxy_info]

# data path
data_path = ["{}_data/".format(gal) for gal in galaxy_names]

# Plots path

plots_path = ["Plots/{}/full_spec_fits".format(gal) for gal in galaxy_names]

# exported data

export_path = ["exported_data/{}".format(gal) for gal in galaxy_names]

for i in np.arange(0, len(galaxy_names)):
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
        
        
    
    
    