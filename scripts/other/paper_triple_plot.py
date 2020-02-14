import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# plot out each galaxy
# 3,1 subplots (3 across, 1 high, per galaxy)
# Left plot is the A/rN plot with circled PNe, masks etc.
# middle plot is the velocity comparison histogram
# right plot is the PNLF

# 4 galaxies per page / save

gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv")

gal_centre_pix = Table.read("exported_data/galaxy_centre_pix.dat")

# read in necessary data

# make list of galaxies

gal_list = [g for g in gal_df["Galaxy"]]

# Plot 1 required stuff - A/rN
def plot_A_rN(galaxy, ):
    A_rN = np.load(f"exported_data/{galaxy}/{galaxy}_A_rN.npy")
    x_y_list = np.load(f"exported_data/{galaxy}/{galaxy}_PNe_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])
    
    # Mask
    with open("galaxy_info.yaml", "r") as yaml_data:
        galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
    galaxy_data = galaxy_info[galaxy]
    gal_mask_params = galaxy_data["gal_mask"]
    star_mask_params = galaxy_data["star_mask"]
    
    return A_rN, x_PNe, y_PNe, gal_mask_params, star_mask_params




# Plot 2 required stuff - Velocity



# Plot 3 required stuff - PNLF