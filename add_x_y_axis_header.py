import numpy as np
from astropy.io import fits

list_of_galaxies = ["FCC083", "FCC147", "FCC161", "FCC167", "FCC184", "FCC193",
                    "FCC219", "FCC249", "FCC255","FCC276","FCC277", "FCC310",]
for g in list_of_galaxies:
    # open galaxy
    raw_galaxy = fits.open(f"/local/tspriggs/Fornax_data_cubes/{g}center.fits")

    # shape order - wave, y, x
    raw_shape = np.shape(raw_galaxy[1].data)

    y_axis, x_axis = raw_shape[1], raw_shape[2] # store y and x in variables, taken from raw_shape

    # open residual data of galaxy and set two new cards in header: YAXIS and XAXIS
    with fits.open(f"galaxy_data/{g}_data/{g}_residuals_list.fits", mode="update") as res_galaxy:
    
        res_hdr = res_galaxy[0].header # store header
    
        res_hdr.set("YAXIS", value=y_axis) # set YAXIS
        res_hdr.set("XAXIS", value=x_axis) # set XAXIS
    
        res_galaxy.flush()  # changes are written back to original.fits
    
    print(f"{g} has been updated." )