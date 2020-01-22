import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astroquery.simbad import Simbad

gal_df = pd.read_csv("../exported_data/galaxy_dataframe.csv")

x_y = np.ones((len(gal_df["Galaxy"]),2))
RA_DEC = np.ones_like(x_y)

# Open yaml file
# for gal in yaml file:
# if centre = [1,1] = use header
# else use the yaml pixel coordinates and convert to RA and DEC


# Output needs to be Ra and DEC coordinates of the pointings

for i, gal in enumerate(gal_df["Galaxy"]):
    # open up raw file and get the WCS and RA and DEC from header
    hdulist = fits.open(f"/local/tspriggs/Fornax_data_cubes/{gal}center.fits")
    hdr = hdulist[1].header
    wcs_obj = WCS(hdr, naxis=2)
    # open yaml entry for galaxy
    with open("../galaxy_info.yaml", "r") as yaml_data:
        galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
    galaxy_data = galaxy_info[gal+"_center"]
    gal_cen = galaxy_data["centre"] 
    if gal_cen == [1,1]:
        RA_DEC[i] = [hdr["CRVAL1"], hdr["CRVAL2"]]
    else:
        px_to_sky = utils.pixel_to_skycoord(gal_cen[0], gal_cen[1], wcs_obj)
        RA_DEC[i] = [px_to_sky.ra.deg, px_to_sky.dec.deg]
    
#     RA_DEC[i] = [hdr["CRVAL1"], hdr["CRVAL2"]]
    
    # Query Simbad for galaxy centre RA and DEC
    result_table = Simbad.query_object(gal)
    RA = result_table["RA"]
    DEC = result_table["DEC"]
    x_y[i] = utils.skycoord_to_pixel(SkyCoord(ra=Angle(RA, u.hourangle), dec=Angle(DEC, u.deg), frame="fk5"), wcs_obj)
    

t = Table([gal_df["Galaxy"], x_y[:,0], x_y[:,1], RA_DEC[:,0], RA_DEC[:,1]], names=("Galaxy", "x_pix", "y_pix", "RA cen", "DEC cen"))

t.write("../exported_data/galaxy_centre_pix.dat", format="ascii", overwrite=True)