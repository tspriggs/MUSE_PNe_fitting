import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astroquery.simbad import Simbad

gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv")

x_y = np.ones((len(gal_df["Galaxy"]),2))

for i, gal in enumerate(gal_df["Galaxy"]):
    hdulist = fits.open(f"/local/tspriggs/Fornax_data_cubes/{gal}center.fits")
    hdr = hdulist[1].header
    wcs_obj = WCS(hdr, naxis=2)
    result_table = Simbad.query_object(gal)
    RA = result_table["RA"]
    DEC = result_table["DEC"]
    x_y[i] = utils.skycoord_to_pixel(SkyCoord(ra=Angle(RA, u.hourangle), dec=Angle(DEC, u.deg), frame="fk5"), wcs_obj)

t = Table([gal_df["Galaxy"], x_y[:,0], x_y[:,1]], names=("Galaxy", "x_pix", "y_pix"))

t.write("exported_data/galaxy_centre_pix.dat", format="ascii")