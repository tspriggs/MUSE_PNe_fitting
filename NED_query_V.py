import numpy as np
from astroquery.ned import Ned
from astropy.io import ascii
from astropy.table import Table


gal_name_list = ["205", "221", "224", "1316", "1344", "1399", "3031", "3115",
                 "3377", "3379", "3384", "4374", "4382", "4406", "4477", "4486",
                 "4594", "4649", "4697", "5128"]

FUV = np.ones(len(gal_name_list))
V   = np.ones(len(gal_name_list))
FUV_minus_V = np.ones(len(gal_name_list))

for i, gal_name in enumerate(gal_name_list):
    photometry_table = Ned.get_table("NGC"+gal_name, table="photometry")
    
    V_measure = photometry_table[np.where(photometry_table["Observed Passband"] == b"V (Johnson)")]
    FUV_measure = photometry_table[np.where(photometry_table["Observed Passband"] == b"FUV (GALEX) AB")]
    
    FUV_mag = FUV_measure["Photometry Measurement"]
    V_mag = V_measure["Photometry Measurement"]
    
    if (len(FUV_mag) > 0) and (len(V_mag)):
        FUV[i] = np.nanmedian(FUV_mag)
        V[i] = np.nanmedian(V_mag)
    else: 
        continue

FUV_minus_V = FUV - V

t = Table([FUV, V, FUV_minus_V], names=("FUV", "V", "FUV - V"))
ascii.write(t, "exported_data/FUV_minus_V_table.dat", overwrite=True)

# print("Mean V mag: ", mean_V)
# print("Mean FUV mag: ", mean_FUV)
# print("Mean FUV - V mag", mean_FUV - mean_V)