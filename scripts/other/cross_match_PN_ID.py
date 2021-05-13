from astropy.io import ascii, fits
import numpy as np
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import pandas as pd
import matplotlib.pyplot as plt
import yaml



with open("../config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

# centre pointings only:
gal_names_c = ['FCC119', 'FCC143', 'FCC182', 'FCC249', 'FCC255', 'FCC276', 'FCC277', 'FCC301']

for gal in gal_names_c:
    PNe_df_cen = pd.read_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv")
    PNe_df_cen["index"] = ["C-00" for i in range(len(PNe_df_cen))]
    PNe_df_cen.loc[PNe_df_cen["ID"]!="-", "index"] = [f"C-{i}" for i in np.arange(1, len(PNe_df_cen.loc[PNe_df_cen["ID"]!="-"])+1)]
    PNe_df_cen.to_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv")


# centre and halo only situations:
gal_names_c_h = {'FCC083':[41,48,61,58,60,66],
                'FCC147':[100,101,114],
                'FCC148':[5,10,23,22,27,30,33,44,52],
                'FCC153':[8,13,15,18,22,26,35,39,44],
                'FCC161':[37,38,43,46,58,63,67,69],
                'FCC170':[1,2,3,4,6,7,9,8,12,13,14],
                'FCC177':[46,47,49,51,52,53,54,55,56],
                'FCC190':[2,4,3,5,6,8,12],
                'FCC193':[0,1,8,10],
                'FCC219':[89],
                'FCC310':[2,4,5,7,6,8,9,10,11,12,13,15,14,17,18,16,20,19,22,25,26,24,28,27,30,23,31,34,33,36,35,38,39,40,44,42,43,45]}
                

for gal in gal_names_c_h.keys():
    PNe_df_cen = pd.read_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv")
    PNe_df_halo = pd.read_csv(f"exported_data/{gal}/{gal}halo_PNe_df.csv")
    PNe_df_cen["index"] = ["C-00" for i in range(len(PNe_df_cen))]
    PNe_df_halo["index"] = ["H-00" for i in range(len(PNe_df_halo))]
    PNe_df_cen["index"].loc[PNe_df_cen["ID"]!="-"] = [f"C-{i}" for i in np.arange(1, len(PNe_df_cen.loc[PNe_df_cen["ID"]!="-"])+1)]
    PNe_df_halo["index"].loc[PNe_df_halo["ID"]!="-"] = [f"H-{i}" for i in np.arange(1, len(PNe_df_halo.loc[PNe_df_halo["ID"]!="-"])+1)]
    halo_index = galaxy_info[f"{gal}_halo"]["crossmatch_filter"]
    for i, j in zip(halo_index, gal_names_c_h[f"{gal}"]):
        try:
            PNe_df_halo["index"].iloc[i] = PNe_df_halo["index"].iloc[i] + " / " + PNe_df_cen.loc[PNe_df_cen["ID"]!="-","index"].iloc[j]
        except:
            print("Crossmatch object not detected in central pointing.")
    
    PNe_df_cen.to_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv")
    PNe_df_halo.to_csv(f"exported_data/{gal}/{gal}halo_PNe_df.csv")

# centre, middle and halo sitations:
gal_names_c_m_h = {'FCC167':[[98, 102, 105, 107, 108,109, 110, 112], [41,42,43]], 
                    'FCC184':[[39,41,52,54],[12]]}

for gal in gal_names_c_m_h.keys():
    PNe_df_cen = pd.read_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv")
    PNe_df_mid = pd.read_csv(f"exported_data/{gal}/{gal}middle_PNe_df.csv")
    PNe_df_halo = pd.read_csv(f"exported_data/{gal}/{gal}halo_PNe_df.csv")

    PNe_df_cen["index"] = ["C-00" for i in range(len(PNe_df_cen))]
    PNe_df_mid["index"] = ["M-00" for i in range(len(PNe_df_mid))]
    PNe_df_halo["index"] = ["H-00" for i in range(len(PNe_df_halo))]

    PNe_df_cen["index"].loc[PNe_df_cen["ID"]!="-"] = [f"C-{i}" for i in np.arange(1, len(PNe_df_cen.loc[PNe_df_cen["ID"]!="-"])+1)]
    PNe_df_mid["index"].loc[PNe_df_mid["ID"]!="-"] = [f"M-{i}" for i in np.arange(1, len(PNe_df_mid.loc[PNe_df_mid["ID"]!="-"])+1)]
    PNe_df_halo["index"].loc[PNe_df_halo["ID"]!="-"] = [f"H-{i}" for i in np.arange(1, len(PNe_df_halo.loc[PNe_df_halo["ID"]!="-"])+1)]

    mid_index = galaxy_info[f"{gal}_middle"]["crossmatch_filter"]
    for i, j in zip(mid_index, gal_names_c_m_h[f"{gal}"][0]):
        try:
            PNe_df_mid["index"].iloc[i] = PNe_df_mid["index"].iloc[i] + " / " + PNe_df_cen.loc[PNe_df_cen["ID"]!="-","index"].iloc[j]
        except:
            print("Crossmatch object not detected in central pointing.")

    halo_index = galaxy_info[f"{gal}_halo"]["crossmatch_filter"]
    for i, j in zip(halo_index, gal_names_c_m_h[f"{gal}"][1]):
        try:
            PNe_df_halo["index"].iloc[i] = PNe_df_halo["index"].iloc[i] + " / " + PNe_df_mid.loc[PNe_df_mid["ID"]!="-","index"].iloc[j]
        except:
            print("Crossmatch object not detected in central pointing.")
    
    PNe_df_cen.to_csv(f"exported_data/{gal}/{gal}center_PNe_df.csv") 
    PNe_df_mid.to_csv(f"exported_data/{gal}/{gal}middle_PNe_df.csv") 
    PNe_df_halo.to_csv(f"exported_data/{gal}/{gal}halo_PNe_df.csv")
