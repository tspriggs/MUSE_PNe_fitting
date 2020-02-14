from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import pandas as pd
import numpy as np

gal_df = pd.read_csv("../exported_data/galaxy_dataframe.csv")

gal_list = [gal for gal in gal_df["Galaxy"]]

# Search Simbad for the ID's of each FCC galaxy
# Find which index is the LEDA number, then split to "LEDA" and "number"
# Store number for use when searching CosmicFlows3 catalogue
# LEDA_list = []
# for g in gal_list:
#     LEDA_indx = [i for i, s in enumerate(Simbad.query_objectids(g)) if "LEDA" in s[0]][0]
#     LEDA_n = Simbad.query_objectids(g)[LEDA_indx][0]
#     LEDA_list.append(LEDA_n.split()[1])

viz_cat = Vizier(columns=["DM", "e_DM", "r_Dist"], catalog="J/AJ/152/50/table3")

# dissasemble this
viz_results=[viz_cat.query_object(i)[0] for i in gal_list]

dM = [viz_results[i]["DM"][0] for i in range(len(gal_list))]
dM_err = [viz_results[i]["e_DM"][0] for i in range(len(gal_list))]
