import numpy as np
import matplotlib.pyplot as plt
import lmfit
import pandas as pd
import yaml
import sep
import scipy as sp
# import pymc3 as pm

from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
from tqdm import tqdm
from photutils import CircularAperture
from IPython.display import display

from photutils import CircularAperture
from scipy.stats import norm, chi2
from scipy import stats

#from astroquery.vizier import Vizier
from astropy.io import ascii, fits
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext

from functions.file_handling import paths, open_data
from functions.PNe_functions import dM_to_D, D_to_dM

##### READ IN ###########

gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
pd.set_option('display.max_columns', 100)
idx = pd.IndexSlice

gal_names = np.unique(gal_df.index.get_level_values(0))
galaxy_loc = [np.unique(gal_df.index.get_level_values(0)),["center"]*len(np.unique(gal_df.index.get_level_values(0)))]
galaxy_halo = [gal_df.loc[idx[:, 'halo'], :].index.get_level_values(0), ["halo"]*len(gal_df.loc[idx[:, 'halo'], :].index.get_level_values(0))]
galaxy_mid = [gal_df.loc[idx[:, 'middle'], :].index.get_level_values(0), ["middle"]*len(gal_df.loc[idx[:, 'middle'], :].index.get_level_values(0))]

gal_cen_tuples = list(zip(*galaxy_loc))
gal_halo_tuples = list(zip(*galaxy_halo))
gal_middle_tuples = list(zip(*galaxy_mid))
# gal_df["marker"] = ["o", "v", "<","3", ">", "8", "s", "p", "P", "*", "h", "H", "+","X", "x", "D", "d", ".", "o", "v"] # 17

# PN_N_filter  = (gal_df.loc[idx[:, 'center'], "PNe N"]>20)
PN_N_filter  = (gal_df.loc[idx[:, 'center'], "PNe N"]>=5) &  (gal_df.loc[gal_cen_tuples, "Bl dM"].notna()) 

# UV_PN_filter  = (PN_N_filter) & (gal_df.query("loc == 'center'").query("Galaxy != 'FCC119'").query("Galaxy != 'FCC184'"))#& pd.notnull(gal_df["alpha2.5"])
PN_filter_FO = (PN_N_filter) & (~gal_df.loc[idx[:, 'center'],:].index.get_level_values(0).isin(["FCC153", "FCC170", "FCC177"]))

# lit_gal_df = pd.read_csv("../exported_data/lit_galaxy_df.csv")
with open("config/galaxy_info.yaml", "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
galaxy_info = [yaml_info[f"{gal_name}_center"] for gal_name in gal_names]


#### make a table ready for LaTeX document #######
sim_D = np.asarray(10.**(((gal_df["PNLF dM"]) -25.) / 5.))
sim_D_err_up = np.asarray(gal_df["PNLF dM err up"].values / (5/(np.log(10) * sim_D)))
sim_D_err_lo = np.asarray(gal_df["PNLF dM err lo"].values / (5/(np.log(10) * sim_D)))

dM_list = gal_df["PNLF dM"].values
dM_u_list = gal_df["PNLF dM err up"].values
dM_l_list = gal_df["PNLF dM err lo"].values

latex_table = pd.DataFrame()

alt_name_list = []
dM_for_table = []
D_for_table = []

for i, (gal, loc) in enumerate(zip(gal_df.index.get_level_values(0), gal_df.index.get_level_values(1))):
    if loc == "center":
        dM_for_table.append(f"${gal_df.loc[(gal, loc), 'PNLF dM']:.2f}"+"^{+"+f"{gal_df.loc[(gal, loc), 'PNLF dM err up']:.2f}"+"}"+
        "_{"+f" -{gal_df.loc[(gal, loc), 'PNLF dM err lo']:.2f} "+"}$")
        D_for_table.append(f"${sim_D[i]:.2f}"+"^{+"+f"{sim_D_err_up[i]:.2f}"+"}"+"_{"+f"-{sim_D_err_lo[i]:.2f}"+"}$")
    else:
        dM_for_table.append("--")
        D_for_table.append("--")

    alt_name = yaml_info[f"{gal}_{loc}"]["alt_name"] 
    if (isinstance(alt_name, str)) and (loc=="center"):
        alt_name_list.append(f"{alt_name.split('_')[0]} {alt_name.split('_')[1]}")
    else:
        alt_name_list.append("--")


latex_table = pd.DataFrame()



gal_table = Table([gal_df.index.get_level_values(0), gal_df.index.get_level_values(1), gal_df["PNe N"], 
                   [f"${int(n)}"+"^{+"+f"{int(e_u)}"+"}"+f"_{ {-int(e_l)} }$" for n,e_u, e_l in zip(round(gal_df["PNLF N"]), round(gal_df["N err up"]), round(gal_df["N err lo"]))], 
                   dM_for_table,  D_for_table,
                #    [f"${D}"+"^{+"+f"{u}"+"}"+f"_{ {-l} }$" for D,u,l in zip(gal_df["PNLF dM"].round(2),gal_df["PNLF dM err up"].round(2), gal_df["PNLF dM err lo"].round(2))],
                #    [f"${round(D,2)}"+"^{+"+f"{u}"+"}"+f"_{ {-l} }$" for D,u,l in zip(sim_D, sim_D_err_up.round(2), sim_D_err_lo.round(2))],
                   [f"${D:.2f} \pm {err:.2f}$" for D,err in zip(gal_df["Bl dM"],gal_df["Bl dM err"])],
                   [f"${D:.2f} \pm {err:.2f}$" for D,err in zip(gal_df["lit dM"],gal_df["lit dM err"])],
                   alt_name_list,
                   ], 
                   names=("Galaxy", "Area", "$N_{PNe}$", "$N_{PNLF}$", "$\mu_{PNLF}$", "$D_{PNLF}$", "$\mu_{SBF}$", "$\mu_{CF2}$", "Alt Name") )

ascii.write(gal_table, f"exported_data/for_catalogue_dM.txt", format="latex", overwrite=True)
