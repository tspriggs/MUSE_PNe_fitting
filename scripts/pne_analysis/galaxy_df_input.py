import pandas as pd
import numpy as np
import yaml

from functions.low_number_stats import poissonLimits

def galaxy_df_input(gal_name, loc, PNe_N=np.nan, PNLF_N=np.nan, L_bol=np.nan, L_bol_p=np.nan, L_bol_m=np.nan, R=np.nan, V=np.nan, lit_R=np.nan,
                    sigma=np.nan, metal_M_H=np.nan, metal_Mg_Fe=np.nan, age=np.nan, mass=np.nan,
                    PNLF_dM=np.nan, PNLF_dM_err_up=np.nan, PNLF_dM_err_lo=np.nan, lit_dM=np.nan, lit_dM_err=np.nan,
                    GALEX_FUV=np.nan, GALEX_NUV=np.nan, app_GALEX_FUV=np.nan, app_GALEX_NUV=np.nan,
                    lit_Lbol=np.nan, lit_Lbol_p=np.nan, lit_Lbol_m=np.nan, lit_PNLF_N=np.nan, lit_PNLF_N_err=np.nan,
                    Bl_dM=np.nan, Bl_dM_err=np.nan, Bl_Bmag=np.nan,                  
                    R_app=np.nan, V_app=np.nan, sigma_app=np.nan, sigma_app_err=np.nan, ppxf_M_H=np.nan,
                    GIST_metal=np.nan, GIST_alpha=np.nan, GIST_metal_emiles=np.nan, 
                    c2=np.nan, c2_err=np.nan, c3=np.nan, c3_err=np.nan):
    
    galaxy_df.loc[(gal_name, loc), "PNe N"]       = PNe_N
    galaxy_df.loc[(gal_name, loc), "PNLF N"]      = PNLF_N
    galaxy_df.loc[(gal_name, loc), "PNLF dM"]     = PNLF_dM
    galaxy_df.loc[(gal_name, loc), "PNLF dM err up"] = PNLF_dM_err_up   
    galaxy_df.loc[(gal_name, loc), "PNLF dM err lo"] = PNLF_dM_err_lo
    galaxy_df.loc[(gal_name, loc), "Lbol"]        = L_bol
    galaxy_df.loc[(gal_name, loc), "Lbol p"]      = L_bol_p
    galaxy_df.loc[(gal_name, loc), "Lbol m"]      = L_bol_m
    galaxy_df.loc[(gal_name, loc), "lit Lbol"]    = lit_Lbol
    galaxy_df.loc[(gal_name, loc), "lit Lbol p"]  = lit_Lbol_p
    galaxy_df.loc[(gal_name, loc), "lit Lbol m"]  = lit_Lbol_m
    galaxy_df.loc[(gal_name, loc), "Rmag"]        = R
    galaxy_df.loc[(gal_name, loc), "Rmag app"]    = R_app
    galaxy_df.loc[(gal_name, loc), "lit Rmag"]    = lit_R
    galaxy_df.loc[(gal_name, loc), "Vmag"]        = V
    galaxy_df.loc[(gal_name, loc), "Vmag app"]    = V_app
    galaxy_df.loc[(gal_name, loc), "sigma"]       = sigma
    galaxy_df.loc[(gal_name, loc), "sigma app"]   = sigma_app
    galaxy_df.loc[(gal_name, loc), "sigma app err"] = sigma_app_err
    galaxy_df.loc[(gal_name, loc), "M/H"]         = metal_M_H
    galaxy_df.loc[(gal_name, loc), "ppxf M/H"]    = ppxf_M_H
    galaxy_df.loc[(gal_name, loc), "Mg/Fe"]       = metal_Mg_Fe
    galaxy_df.loc[(gal_name, loc), "age"]         = age
    galaxy_df.loc[(gal_name, loc), "GALEX FUV"]   = GALEX_FUV
    galaxy_df.loc[(gal_name, loc), "GALEX NUV"]   = GALEX_NUV
    galaxy_df.loc[(gal_name, loc), "app GALEX FUV"] = app_GALEX_FUV
    galaxy_df.loc[(gal_name, loc), "app GALEX NUV"] = app_GALEX_NUV
    galaxy_df.loc[(gal_name, loc), "lit dM"]      = lit_dM
    galaxy_df.loc[(gal_name, loc), "lit dM err"]  = lit_dM_err
    galaxy_df.loc[(gal_name, loc), "lit PNLF N"]  = lit_PNLF_N
    galaxy_df.loc[(gal_name, loc), "lit PNLF N err"] = lit_PNLF_N_err
    galaxy_df.loc[(gal_name, loc), "Mass"]        = mass
    galaxy_df.loc[(gal_name, loc), "Bl dM"]       = Bl_dM
    galaxy_df.loc[(gal_name, loc), "Bl dM err"]   = Bl_dM_err
    galaxy_df.loc[(gal_name, loc), "Bl Bmag"]     = Bl_Bmag
    galaxy_df.loc[(gal_name, loc), "GIST metal"]  = GIST_metal
    galaxy_df.loc[(gal_name, loc), "GIST alpha"]  = GIST_alpha
    galaxy_df.loc[(gal_name, loc), "GIST metal emiles"] = GIST_metal_emiles
    galaxy_df.loc[(gal_name, loc), "c2"]          = c2
    galaxy_df.loc[(gal_name, loc), "c2 err"]      = c2_err
    galaxy_df.loc[(gal_name, loc), "c3"]          = c3
    galaxy_df.loc[(gal_name, loc), "c3 err"]      = c3_err



galaxy_df = pd.DataFrame(columns=("Galaxy", "loc", "PNe N", "PNLF N", "N err", "PNLF dM", "PNLF dM err up", "PNLF dM err lo",
                                    "Lbol", "lit Lbol", "alpha2.5", "alpha2.5 err up", "alpha2.5 err lo", "age", 
                                    "Vmag", "Rmag", "sigma", "M/H", "Mg/Fe", "ppxf M/H", "Bl dM", "Bl dM err", "Bl Bmag",
                                    "Rmag app", "Vmag app", "sigma app", "sigma app err", "GIST metal", "GIST alpha", 
                                    "GALEX FUV", "GALEX NUV", "GALEX FUV-NUV", "app GALEX FUV", "app GALEX NUV", "c2", "c2 err", "c3", "c3 err"))

with open("config/galaxy_info.yaml", "r") as yaml_data: 
     galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

F3D_gals = []
F3D_locs = []
for gal in galaxy_info:
        name, loc = gal.split("_")
        F3D_gals.append(name)
        F3D_locs.append(loc)

tuples = list(zip(*[F3D_gals,F3D_locs]))
galaxy_df["Galaxy"] = F3D_gals
galaxy_df["loc"] = F3D_locs


galaxy_df.set_index(["Galaxy", "loc"], inplace=True)



idx = pd.IndexSlice
galaxy_df.loc[idx[:, 'center'], "GALEX FUV-NUV"] = galaxy_df.loc[idx[:, 'center'], "GALEX FUV"] - galaxy_df.loc[idx[:, 'center'], "GALEX NUV"]
galaxy_df.loc[idx[:, 'center'], "app GALEX FUV-NUV"] = galaxy_df.loc[idx[:, 'center'], "app GALEX FUV"] - galaxy_df.loc[idx[:, 'center'], "app GALEX NUV"]

gal_names = np.unique(galaxy_df.index.get_level_values(0))


for gal, loc in zip(F3D_gals, F3D_locs):
    PN_result_df = pd.read_csv(f"exported_data/{gal}/{gal}{loc}_PN_result_df.csv")
    PN_result_df = PN_result_df.to_dict("list")
    galaxy_df.loc[(gal, loc), "PNe N"] = PN_result_df['PNe N'][0]
    galaxy_df.loc[(gal, loc), "PNLF N"] = PN_result_df['PNLF N'][0]
    if loc == "center":
        galaxy_df.loc[(gal, loc), "PNLF dM"] = PN_result_df['PNLF dM'][0]
        galaxy_df.loc[(gal, loc), "PNLF dM err up"] = PN_result_df['PNLF dM err up'][0]
        galaxy_df.loc[(gal, loc), "PNLF dM err lo"] = PN_result_df['PNLF dM err lo'][0]
    elif loc in ["halo", "middle"]:
        galaxy_df.loc[(gal, loc), "PNLF dM"] = galaxy_df.loc[(gal, "center"), "PNLF dM"]
        galaxy_df.loc[(gal, loc), "PNLF dM err up"] = galaxy_df.loc[(gal, "center"),'PNLF dM err up']
        galaxy_df.loc[(gal, loc), "PNLF dM err lo"] = galaxy_df.loc[(gal, "center"),'PNLF dM err lo']

    Lbol_results_df = pd.read_csv(f"exported_data/{gal}/{gal}{loc}_Lbol_df.csv")
    Lbol_results_df = Lbol_results_df.to_dict("list")
    galaxy_df.loc[(gal, loc), "Lbol"] = Lbol_results_df["Lbol"][0]
    galaxy_df.loc[(gal, loc), "Lbol p"] = Lbol_results_df["Lbol_err_up"][0]
    galaxy_df.loc[(gal, loc), "Lbol m"] = Lbol_results_df["Lbol_err_lo"][0]
    galaxy_df.loc[(gal, loc), "Vmag"] = Lbol_results_df["mag_v"][0]
    galaxy_df.loc[(gal, loc), "Rmag"] = Lbol_results_df["mag_r"][0]


### Calculate alpha, log alpha and associated errors

# add dM error in quadrature, of 0.02 mag
galaxy_df["PNLF dM err up"] = np.sqrt(galaxy_df["PNLF dM err up"].astype(float)**2 + 0.02**2)
galaxy_df["PNLF dM err lo"] = np.sqrt(galaxy_df["PNLF dM err lo"].astype(float)**2 + 0.02**2)


# galaxy_df["N err"] = (1/(galaxy_df["PNe N"]**(0.5)))*galaxy_df["PNLF N"]
# replace with Poisson limits
galaxy_df["N err up"] = [((poissonLimits(pn_n)[0]*pnlf_n) / pn_n) - pnlf_n for pn_n, pnlf_n in zip(galaxy_df["PNe N"], galaxy_df["PNLF N"])]
galaxy_df["N err lo"] = [pnlf_n - ((poissonLimits(pn_n)[1]*pnlf_n) / pn_n) for pn_n, pnlf_n in zip(galaxy_df["PNe N"], galaxy_df["PNLF N"])]



galaxy_df["alpha2.5"] = (galaxy_df["PNLF N"]/galaxy_df["Lbol"]).astype(float)
galaxy_df["log alpha2.5"] = np.log10(galaxy_df["alpha2.5"].values.astype(float))


# Normal method
up_lo = np.array([poissonLimits(n) for n in galaxy_df["PNe N"]])
n_err = [[pn/np.sqrt(e[0]), pn/np.sqrt(e[1])] for e,pn in zip( up_lo, galaxy_df["PNLF N"])]

alpha_err_upper = [a * np.sqrt( (n_e_up/N)**2  + (L_up/L)**2 ) for a, n_e_up, N, L_up, L in zip(galaxy_df["alpha2.5"],galaxy_df["N err up"],
                                                                                galaxy_df["PNLF N"],galaxy_df["Lbol p"],galaxy_df["Lbol"] )  ]

alpha_err_lower = [a * np.sqrt( (n_e_lo/N)**2  + (L_lo/L)**2 ) for a, n_e_lo, N, L_lo, L in zip(galaxy_df["alpha2.5"],galaxy_df["N err lo"],
                                                                                galaxy_df["PNLF N"],galaxy_df["Lbol m"],galaxy_df["Lbol"] )  ]



galaxy_df["alpha2.5 err up"] = alpha_err_upper
galaxy_df["alpha2.5 err lo"] = alpha_err_lower

galaxy_df["log alpha2.5 err up"] = 0.434 * (galaxy_df["alpha2.5 err up"] / galaxy_df["alpha2.5"])
galaxy_df["log alpha2.5 err lo"] = 0.434 * (galaxy_df["alpha2.5 err lo"] /  galaxy_df["alpha2.5"])


# save galaxy_df to fits for safe keeping
galaxy_df.to_csv("exported_data/galaxy_dataframe.csv")

