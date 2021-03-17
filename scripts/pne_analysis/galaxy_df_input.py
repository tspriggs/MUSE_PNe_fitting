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

# FCC083
# galaxy_df_input("FCC083", "center", lit_Lbol=4_851_065_531.00737, lit_R=10.84, 
#                 sigma=103, metal_M_H=-0.20, metal_Mg_Fe=0.26, age=13.2, ppxf_M_H=-1.18,
#                 c2=0.25917, c2_err=0.27, c3=5.7856, c3_err=2.37,
#                 lit_dM=31.42, lit_dM_err=0.24, lit_PNLF_N=125.06, lit_PNLF_N_err=16.1, 
#                 app_GALEX_FUV=19.328, app_GALEX_NUV=17.771, GALEX_FUV=19.787, GALEX_NUV=18.105,
#                 Bl_dM=31.422, Bl_dM_err=0.071, Bl_Bmag=12.3,
#                 R_app=11.59, V_app=12.03, sigma_app=127, sigma_app_err=5.4, GIST_metal=-0.375, GIST_alpha=0.316, GIST_metal_emiles=-0.537)


# # FCC119
# galaxy_df_input("FCC119", "center", lit_Lbol=2.62611526e+08, lit_Lbol_p=25336286.36197448, lit_Lbol_m=23106967.79114595,
#                 lit_R=np.nan,  sigma=17, 
#                 metal_M_H=-0.51, metal_Mg_Fe=0.11, age=6.7, mass=0.1e10, ppxf_M_H=-1.25,
#                 c2=0.357, c2_err=0.472, c3=0.0, c3_err=0.5,
#                 lit_dM=31.52, lit_dM_err=0.24, lit_PNLF_N=15.3, lit_PNLF_N_err=4.6, 
#                 app_GALEX_FUV=19.807, app_GALEX_NUV=18.842, GALEX_FUV=20.134, GALEX_NUV=19.003, 
#                 Bl_dM=31.539, Bl_dM_err=0.1, Bl_Bmag=15.0,
#                 R_app=np.nan, V_app=np.nan, sigma_app=np.nan, sigma_app_err=np.nan, GIST_metal=-0.479, GIST_alpha=0.178, GIST_metal_emiles=-0.609)

# # FCC143
# galaxy_df_input("FCC143", "center", lit_Lbol=1.40147162e+09, lit_Lbol_p=1.15523933e+08, lit_Lbol_m=1.06726426e+08,
#                 lit_R=12.66, sigma=62, 
#                 metal_M_H=-0.18, metal_Mg_Fe=0.18, age=12.6, mass=0.28e10, ppxf_M_H=0.165,
#                 c2=0.279844, c2_err=0.426, c3=3.757, c3_err=4.167,
#                 lit_dM=31.22, lit_dM_err=0.24, lit_PNLF_N=24.1, lit_PNLF_N_err=6.2, 
#                 app_GALEX_FUV=20.685, app_GALEX_NUV=19.121, GALEX_FUV=20.727, GALEX_NUV=19.167, 
#                 Bl_dM=31.427, Bl_dM_err=0.086, Bl_Bmag=14.3,
#                 R_app=np.nan, V_app=np.nan, sigma_app=np.nan, sigma_app_err=np.nan, GIST_metal=-0.306, GIST_alpha=0.244, GIST_metal_emiles=-0.451)


# # FCC147
# galaxy_df_input("FCC147", "center", lit_Lbol=6_134_382_862.249334,  lit_R=10.50, 
#                 sigma=131, metal_M_H=0.04, metal_Mg_Fe=0.23, age=13.5, mass=2.4e10, ppxf_M_H=0.0834,
#                 c2=0.31244, c2_err=0.357, c3=0.0, c3_err=0.33,
#                 lit_dM=31.56, lit_dM_err=0.24, lit_PNLF_N=175.7, lit_PNLF_N_err=27, 
#                 app_GALEX_FUV=18.857, app_GALEX_NUV=17.654, GALEX_FUV=19.095, GALEX_NUV=17.823,
#                 Bl_dM=31.458, Bl_dM_err=0.07, Bl_Bmag=11.9,
#                 R_app=11.23, V_app=11.68, sigma_app=133, sigma_app_err=5.2, GIST_metal=-0.207, GIST_alpha=0.247, GIST_metal_emiles=-0.319)


# # FCC148
# galaxy_df_input("FCC148", "center", lit_Lbol=6_134_382_862.249334, lit_R=11.7,
#                 sigma=43, metal_M_H=-0.22, metal_Mg_Fe=0.09, age=9.8, mass=0.58e10, ppxf_M_H=-0.221,
#                 c2=0.26189, c2_err=0.226, c3=3.7268, c3_err=2.08,
#                 lit_dM=31.48, lit_dM_err=0.14, lit_PNLF_N=175.7, lit_PNLF_N_err=27, 
#                 app_GALEX_FUV=21.416, app_GALEX_NUV=20.252, GALEX_FUV=21.416, GALEX_NUV=20.252,
#                 Bl_dM=31.5, Bl_dM_err=0.072, Bl_Bmag=13.6,
#                 R_app=15.89, V_app=16.22, sigma_app=39, sigma_app_err=13, GIST_metal=-0.365, GIST_alpha=0.161, GIST_metal_emiles=-0.46)


# #FCC153
# galaxy_df_input("FCC153", "center", lit_R=11.7, sigma=55, 
#                 metal_M_H=-0.05, metal_Mg_Fe=0.11, age=10.7, mass=0.76e10, ppxf_M_H=-1.65,
#                 c2=0.365669, c2_err=0.375, c3=0.595, c3_err=0.196,
#                 lit_dM=31.32, lit_dM_err=0.24, lit_PNLF_N=79.4, lit_PNLF_N_err=12.9, 
#                 app_GALEX_FUV=19.763, app_GALEX_NUV=18.152, GALEX_FUV=19.763, GALEX_NUV=18.152,
#                 Bl_dM=31.588, Bl_dM_err=0.071, Bl_Bmag=13.0,
#                 R_app=12.10, V_app=12.52, sigma_app=98, sigma_app_err=13, GIST_metal=-0.01, GIST_alpha=0.138, GIST_metal_emiles=-0.1)


# # FCC161
# galaxy_df_input("FCC161", "center", lit_Lbol=6815479441.900765, lit_R=10.47, 
#                  sigma=96, metal_M_H=-0.13, metal_Mg_Fe=0.20, age=12.9, mass=2.63e10, ppxf_M_H=-1.72,
#                 c2=0.35006, c2_err=0.285, c3=2.4788, c3_err=0.262,
#                 lit_dM=31.24, lit_dM_err=0.24, lit_PNLF_N=181, lit_PNLF_N_err=17.8,
#                 app_GALEX_FUV=19.225, app_GALEX_NUV=17.729, GALEX_FUV=19.225, GALEX_NUV=17.729,
#                 R_app=11.28, V_app=11.71, sigma_app=89, sigma_app_err=3.9, GIST_metal=-0.242, GIST_alpha=0.235, GIST_metal_emiles=-0.369)


# # FCC167
# galaxy_df_input("FCC167", "center", lit_Lbol=14370550120.932693, lit_R=9.27, 
#                 sigma=195.3, metal_M_H=0.09, metal_Mg_Fe=0.20, age=13.5, mass=9.85e10, ppxf_M_H=-0.049,
#                 c2=0.277, c2_err=0.206, c3=3.259, c3_err=0.70,
#                 lit_dM=31.35, lit_dM_err=0.15, lit_PNLF_N=319.8, lit_PNLF_N_err=33.5, 
#                 app_GALEX_FUV=18.256, app_GALEX_NUV=16.995, GALEX_FUV=19.23, GALEX_NUV=17.826,
#                 Bl_dM=31.570, Bl_dM_err=0.065, Bl_Bmag=11.3,
#                 R_app=10.75, V_app=11.20, sigma_app=193, sigma_app_err=3.6, GIST_metal=-0.184, GIST_alpha=0.2, GIST_metal_emiles=-0.197 )


# # FCC170
# galaxy_df_input("FCC170", "center", lit_R=10.99, sigma=113, 
#                 metal_M_H=-0.05, metal_Mg_Fe=0.17, age=13.2, mass=0.85e10, ppxf_M_H=0.0873,
#                 c2=0.34205, c2_err=0.345, c3=1.45, c3_err=0.29,
#                 lit_dM=31.69, lit_dM_err=0.28, lit_PNLF_N=245.9, lit_PNLF_N_err=39.4, 
#                 app_GALEX_FUV=19.202, app_GALEX_NUV=17.85, GALEX_FUV=19.666, GALEX_NUV=18.251,
#                 Bl_dM=31.705, Bl_dM_err=0.076, Bl_Bmag=13.0,
#                 R_app=11.36, V_app=11.81, sigma_app=131, sigma_app_err=2.6, GIST_metal=-0.193, GIST_alpha=0.201, GIST_metal_emiles=-0.266)


# # FCC177
# galaxy_df_input("FCC177", "center", lit_R=11.80, sigma=42, 
#                 metal_M_H=-0.14, metal_Mg_Fe=0.11, age=9.8, mass=2.25e10, ppxf_M_H=-0.661,
#                 c2=0.26866, c2_err=0.228, c3=6.99, c3_err=5.04,
#                 lit_dM=31.49, lit_dM_err=0.28, lit_PNLF_N=78, lit_PNLF_N_err=10.8, 
#                 app_GALEX_FUV=19.865, app_GALEX_NUV=18.304, GALEX_FUV=20.064, GALEX_NUV=18.47, 
#                 Bl_dM=31.509, Bl_dM_err=0.065, Bl_Bmag=13.2,
#                 R_app=12.49, V_app=12.90, sigma_app=56, sigma_app_err=8.9, GIST_metal=-0.343, GIST_alpha=0.182, GIST_metal_emiles=-0.458)


# # FCC182
# galaxy_df_input("FCC182", "center", lit_Lbol=915469188.4100325,  lit_R=13.58, sigma=39,
#                 metal_M_H=-0.22, metal_Mg_Fe=0.11, age=12.6, mass=0.15e10, ppxf_M_H=-0.709,
#                 c2=0.19226, c2_err=0.105, c3=9.8, c3_err=9055.4,
#                 lit_dM=31.44, lit_dM_err=0.28, lit_PNLF_N=15.8, lit_PNLF_N_err=5.3, 
#                 app_GALEX_FUV=20.576, app_GALEX_NUV=19.157, GALEX_FUV=20.576, GALEX_NUV=19.157, 
#                 Bl_dM=31.458, Bl_dM_err=0.086, Bl_Bmag=14.9,
#                 R_app=13.84, V_app=14.27, sigma_app=27, sigma_app_err=9.3, GIST_metal=-0.315, GIST_alpha=0.273, GIST_metal_emiles=-0.535)


# # FCC184
# galaxy_df_input("FCC184", "center", lit_Lbol=5785886952.923201,  lit_R=10.00, 
#                  sigma=143, metal_M_H=0.21, metal_Mg_Fe=0.19, age=13.2, mass=4.7e10, ppxf_M_H=-0.133,
#                 c2=0.3048, c2_err=0.414, c3=4.66, c3_err=1.73,
#                 lit_dM=31.41, lit_dM_err=0.28, lit_PNLF_N=92, lit_PNLF_N_err=12.5, 
#                 app_GALEX_FUV=17.058, app_GALEX_NUV=16.33, GALEX_FUV=19.89, GALEX_NUV=18.767, 
#                 Bl_dM=31.430, Bl_dM_err=0.087, Bl_Bmag=12.3,
#                 R_app=10.90, V_app=11.39, sigma_app=156, sigma_app_err=4.3, GIST_metal=-0.017, GIST_alpha=0.17, GIST_metal_emiles=0.004)

# # FCC190
# galaxy_df_input("FCC190", "center", lit_Lbol=2029120876.275461,  lit_R=12.26, 
#                 sigma=75, metal_M_H=-0.13, metal_Mg_Fe=0.16, age=12.9, mass=0.54e10, ppxf_M_H=-0.953,
#                 c2=0.2865, c2_err=0.384, c3=3.346, c3_err=1.81,
#                 lit_dM=31.52, lit_dM_err=0.28, lit_PNLF_N=35.5, lit_PNLF_N_err=8.1, 
#                 app_GALEX_FUV=19.944, app_GALEX_NUV=18.617, GALEX_FUV=19.944, GALEX_NUV=18.617,
#                 Bl_dM=31.540, Bl_dM_err=0.073, Bl_Bmag=13.5,
#                 R_app=12.67, V_app=13.09, sigma_app=75, sigma_app_err=11, GIST_metal=-0.247, GIST_alpha=0.288, GIST_metal_emiles=-0.417)


# # FCC193
# galaxy_df_input("FCC193", "center", lit_Lbol=5369365945.888745, lit_R=10.69, 
#                  sigma=95, metal_M_H=-0.09, metal_Mg_Fe=0.13, age=11.7, mass=3.32e10, ppxf_M_H=-1.17,
#                 c2=0.23146, c2_err=0.086, c3=7.65, c3_err=5.97,
#                 lit_dM=31.42, lit_dM_err=0.22, lit_PNLF_N=237.4, lit_PNLF_N_err=20.3, 
#                 app_GALEX_FUV=19.366, app_GALEX_NUV=17.639, GALEX_FUV=20.012, GALEX_NUV=18.209, 
#                 Bl_dM=31.627, Bl_dM_err=0.072, Bl_Bmag=12.8,
#                 R_app=11.45, V_app=11.87, sigma_app=109, sigma_app_err=4.3, GIST_metal=-0.174, GIST_alpha=0.214, GIST_metal_emiles=-0.297)


# # FCC219
# galaxy_df_input("FCC219", "center", lit_Lbol=13682820566.074028,  lit_R=8.57, 
#                sigma=154, metal_M_H=0.14, metal_Mg_Fe=0.18, age=11.7, mass=12.7e10, ppxf_M_H=0.116,
#                 c2=0.276118, c2_err=0.327, c3=10.0, c3_err=7.38,
#                 lit_dM=31.37, lit_dM_err=0.22, lit_PNLF_N=271.8, lit_PNLF_N_err=36, 
#                 app_GALEX_FUV=19.022, app_GALEX_NUV=17.921, GALEX_FUV=19.022, GALEX_NUV=17.921,
#                 Bl_dM=31.544, Bl_dM_err=0.068, Bl_Bmag=10.9,
#                 R_app=10.35, V_app=10.80, sigma_app=236, sigma_app_err=2.3, GIST_metal=-0.021, GIST_alpha=0.188, GIST_metal_emiles=-0.017)


# # FCC249
# galaxy_df_input("FCC249", "center", lit_Lbol=3346355920.280593,  lit_R=12.07,
#                 sigma=104, metal_M_H=-0.26, metal_Mg_Fe=0.24, age=13.5, mass=0.5e10, ppxf_M_H=-1.25,
#                 c2=0.25524, c2_err=0.384, c3=8.797, c3_err=40.1,
#                 lit_dM=31.82, lit_dM_err=0.24, lit_PNLF_N=66.4, lit_PNLF_N_err=17.7, 
#                 app_GALEX_FUV=20.466, app_GALEX_NUV=19.249, GALEX_FUV=20.466, GALEX_NUV=19.249, 
#                 Bl_dM=31.799, Bl_dM_err=0.082, Bl_Bmag=13.6,
#                 R_app=12.54, V_app=12.95, sigma_app=100, sigma_app_err=7.7, GIST_metal=-0.578, GIST_alpha=0.293, GIST_metal_emiles=-0.664)


# # FCC255
# galaxy_df_input("FCC255", "center", lit_Lbol=1.70372442e+09, lit_Lbol_p=1.08447298e+08, lit_Lbol_m=1.01957397e+08,
#                  lit_R=12.57, sigma=38, metal_M_H=-0.17, metal_Mg_Fe=0.1, age=4.6, mass=0.5e10, ppxf_M_H=0.142,
#                 c2=0.21234, c2_err=0.164, c3=7.34, c3_err=16.58,
#                 lit_dM=31.48, lit_dM_err=0.28, lit_PNLF_N=51., lit_PNLF_N_err=8.7, 
#                 app_GALEX_FUV=20.972, app_GALEX_NUV=19.252, GALEX_FUV=20.972, GALEX_NUV=19.252, 
#                 Bl_dM=31.502, Bl_dM_err=0.067, Bl_Bmag=13.7,
#                 R_app=np.nan, V_app=np.nan, sigma_app=np.nan, sigma_app_err=np.nan, GIST_metal=-0.432, GIST_alpha=0.208, GIST_metal_emiles=-0.567)


# # FCC276
# galaxy_df_input("FCC276", "center", lit_Lbol=6038307972.658742,  lit_R=10.15, 
#                  sigma=123, metal_M_H=-0.25, metal_Mg_Fe=0.20, age=13.8, mass=1.81e10, ppxf_M_H=-0.383,
#                 c2=0.20812, c2_err=0.163, c3=10.0, c3_err=10.6,
#                 lit_dM=31.5, lit_dM_err=0.22, lit_PNLF_N=188.6, lit_PNLF_N_err=22.7, 
#                 app_GALEX_FUV=19.786, app_GALEX_NUV=18.24, GALEX_FUV=19.962, GALEX_NUV=18.397, 
#                 Bl_dM=31.459, Bl_dM_err=0.068, Bl_Bmag=11.8,
#                 R_app=11.33, V_app=11.77, sigma_app=143, sigma_app_err=5.3, GIST_metal=-0.298, GIST_alpha=0.265, GIST_metal_emiles=-0.431)


# # FCC277
# galaxy_df_input("FCC277", "center", lit_Lbol=1383814378.4746728,  lit_R=12.34,  
#                 sigma=80, metal_M_H=-0.34, metal_Mg_Fe=0.11, age=11.7, mass=0.34e10, ppxf_M_H=-0.444,
#                 c2=0.53238, c2_err=0.51, c3=0.0, c3_err=0.39,
#                 lit_dM=31.56, lit_dM_err=0.28, lit_PNLF_N=53.4, lit_PNLF_N_err=10.9, 
#                 app_GALEX_FUV=20.285, app_GALEX_NUV=18.583, GALEX_FUV=20.285, GALEX_NUV=18.583, 
#                 Bl_dM=31.579, Bl_dM_err=0.078, Bl_Bmag=13.8,
#                 R_app=12.88, V_app=13.9, sigma_app=80, sigma_app_err=5.4, GIST_metal=0.374, GIST_alpha=0.018,GIST_metal_emiles=0.367 )


# #FCC301
# galaxy_df_input("FCC301", "center", lit_Lbol=1.62991739e+09, lit_Lbol_p=1.21402957e+08, lit_Lbol_m=1.12987205e+08,
#                 lit_R=12.65, sigma=49, 
#                 metal_M_H=-0.38, metal_Mg_Fe=0.09, age=10.2, mass=0.2e10, ppxf_M_H=-0.521,
#                 c2=0.302544, c2_err=0.388, c3=2.66, c3_err=1.4,
#                 lit_dM=31.06, lit_dM_err=0.24, lit_PNLF_N=30.8, lit_PNLF_N_err=6.7, 
#                 app_GALEX_FUV=np.nan, app_GALEX_NUV=18.502, GALEX_FUV=np.nan, GALEX_NUV=18.524, 
#                 Bl_dM=31.473, Bl_dM_err=0.078, Bl_Bmag=14.2,
#                 R_app=np.nan, V_app=np.nan, sigma_app=np.nan, sigma_app_err=np.nan, GIST_metal=-0.447, GIST_alpha=0.184, GIST_metal_emiles=-0.58)


# # FCC310
# galaxy_df_input("FCC310", "center", lit_Lbol=2278553565.2250648,  lit_R=11.81, 
#                 sigma=48, metal_M_H=-0.30, metal_Mg_Fe=0.14, age=12.0, mass=0.54e10, ppxf_M_H=-0.829,
#                 c2=0.25706, c2_err=0.365, c3=9.057, c3_err=10.40,
#                 lit_dM=31.48, lit_dM_err=0.28, lit_PNLF_N=67.7, lit_PNLF_N_err=10.8, 
#                 app_GALEX_FUV=np.nan, app_GALEX_NUV=18.446, GALEX_FUV=np.nan, GALEX_NUV=18.446,
#                 Bl_dM=31.499, Bl_dM_err=0.065, Bl_Bmag=13.5,
#                 R_app=12.85, V_app=13.27, sigma_app=53, sigma_app_err=7.9, GIST_metal=-0.271, GIST_alpha=0.218, GIST_metal_emiles=-0.427)


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

