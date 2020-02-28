import pandas as pd
import numpy as np

from functions.low_number_stats import poissonLimits

def galaxy_df_input(gal_name, PNe_N, PNLF_N, L_bol, L_bol_p, L_bol_m, R, lit_R, sigma, metal_M_H, metal_Mg_Fe, age, mass=np.nan,
                    D_PNLF=np.nan, D_PNLF_err=np.nan, dM_PNLF=np.nan, dM_PNLF_err=np.nan, lit_dM=np.nan, lit_dM_err=np.nan,
                    FUV=np.nan, FUV_err=np.nan, NUV=np.nan, NUV_err=np.nan, GALEX_FUV=np.nan, GALEX_NUV=np.nan,
                    V=np.nan, B=np.nan, lit_Lbol=np.nan, lit_Lbol_p=np.nan, lit_Lbol_m=np.nan, lit_PNLF_N=np.nan):
    
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "D PNLF"]      = D_PNLF
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "D PNLF err"]  = D_PNLF_err
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "dM PNLF"]     = dM_PNLF
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "dM PNLF err"] = dM_PNLF_err
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "PNe N"]       = PNe_N
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "PNLF N"]      = PNLF_N
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Lbol"]        = L_bol
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Lbol p"]      = L_bol_p
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Lbol m"]      = L_bol_m
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Rmag"]        = R
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit Rmag"]    = lit_R
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "sigma"]       = sigma
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "M/H"]         = metal_M_H
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Mg/Fe"]       = metal_Mg_Fe
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "age"]         = age
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "GALEX FUV"]   = GALEX_FUV
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "FUV"]         = FUV
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "FUV err"]     = FUV_err
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "GALEX NUV"]   = GALEX_NUV
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "NUV"]         = NUV
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "NUV err"]     = NUV_err
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Vmag"]        = V
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Bmag"]        = B
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit dM"]      = lit_dM
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit dM err"]  = lit_dM_err
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit Lbol"]    = lit_Lbol
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit Lbol p"]  = lit_Lbol
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit Lbol m"]  = lit_Lbol
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "lit PNLF N"]  = lit_PNLF_N
    galaxy_df.loc[galaxy_df["Galaxy"]==gal_name, "Mass"]        = mass




galaxy_df = pd.DataFrame(columns=("Galaxy", "D PNLF", "D PNLF err", "dM PNLF", "dM PNLF err", "PNe N", "PNLF N", "N err", "Lbol", "lit Lbol",
                                  "alpha2.5", "alpha2.5 err up", "alpha2.5 err lo", "age", "FUV", "FUV err", "NUV", "NUV err", "Vmag", 
                                  "Rmag", "Bmag", "sigma", "M/H", "Mg/Fe"))

F3D_galaxies = ["FCC083", "FCC143", "FCC147", "FCC153", "FCC161", "FCC167", "FCC170", "FCC177", "FCC182", "FCC184", "FCC190", 
                "FCC193", "FCC219", "FCC249", "FCC255", "FCC276", "FCC277", "FCC301", "FCC310" ]

galaxy_df["Galaxy"] = F3D_galaxies


# FCC083
galaxy_df_input("FCC083", PNe_N=52, PNLF_N=91.2, L_bol=6155921555.012122, L_bol_p=557184936.4709969,L_bol_m=485556182.4988022,
                lit_Lbol=4_851_065_531.00737, R=11.51, lit_R=10.84, 
                V=11.95, sigma=103, metal_M_H=-0.20, metal_Mg_Fe=0.26, age=13.2,
                D_PNLF=18.196, D_PNLF_err=0.789, dM_PNLF=31.405, dM_PNLF_err=0.094,
                lit_dM=31.42, lit_dM_err=0.24, lit_PNLF_N=98.2, GALEX_FUV=18.88, GALEX_NUV=17.35)
#GALEX_FUV=18.5455, GALEX_NUV=17.0531 )


# FCC143
galaxy_df_input("FCC143", PNe_N=15, PNLF_N=61.9, L_bol=1_812_169_020.1260302, L_bol_p=323968639.79834986,L_bol_m=192231433.10896087,
                lit_Lbol=800_333_267.0254638, lit_Lbol_p=998322523.1877589, lit_Lbol_m=641609623.5737106,
                R=13.265, lit_R=12.66, V=13.70, sigma=62, 
                metal_M_H=-0.18, metal_Mg_Fe=0.18, age=12.6,
                D_PNLF=26.39, D_PNLF_err=2.17, dM_PNLF=32.11, dM_PNLF_err=0.179, lit_dM=31.22, lit_dM_err=0.24,
                mass=0.28e10, GALEX_FUV=19.78, GALEX_NUV=18.315)
#GALEX_FUV=19.2737, GALEX_NUV=17.9192)

# FCC147
galaxy_df_input("FCC147", PNe_N=61, PNLF_N=183.3, L_bol=1.04134104e+10, L_bol_p=1.40622972e+09, L_bol_m=9.89996262e+08,
                lit_Lbol=6_134_382_862.249334, R=11.16, lit_R=10.50, 
                V=11.6, sigma=131, metal_M_H=0.04, metal_Mg_Fe=0.23, age=13.5,
                D_PNLF=18.42, D_PNLF_err=1.17, dM_PNLF=31.383, dM_PNLF_err=0.137, lit_dM=31.56, lit_dM_err=0.24, lit_PNLF_N=127.3,
                mass=2.4e10, GALEX_FUV=18.45, GALEX_NUV=17.12)
#GALEX_FUV=18.1972, GALEX_NUV=16.9053)

#FCC153
galaxy_df_input("FCC153", PNe_N=32, PNLF_N=104.9, L_bol=5989757852.46783, L_bol_p=711496784.901515, L_bol_m=556481397.9152727,
                R=11.96, lit_R=11.7, V=12.4, sigma=55, 
                metal_M_H=-0.05, metal_Mg_Fe=0.11, age=10.7,
                D_PNLF=22.09, D_PNLF_err=1.24, dM_PNLF=31.72, dM_PNLF_err=0.12, lit_dM=31.32, lit_dM_err=0.24, lit_PNLF_N=0,
                mass=0.76e10, GALEX_FUV=19.47, GALEX_NUV=17.76)
#GALEX_FUV=18.9519, GALEX_NUV=17.4193)
   
# FCC161
galaxy_df_input("FCC161", PNe_N=97, PNLF_N=172, L_bol=10597915354.533407, L_bol_p=934197602.9970303,L_bol_m=847887353.5454464,
                lit_Lbol=6815479441.900765, R=11.19, lit_R=10.47, 
                V=11.62, sigma=96, metal_M_H=-0.13, metal_Mg_Fe=0.20, age=12.9,
                D_PNLF=20.084, D_PNLF_err=0.846, dM_PNLF=31.514, dM_PNLF_err=0.091, lit_dM=31.24, lit_dM_err=0.24, lit_PNLF_N=133.3,
                mass=2.63e10, GALEX_FUV=18.78, GALEX_NUV=17.17)
#GALEX_FUV=18.4618, GALEX_NUV=16.9434)


# FCC167
galaxy_df_input("FCC167", PNe_N=92, PNLF_N=277, L_bol=16987510574.317451, L_bol_p=1821360647.5004501, L_bol_m=1511982457.9515953,
                lit_Lbol=14370550120.932693, R=10.49, lit_R=9.27, V=10.94, 
                sigma=143, metal_M_H=0.09, metal_Mg_Fe=0.20, age=13.5,
                D_PNLF=17.678, D_PNLF_err=0.906, dM_PNLF=31.237, dM_PNLF_err=0.111, lit_dM=31.35, lit_dM_err=0.15, lit_PNLF_N=312.7,
                mass=9.85e10, GALEX_FUV=17.81, GALEX_NUV=16.44)
#GALEX_FUV=17.6279, GALEX_NUV=16.2929)


# FCC170
galaxy_df_input("FCC170", PNe_N=31, PNLF_N=135, L_bol=10216149315.749546, L_bol_p=1718748046.9701996, L_bol_m=1134665264.7558975,
                R=11.26, lit_R=10.99, V=11.7, sigma=113, 
                metal_M_H=-0.05, metal_Mg_Fe=0.17, age=13.2,
                D_PNLF=19.60, D_PNLF_err=1.52, dM_PNLF=31.46, dM_PNLF_err=0.17, lit_dM=31.69, lit_dM_err=0.28,
                mass=0.85e10, GALEX_FUV=18.76, GALEX_NUV=17.30)
#GALEX_FUV=18.47, GALEX_NUV=17.08)


# FCC177
galaxy_df_input("FCC177", PNe_N=47, PNLF_N=79.4, L_bol=3.79144905e+09, L_bol_p=3.80931732e+08, L_bol_m=3.26280815e+08,
                R=12.35, lit_R=11.80, V=12.76, sigma=42, 
                metal_M_H=-0.14, metal_Mg_Fe=0.11, age=9.8,
                D_PNLF=20.81, D_PNLF_err=1.00, dM_PNLF=31.55, dM_PNLF_err=0.1, lit_dM=31.49, lit_dM_err=0.28,
                mass=2.25e10, GALEX_FUV=19.81, GALEX_NUV=17.91)
#GALEX_FUV=19.093, GALEX_NUV=17.5049)


# FCC182
galaxy_df_input("FCC182", PNe_N=8, PNLF_N=21.7, L_bol=1254300037.5442975, L_bol_p=131350934.99231172, L_bol_m=100376688.62850046,
                lit_Lbol=915469188.4100325, R=13.83, lit_R=13.58, V=14.25, sigma=39,
                metal_M_H=-0.22, metal_Mg_Fe=0.11, age=12.6,
                D_PNLF=23.34, D_PNLF_err=1.16, dM_PNLF=31.84, dM_PNLF_err=0.11, lit_dM=31.44, lit_dM_err=0.28, lit_PNLF_N=15.4,
                mass=0.15e10, GALEX_FUV=20.34, GALEX_NUV=18.96)
#GALEX_FUV=19.424, GALEX_NUV=18.1535)

# FCC184
galaxy_df_input("FCC184", PNe_N=55, PNLF_N=122.1, L_bol=8855816962.601816, L_bol_p=2405447673.0345345, L_bol_m=1385457757.7327204,
                lit_Lbol=5785886952.923201, R=10.00, lit_R=10.00, 
                V=10.69, sigma=143, metal_M_H=0.21, metal_Mg_Fe=0.19, age=13.2,
                D_PNLF=21.74, D_PNLF_err=2.61, dM_PNLF=31.686, dM_PNLF_err=0.261, lit_dM=31.41, lit_dM_err=0.28,
                mass=4.7e10, GALEX_FUV=16.94, GALEX_NUV=16.14)
#GALEX_FUV=16.8681, GALEX_NUV=16.0425) # R=11.67, V=12.12

# FCC190
galaxy_df_input("FCC190", PNe_N=19, PNLF_N=51.1, L_bol=3797263092.554169, L_bol_p=667130971.5393395, L_bol_m=426010820.6655054,
                lit_Lbol=2029120876.275461, R=12.60, lit_R=12.26, V=13.02, 
                sigma=75, metal_M_H=-0.13, metal_Mg_Fe=0.16, age=12.9,
                D_PNLF=23.24, D_PNLF_err=1.88, dM_PNLF=31.831, dM_PNLF_err=0.176, lit_dM=31.52, lit_dM_err=0.28, lit_PNLF_N=26.2,
                mass=0.54e10, GALEX_FUV=19.66, GALEX_NUV=18.20)
#GALEX_FUV=19.0437, GALEX_NUV=17.7354)   


# FCC193
galaxy_df_input("FCC193", PNe_N=115, PNLF_N=235.5, L_bol=8_609_735_520.722036, L_bol_p=700921160.622736,L_bol_m=636891904.3567667,
                lit_Lbol=5369365945.888745, R=11.4, lit_R=10.69, 
                V=11.8, sigma=95, metal_M_H=-0.09, metal_Mg_Fe=0.13, age=11.7,
                D_PNLF=20.102, D_PNLF_err=0.79, dM_PNLF=31.516, dM_PNLF_err=0.085, lit_dM=31.42, lit_dM_err=0.22, lit_PNLF_N=187.9,
                FUV=18.66, FUV_err=0.04, NUV=16.73, NUV_err=0.01,
                mass=3.32e10, GALEX_FUV=18.5455, GALEX_NUV=17.0531)
#GALEX_FUV=18.7433, GALEX_NUV=17.0546)


# FCC219
galaxy_df_input("FCC219", PNe_N=56, PNLF_N=287, L_bol=27133327669.221405, L_bol_p=2491646177.480961, L_bol_m=2214995706.8332253, 
                lit_Lbol=13682820566.074028, R=10.2, lit_R=8.57, 
                V=10.65, sigma=154, metal_M_H=0.14, metal_Mg_Fe=0.18, age=11.7,
                D_PNLF=19.236, D_PNLF_err=0.838, dM_PNLF=31.421, dM_PNLF_err=0.095, lit_dM=31.37, lit_dM_err=0.22, lit_PNLF_N=166.,
                mass=12.7e10, GALEX_FUV=17.02, GALEX_NUV=16.10)
#GALEX_FUV=17.0931, GALEX_NUV=16.1936)


# FCC249
galaxy_df_input("FCC249", PNe_N=13, PNLF_N=56.2, L_bol=3938912485.494244, L_bol_p=890767600.3930507, L_bol_m=501341490.3695078, 
                lit_Lbol=3346355920.280593, R=12.45, lit_R=12.07, V=12.52, 
                sigma=104, metal_M_H=-0.26, metal_Mg_Fe=0.24, age=13.5,
                D_PNLF=21.55, D_PNLF_err=2.2, dM_PNLF=31.67, dM_PNLF_err=0.22, lit_dM=31.82, lit_dM_err=0.24, lit_PNLF_N=33.,
                FUV=19.796, FUV_err=0.162, NUV=17.73, NUV_err=0.051,
                mass=0.5e10, B=13.56, GALEX_FUV=19.17, GALEX_NUV=17.90)
#GALEX_FUV=18.9208, GALEX_NUV=17.5866)


# FCC255
galaxy_df_input("FCC255", PNe_N=35, PNLF_N=64.9, L_bol=2.50698399e+09, L_bol_p=2.0829626e+08, L_bol_m=1.86873335e+08,
                R=13.08, lit_R=12.57, V=13.49, sigma=38, metal_M_H=-0.17, metal_Mg_Fe=0.1, age=4.6,
                D_PNLF=21.90, D_PNLF_err=0.874, dM_PNLF=31.718, dM_PNLF_err=0.087, lit_dM=31.48, lit_dM_err=0.28,
                mass=0.5e10, B=13.86, GALEX_FUV=20.63, GALEX_NUV=17.43)
#GALEX_FUV=19.5599, GALEX_NUV=17.55)


# FCC276
galaxy_df_input("FCC276", PNe_N=62, PNLF_N=200.7, L_bol=9828022445.634739, L_bol_p=1003863014.2505722, L_bol_m=853635244.0624256,
                lit_Lbol=6038307972.658742, R=11.2, lit_R=10.15, 
                V=11.63, sigma=123, metal_M_H=-0.25, metal_Mg_Fe=0.20, age=13.8,
                D_PNLF=19.59, D_PNLF_err=0.95, dM_PNLF=31.46, dM_PNLF_err=0.11, lit_dM=31.5, lit_dM_err=0.22, lit_PNLF_N=138.9,
                mass=1.81e10, GALEX_FUV=18.09, GALEX_NUV=16.91)
#GALEX_FUV=18.0881, GALEX_NUV=16.7835)

# FCC277
galaxy_df_input("FCC277", PNe_N=23, PNLF_N=57.1, L_bol=2.07358811e+09, L_bol_p=2.28097039e+08, L_bol_m=1.862613e+08,
                lit_Lbol=1383814378.4746728, R=13.199, lit_R=12.34, V=13.595, 
                D_PNLF=21.512, D_PNLF_err=1.12, dM_PNLF=31.662, dM_PNLF_err=0.113, lit_dM=31.56, lit_dM_err=0.28, lit_PNLF_N=42.3,
                sigma=80, metal_M_H=-0.34, metal_Mg_Fe=0.11, age=11.7, FUV=20.283, FUV_err=0.229,
                NUV=17.962, NUV_err=0.052, B=13.77, mass=0.34e10, GALEX_FUV=19.97, GALEX_NUV=17.92)
#GALEX_FUV=19.1821, GALEX_NUV=17.5983)


#FCC301
galaxy_df_input("FCC301", PNe_N=23, PNLF_N=36.8, L_bol=1.12183761e+09, L_bol_p=1.66204304e+08,L_bol_m=1.44757898e+08, R=15.33, lit_R=12.65, 
                V=15.58, sigma=49, metal_M_H=-0.38, metal_Mg_Fe=0.09, age=10.2,
                D_PNLF=16.56, D_PNLF_err=1.14, dM_PNLF=31.096, dM_PNLF_err=0.15, lit_dM=31.06, lit_dM_err=0.24,
                mass=0.2e10, B=14.08, GALEX_NUV=18.25)
#GALEX_NUV=17.8031)


# FCC310
galaxy_df_input("FCC310", PNe_N=41, PNLF_N=84.2, L_bol=2.76310885e+09, L_bol_p=2.73429571e+08, L_bol_m=2.31830967e+08,
                lit_Lbol=2278553565.2250648, R=12.776, lit_R=11.81, V=13.211,
                sigma=48, metal_M_H=-0.30, metal_Mg_Fe=0.14, age=12.0,
                D_PNLF=21.24, D_PNLF_err=1.00, dM_PNLF=31.699, dM_PNLF_err=0.10, lit_dM=31.48, lit_dM_err=0.28, lit_PNLF_N=70.0,
                mass=0.54e10, B=13.52, GALEX_NUV=17.449)



galaxy_df["N err"]  = (1/(galaxy_df["PNe N"]**(0.5)))*galaxy_df["PNLF N"]
galaxy_df["alpha2.5"] = (galaxy_df["PNLF N"]/galaxy_df["Lbol"]).astype(float)
galaxy_df["log alpha2.5"] = np.log10(galaxy_df["alpha2.5"].values.astype(float))

up_lo = np.array([poissonLimits(n) for n in galaxy_df["PNe N"]])
n_err = [[pn/np.sqrt(e[0]), pn/np.sqrt(e[1])] for e,pn in zip( up_lo, galaxy_df["PNLF N"])]

alpha_err_upper = [a * np.sqrt( (n_e[0]/N)**2  + (L_up/L)**2 ) for a, n_e, N, L_up, L in zip(galaxy_df["alpha2.5"],n_err,
                                                                                galaxy_df["PNLF N"],galaxy_df["Lbol p"],galaxy_df["Lbol"] )  ]

alpha_err_lower = [a * np.sqrt( (n_e[1]/N)**2  + (L_lo/L)**2 ) for a, n_e, N, L_lo, L in zip(galaxy_df["alpha2.5"],n_err,
                                                                                galaxy_df["PNLF N"],galaxy_df["Lbol m"],galaxy_df["Lbol"] )  ]

galaxy_df["alpha2.5 err up"] = alpha_err_upper
galaxy_df["alpha2.5 err lo"] = alpha_err_lower

galaxy_df["log alpha2.5 err up"] = 0.434 * (galaxy_df["alpha2.5 err up"] / galaxy_df["alpha2.5"])
galaxy_df["log alpha2.5 err lo"] = 0.434 * (galaxy_df["alpha2.5 err lo"] /  galaxy_df["alpha2.5"])

# save galaxy_df to fits for safe keeping
galaxy_df.to_csv("exported_data/galaxy_dataframe.csv")

#with open("exported_data/galaxy_dataframe.csv", mode="a") as file:
#    galaxy_df.to_csv(file, header=False)