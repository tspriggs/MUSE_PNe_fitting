import numpy as np
from astropy.io import fits
import gc
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import pandas as pd
from PNLF import open_data, reconstructed_image, KS2_test, completeness
from ppxf_gal_L import ppxf_L_tot
from scripts.low_number_stats import poissonLimits


###
# PNLF and Lbol from lit distances

############################
def run_PNLF(gal, loc, M_5007, m_5007, lit_dist, gal_params, PSF_params, z):
    galaxy_image, wave, hdr = reconstructed_image(gal, "center")
    y_data = hdr["NAXIS2"]
    x_data = hdr["NAXIS1"]
    galaxy_image = galaxy_image.reshape([y_data, x_data])
    
    
    # Total PNLF
    PNLF, PNLF_corr, completeness_ratio, Abs_M, app_m = completeness(gal, loc, M_5007, PSF_params, lit_dist, galaxy_image, peak=3.0,
                                          gal_mask_params=gal_params["gal_mask"], star_mask_params=gal_params["star_mask"], c1=0.307, z=z ) # Estimating the completeness for the central pointing
    
    step = abs(Abs_M[1]-Abs_M[0])
    # Getting the normalisation - sum of correctied PNLF, times bin size
    total_norm = np.sum(np.abs(PNLF_corr)) * step
    
    # Scaling factor
    scal = len(PNe_mag) / total_norm
    
    # Constraining to -2.0 in magnitude
    idx = np.where(Abs_M <= np.min(PNe_mag)+2.5)
    # Total number of PNe
    # tot_N_PNe = np.sum(PNLF_corr[idx]*scal)*abs(Abs_M[1]-Abs_M[0])
    # tot_N_PNe = np.sum(PNLF[idx]*scal) * step
    
    
    
    plt.figure(figsize=(14,10))
    
    binwidth = 0.2
    
    # hist = plt.hist(PNe_mag, bins=np.arange(min(PNe_mag), max(PNe_mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')
    hist = plt.hist(app_mag, bins=np.arange(min(app_mag), max(app_mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')
    
    KS2_stat = KS2_test(dist_1=PNLF_corr[1:18:2]*scal*binwidth, dist_2=hist[0], conf_lim=0.1)
    print(KS2_stat)
    
#     ymax = max(hist[0])
    
#     plt.plot(app_m, PNLF*scal*binwidth, '-', color='blue', marker="o", label=f"{galaxy_name} - PNLF")
#     plt.plot(app_m, PNLF_corr*scal*binwidth,'-.', color='blue', label="Incompleteness corrected PNLF")
    # plt.plot(Abs_M, completeness_ratio*200*binwidth, "--", color="k", label="completeness")
#     plt.xlabel(r'$m_{5007}$', fontsize=30)
#     plt.ylabel(r'$N_{PNe}$', fontsize=30)
    #plt.yticks(np.arange(0,ymax+4, 5))
#     plt.plot(0,0, alpha=0.0, label=f"KS2 test = {round(KS2_stat[0],3)}")
#     plt.plot(0,0, alpha=0.0, label=f"pvalue   = {round(KS2_stat[1],3)}")
#     plt.xlim(-5.0+dM,-1.5+dM); 
#     plt.ylim(0,ymax+(2*ymax));
    # plt.xlim(26.0,30.0); plt.ylim(0,45);
    
#     plt.tick_params(labelsize = 25)
    
    #plt.axvline(PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].values.min() - 31.63)
#     plt.legend(loc=2, fontsize=20)
#     plt.savefig(PLOT_DIR+"_PNLF.pdf", bbox_inches='tight')
#     plt.savefig(PLOT_DIR+"_PNLF.png", bbox_inches='tight')
    
    
    N_PNe = np.sum(PNLF[idx]*scal) * step
    
    # print("Number of PNe from PNLF: ", N_PNe, "+/-", (1/np.sqrt(len(PNe_df.loc[PNe_df["Filter"]=="Y"])))*N_PNe)
    print("Number of PNe from PNLF: ", N_PNe, "+/-", (1/np.sqrt(len(PNe_df.loc[PNe_df["Filter"]=="Y"])))*N_PNe)
    
    galaxy_image = []
    gc.collect()
    
    return N_PNe, completeness_ratio, app_m, Abs_M


#############################################################################################################################
def run_Lbol(gal, loc, lit_dM, gal_params, PSF_params, z, dM_err):
    raw_data_cube = f"/local/tspriggs/Fornax_data_cubes/{gal}{loc}.fits"
 # read in raw data cube

    xe, ye, length, width, alpha = gal_params["gal_mask"]
    
    with fits.open(raw_data_cube) as orig_hdulist:
        raw_data_cube = np.copy(orig_hdulist[1].data)
        h1 = orig_hdulist[1].header
        
    s = np.shape(raw_data_cube)
    Y, X = np.mgrid[:s[1], :s[2]]
    elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1    
    
    # Now mask the stars
    star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in gal_params["star_mask"]],0).astype(bool)
    
    total_mask = ((np.isnan(raw_data_cube[1,:,:])==False) & (elip_mask==False) & (star_mask_sum==False))
    indx_mask = np.where(total_mask==True)
    
    good_spectra = np.zeros((s[0], len(indx_mask[0])))
    
    for i, (y, x)  in enumerate(zip(indx_mask[0], indx_mask[1])):
        good_spectra[:,i] = raw_data_cube[:,y,x]
    
    print("Collapsing cube now....")    
        
    gal_lin = np.nansum(good_spectra, 1)
            
    print("Cube has been collapsed...")
    
    ## L_bol = lum_bol_g, lum_bol_r, mag_r, M_r
    L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_params["velocity"], dist_mod=lit_dM, dM_err=[dM_err, dM_err])
    
    raw_data_cube = []
    elip_mask = []
    star_mask_sum = []
    total_mask = []
    indx_mask = []
    good_spextra = []    
    gal_lin = []
    gc.collect()
    
    return L_bol

c = 299792458.0
with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    



galaxy_df = pd.read_csv(f"exported_data/galaxy_dataframe.csv")

lit_galaxy_df = pd.DataFrame(columns=("Galaxy", "lit PNLF N", "lit Lbol", "lit alpha2.5", "lit alpha2.5 err up", "lit alpha2.5 err lo",))

lit_galaxy_df["Galaxy"] = galaxy_df["Galaxy"]


loc = "center"
N_PNe_gal_list = []
completeness_ratio_list = []
app_m_list = []
Abs_M_list = []

for i, gal in enumerate(tqdm(galaxy_df["Galaxy"])):
    galaxy_data = galaxy_info[f"{gal}_{loc}"]
    # Read in galaxies fitted PNe dataframes
    PNe_df = pd.read_csv(f"exported_data/{gal}/{gal}_PNe_df.csv")
       
    
#     lit_dM = galaxy_df.loc[galaxy_df["Galaxy"]==gal, "lit dM"].values
#     lit_dM_err = galaxy_df.loc[galaxy_df["Galaxy"]==gal, "lit dM err"].values
    
    lit_dM = galaxy_df.loc[galaxy_df["Galaxy"]==gal, "dM PNLF"].values
    lit_dM_err = galaxy_df.loc[galaxy_df["Galaxy"]==gal, "dM PNLF err"].values
    lit_dist = 10**((lit_dM -25) / 5)
    PNe_df["lit M 5007"] = PNe_df["m 5007"] - lit_dM
    z = galaxy_data["velocity"]*1e3 / c
    
    PNe_mag = PNe_df["lit M 5007"].loc[PNe_df["Filter"]=="Y"].values
    app_mag = PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].values
    
    PSF_params = {"M_FWHM":galaxy_data["FWHM"],
                  "beta":galaxy_data["beta"],
                  "LSF": galaxy_data["LSF"]}
    
    lit_N_PNe, comp_ratio, app_m, Abs_M = run_PNLF(gal, loc, PNe_mag, app_mag, lit_dist, galaxy_data, PSF_params, z)
    
    completeness_ratio_list.append(comp_ratio)
    app_m_list.append(app_m)
    Abs_M_list.append(Abs_M)
    
    
    
completeness_ratio_list = np.array(completeness_ratio_list)
np.save("exported_data/completeness_ratios", completeness_ratio_list)

app_m_list = np.array(app_m_list)
np.save("exported_data/app_m_list_for_comp_r", app_m_list)

Abs_M_list = np.array(Abs_M_list)
np.save("exported_data/Abs_M_list_for_comp_r", Abs_M_list)
#     lit_galaxy_df.loc[lit_galaxy_df["Galaxy"]==gal, "lit PNLF N"] = lit_N_PNe
#     lit_galaxy_df.loc[lit_galaxy_df["Galaxy"]==gal, "lit PNLF N"] = lit_N_PNe
    
#     if gal not in ["FCC143", "FCC255"]:
#         lit_Lbol = run_Lbol(gal, loc, lit_dM, galaxy_data, PSF_params, z, lit_dM_err)
        
#         lit_galaxy_df.loc[lit_galaxy_df["Galaxy"]==gal, "lit Lbol"] = lit_Lbol[0]
#         lit_galaxy_df.loc[lit_galaxy_df["Galaxy"]==gal, "lit Lbol err p"] = lit_Lbol[1][0] - lit_Lbol[0]
#         lit_galaxy_df.loc[lit_galaxy_df["Galaxy"]==gal, "lit Lbol err m"] = lit_Lbol[0] - lit_Lbol[1][1]


# lit_galaxy_df["lit alpha2.5"] = lit_galaxy_df["lit PNLF N"] / lit_galaxy_df["lit Lbol"]
# up_lo = np.array([poissonLimits(n) for n in galaxy_df["PNe N"]])
# n_err = [[pn/np.sqrt(e[0]), pn/np.sqrt(e[1])] for e,pn in zip( up_lo, lit_galaxy_df["lit PNLF N"])]

# alpha_err_upper = [a * np.sqrt( (n_e[0]/N)**2  + (L_up/L)**2 ) for a, n_e, N, L_up, L in zip(lit_galaxy_df["lit alpha2.5"],n_err,
#                                                                                 lit_galaxy_df["lit PNLF N"],lit_galaxy_df["lit Lbol err p"], lit_galaxy_df["lit Lbol"] )  ]

# alpha_err_lower = [a * np.sqrt( (n_e[1]/N)**2  + (L_lo/L)**2 ) for a, n_e, N, L_lo, L in zip(lit_galaxy_df["lit alpha2.5"],n_err,
#                                                                                 lit_galaxy_df["lit PNLF N"],lit_galaxy_df["lit Lbol err m"],lit_galaxy_df["lit Lbol"] )  ]

# lit_galaxy_df["lit alpha2.5 err up"] = alpha_err_upper
# lit_galaxy_df["lit alpha2.5 err lo"] = alpha_err_lower


# lit_galaxy_df["lit log alpha2.5"] = np.log10(lit_galaxy_df["lit alpha2.5"].values.astype(float))
# lit_galaxy_df["lit log alpha2.5 err up"] = 0.434 * (lit_galaxy_df["lit alpha2.5 err up"] / lit_galaxy_df["lit alpha2.5"])
# lit_galaxy_df["lit log alpha2.5 err lo"] = 0.434 * (lit_galaxy_df["lit alpha2.5 err lo"] / lit_galaxy_df["lit alpha2.5"])


# lit_galaxy_df.to_csv("exported_data/lit_galaxy_df.csv")
