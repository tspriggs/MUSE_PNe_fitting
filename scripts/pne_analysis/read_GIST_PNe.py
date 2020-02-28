import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.patches import Rectangle, Ellipse, Circle
import matplotlib.gridspec as gridspec

from functions.file_handling import paths

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument('--w', action='store_true', default=False)
my_parser.add_argument('--n', action='store_true', default=False)
my_parser.add_argument('--p', action='store_true', default=False)
my_parser.add_argument('--save', action='store_true', default=False)

args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy 

weighted = args.w

show_plot = args.p

plt_save = args.save

# Define Working directory
if weighted == True:
    gist_dir = f"../../gist_PNe/results/{galaxy_name}MUSEPNeweighted_contamination/{galaxy_name}MUSEPNeweighted"
else:
    gist_dir = f"../../gist_PNe/results/{galaxy_name}MUSEPNe_contamination/{galaxy_name}MUSEPNe"

DIR_dict = paths(galaxy_name, "center")

# read in PNe data
PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")
m_5007 = PNe_df["m 5007"]

# Open and name the following result files
gandalf_emission = fits.open(f"{gist_dir}_gandalf-emission_SPAXEL.fits")
gandalf_best     = fits.open(f"{gist_dir}_gandalf-bestfit_SPAXEL.fits")
gandalf_clean    = fits.open(f"{gist_dir}_gandalf-cleaned_SPAXEL.fits")
gandalf_results  = fits.open(f"{gist_dir}_gandalf_SPAXEL.fits")
AllSpectra       = fits.open(f"{gist_dir}_AllSpectra.fits")
ppxf_results     = fits.open(f"{gist_dir}_ppxf.fits")
lamb = np.array(AllSpectra[2].data, dtype=np.float64)
wave = np.exp(lamb)
# [OIII] index 12
# Hb index 11
# [NII] index 18
# Ha index 17
# [SII] I index 19
# [SII] II index 20


###################################################################################################
############################## VELOCITY WORK ######################################################
###################################################################################################

# extract PN vel
# PN_vel = np.array([gandalf_results[2].data[i]["V"][12] for i in np.arange(len(gandalf_results[2].data))])
# # PN_vel = PNe_df["V (km/s)"].loc[PNe_df["Filter"]=="Y"]+60

# v_star     = np.array(ppxf_results[1].data["V"])
# sigma_star = np.array(ppxf_results[1].data["SIGMA"])

# f_ind = PNe_df.loc[PNe_df["Filter"]=="Y"].index.values # index for filtered PN

# vel_ratio = (PN_vel - v_star) / sigma_star


# plt.figure(figsize=(10,8))
# plt.hist(vel_ratio[f_ind], bins=10, edgecolor="k", alpha=0.8, linewidth=1)
# plt.xlabel(r"$\rm \frac{V_{PNe} - V_{*}}{\sigma_*}$", fontsize=30)
# plt.ylabel("PNe count", fontsize=30, labelpad=10)
# plt.tick_params(labelsize = 18)
# plt.xlim(-3.25,3.25)

# # if plt_save == True:
#     # plt.savefig(f"Plots/{galaxy_name}/{galaxy_name}_velocity_bins_plot.pdf", bbox_inches='tight')




# hdulist_ppxf = fits.open(RAW_DIR+f"/{galaxy_name}center_ppxf_SPAXELS.fits")
# v_star, s_star = hdulist_ppxf[1].data.V, hdulist_ppxf[1].data.SIGMA


# hdulist_table = fits.open(RAW_DIR+f"/{galaxy_name}center_table.fits")
# X_star, Y_star = hdulist_table[1].data.XBIN, hdulist_table[1].data.YBIN
# flux_star = hdulist_table[1].data.FLUX

# idx = flux_star.argmax()
# X_star, Y_star = X_star-X_star[idx], Y_star-Y_star[idx]

# cond = np.sqrt( (X_star)**2 + (Y_star)**2 ) <= 5.0
# vsys = np.median(v_star[cond])
# v_star = v_star-vsys

# LOS_z = (vsys * 1e3) / c

# LOS_de_z = np.array(mean_wave_list[:,0] / (1 + LOS_z))
    
# PNe_df["PNe_LOS_V"] = (c * (LOS_de_z - 5006.77) / 5006.77) / 1000. 

# f_ind = PNe_df.loc[PNe_df["Filter"]=="Y"].index

# gal_centre_pix = Table.read("exported_data/galaxy_centre_pix.dat", format="ascii")

# gal_ind = np.where(gal_centre_pix["Galaxy"]==galaxy_name)
# gal_x_c = gal_centre_pix["x_pix"][gal_ind]
# gal_y_c = gal_centre_pix["y_pix"][gal_ind]

# xpne, ypne = (x_PNe[f_ind]-gal_x_c)*0.2, (y_PNe[f_ind]-gal_y_c)*0.2

# # Estimating the velocity dispersion of the PNe along the LoS
# def sig_PNe(X_star,Y_star,v_stars,sigma,x_PNe,y_PNe,vel_PNe):

#     d_PNe_to_skin = np.zeros(len(x_PNe))
#     Vs_PNe = np.ones(len(x_PNe)) # Velocity of the closest star
#     Ss_PNe = np.ones(len(x_PNe)) # Sigma for each PNe
#     i_skin_PNe = []

#     """ To estimate the velocity dispersion for PNe we need to
#     extract the sigma of the closest stars for each PNe """

#     for i in range(len(x_PNe)):
#         r_tmp = np.sqrt((X_star-x_PNe[i])**2+(Y_star-y_PNe[i])**2)
#         d_PNe_to_skin[i] = min(r_tmp)
#         i_skin_PNe.append(r_tmp.argmin())

#     Vs_PNe  = v_stars[i_skin_PNe]
#     Ss_PNe  = sigma[i_skin_PNe]
#     rad_PNe = np.sqrt(x_PNe**2+y_PNe**2)
#     k = np.where(d_PNe_to_skin > 1.0)

#     return rad_PNe, (vel_PNe-Vs_PNe)/Ss_PNe, k, Vs_PNe, Ss_PNe

# rad_PNe, vel_ratio, k, Vs_PNe, Ss_PNe  = sig_PNe(X_star, Y_star, v_star, s_star, xpne, ypne, PNe_df["PNe_LOS_V"].loc[PNe_df["Filter"]=="Y"])


# interlopers = vel_ratio[(vel_ratio<-3) | (vel_ratio>3)].index



# plt.figure(figsize=(10,6))
# plt.scatter(PN_vel, PN_vel_ppxf[Y_ind])
# plt.scatter(v_star_orig[:-8], v_star[Y_ind])
# plt.scatter(sigma_star_orig[:-8], sigma_star[Y_ind])
# plt.plot(np.arange(-400,300), np.arange(-400,300), c="k", lw=2,label="y=x", ls="--")
# plt.plot(np.arange(-400,300), np.arange(-400,300)+10, c="r", lw=2, label="y=x+10", ls="--")

# plt.ylabel("GIST ppxf star sigma (km/s)")
# plt.xlabel("spaxel by spaxel star sigma (km/s)")

# plt.xlim(0,250)
# plt.ylim(0,250)
# plt.plot(np.arange(-200,250), np.arange(-200,250), c="k", lw=2, ls="--", label="y=x")
# plt.plot(np.arange(0,250), np.arange(0,250), c="k", lw=2)
# plt.legend()
# plt.savefig("Plots/FCC167_GIST_vs_3D_fit_PN_vel.png")
# plt.savefig("Plots/FCC167_GIST_vs_3D_fit_star_vel.png")
# plt.savefig("Plots/FCC167_GIST_vs_3D_fit_star_sigma.png")





#####################################################################################################
######################################### IMPOSTOR SECTION  #########################################
#####################################################################################################

# AMPLITUDES
Hb_ampl     = np.array([gandalf_results[2].data[i]["Ampl"][11] for i in np.arange(len(gandalf_results[2].data))])
OIII_ampl   = np.array([gandalf_results[2].data[i]["Ampl"][12] for i in np.arange(len(gandalf_results[2].data))])
Ha_ampl     = np.array([gandalf_results[2].data[i]["Ampl"][17] for i in np.arange(len(gandalf_results[2].data))])
NII_ampl    = np.array([gandalf_results[2].data[i]["Ampl"][18] for i in np.arange(len(gandalf_results[2].data))])
SII_I_ampl  = np.array([gandalf_results[2].data[i]["Ampl"][19] for i in np.arange(len(gandalf_results[2].data))])
SII_II_ampl = np.array([gandalf_results[2].data[i]["Ampl"][20] for i in np.arange(len(gandalf_results[2].data))])

# FLUXES
# Hb_flux     = np.array([gandalf_results[2].data[i]["Flux"][11] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
# OIII_flux   = np.array([gandalf_results[2].data[i]["Flux"][12] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
# Ha_flux     = np.array([gandalf_results[2].data[i]["Flux"][17] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
# NII_flux    = np.array([gandalf_results[2].data[i]["Flux"][18] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
# SII_I_flux  = np.array([gandalf_results[2].data[i]["Flux"][19] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
# SII_II_flux = np.array([gandalf_results[2].data[i]["Flux"][20] for i in np.arange(len(gandalf_results[2].data))])* 1e-20
Hb_flux     = np.array([gandalf_results[2].data[i]["Ampl"][11]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
OIII_flux   = np.array([gandalf_results[2].data[i]["Ampl"][12]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
Ha_flux     = np.array([gandalf_results[2].data[i]["Ampl"][17]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
NII_flux    = np.array([gandalf_results[2].data[i]["Ampl"][18]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
SII_I_flux  = np.array([gandalf_results[2].data[i]["Ampl"][19]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
SII_II_flux = np.array([gandalf_results[2].data[i]["Ampl"][20]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20

# SIGNAL TO NOISE (AON)
Hb_AON     = np.array([gandalf_results[2].data[i]["AON"][11] for i in np.arange(len(gandalf_results[2].data))])
OIII_AON   = np.array([gandalf_results[2].data[i]["AON"][12] for i in np.arange(len(gandalf_results[2].data))])
Ha_AON     = np.array([gandalf_results[2].data[i]["AON"][17] for i in np.arange(len(gandalf_results[2].data))])
NII_AON    = np.array([gandalf_results[2].data[i]["AON"][18] for i in np.arange(len(gandalf_results[2].data))])
SII_I_AON  = np.array([gandalf_results[2].data[i]["AON"][19] for i in np.arange(len(gandalf_results[2].data))])
SII_II_AON = np.array([gandalf_results[2].data[i]["AON"][20] for i in np.arange(len(gandalf_results[2].data))])

# Create a dataframe called gand_df, for storing the above info.
gand_df = pd.DataFrame(index=np.arange(0,len(PNe_df)), columns=("Hb amp","Hb flux", "Hb AON",
             "[OIII] amp", "corr [OIII] amp","[OIII] flux","corr [OIII] flux", "[OIII] AON",
             "Ha amp", "Ha flux", "Ha AON", 
             "[NII] amp", "[NII] flux", "[NII] AON",
             "[SII]I amp", "[SII]I flux", "[SII]I AON",
             "[SII]II amp", "[SII]II flux", "[SII]II AON"))

gand_df["Hb amp"]          = Hb_ampl
gand_df["Hb flux"]         = Hb_flux
gand_df["Hb AON"]          = Hb_AON
gand_df["[OIII] amp"]      = OIII_ampl
gand_df["[OIII] flux"]     = OIII_flux
gand_df["[OIII] AON"]      = OIII_AON
gand_df["Ha amp"]          = Ha_ampl
gand_df["Ha flux"]         = Ha_flux
gand_df["Ha AON"]          = Ha_AON
gand_df["[NII] amp"]       = NII_ampl
gand_df["[NII] flux"]      = NII_flux
gand_df["[NII] AON"]       = NII_AON
gand_df["[SII]I amp"]      = SII_I_ampl
gand_df["[SII]I flux"]     = SII_I_flux
gand_df["[SII]I AON"]      = SII_I_AON
gand_df["[SII]II amp"]     = SII_II_ampl
gand_df["[SII]II flux"]    = SII_II_flux
gand_df["[SII]II AON"]     = SII_II_AON

# put m5007 values into gand_df
gand_df["m 5007"] = m_5007
gand_df["ID"] = PNe_df["ID"]

# below is to check all sources, not just pre-filtered sources.
# gand_df["ID"]="PN"

# Read in the distance modulus for the galaxy in question (galaxy_name)
gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv")
dM = gal_df["dM PNLF"].loc[gal_df["Galaxy"]==galaxy_name].values

# set up conditions


###########################################################################################################################
#########################################    Ha/NII conditions    #########################################################
###########################################################################################################################
# First plot conditions
# Ha or NII over 3, for ID's that are PN
HaNII_or_cond = ((gand_df["Ha AON"] >= 3.) | (gand_df["[NII] AON"]>=3.)) & (gand_df["ID"]!="-")  

# Ha and NII both over 3 times the signal to noise, and ID is PN
HaNII_and_cond = ((gand_df["Ha AON"] >= 3.) & (gand_df["[NII] AON"]>=3.)) & (gand_df["ID"]!="-")


# Just Ha over AON of 3
# HaNII_or_cond = (gand_df["Ha AON"] >= 3.) & (gand_df["ID"]=="PN")

# Just Ha over AON of 3, but with different variable name (easy to switch over)
# HaNII_and_cond = (gand_df["Ha AON"] > 3.) & (gand_df["ID"]=="PN")  


###########################################################################################################################
###################################    Ha/NII with SII filters    #########################################################
###########################################################################################################################

# ( (Ha or NII over 3), and SII over 3) and ID==PN
HaNII_SII_or_cond = (((gand_df["Ha AON"] >= 3.) | (gand_df["[NII] AON"] >= 3.)) & np.greater_equal((gand_df["[SII]I AON"] + gand_df["[SII]II AON"]) / 2., 2.5)) & (gand_df["ID"]!="-")

# ( (Ha and NII over 3), and SII over 3) and ID==PN
HaNII_SII_and_cond = (((gand_df["Ha AON"] >= 3.) & (gand_df["[NII] AON"] >= 3.)) & np.greater_equal((gand_df["[SII]I AON"] + gand_df["[SII]II AON"]) / 2., 2.5)) & (gand_df["ID"]!="-")

# Filters for just Ha and SII meeting the conditions
# HaNII_SII_or_cond = ((gand_df["Ha AON"] > 3.) & np.greater_equal((gand_df["[SII]I AON"] + gand_df["[SII]II AON"])/2, 2.5)) & (gand_df["ID"]=="PN")
# HaNII_SII_and_cond = ((gand_df["Ha AON"] > 3.)  & np.greater_equal((gand_df["[SII]I AON"] + gand_df["[SII]II AON"])/2, 2.5)) & (gand_df["ID"]=="PN")


######## Filter OIII / Ha+NII vs m_5007 data #############################
# Flux_[OIII] / F_Ha + (1.34 * F_NII) where Ha or NII are above 3 times AON
HaNII_ratio_cond_or = gand_df["[OIII] flux"].loc[HaNII_or_cond].values / (gand_df["Ha flux"].loc[HaNII_or_cond].values + 1.34*gand_df["[NII] flux"].loc[HaNII_or_cond].values)

# Flux_[OIII] / F_Ha + (1.34 * F_NII) where Ha and NII are above 3 times AON
HaNII_ratio_cond_and = gand_df["[OIII] flux"].loc[HaNII_and_cond].values / (gand_df["Ha flux"].loc[HaNII_and_cond].values + 1.34*gand_df["[NII] flux"].loc[HaNII_and_cond].values)

# F_[OIII] / F_Ha + (1.34 * F_NII) where (Ha or NII are over 3 times AON), and SII AON over 3 and ID==PN
HaNII_ratio_SII = gand_df["[OIII] flux"].loc[HaNII_SII_or_cond].values / (gand_df["Ha flux"].loc[HaNII_SII_or_cond].values + 1.34*gand_df["[NII] flux"].loc[HaNII_SII_or_cond].values)


############## NII correction ##############
# calcualte the ratio of all fluxes, regardless of AON
ratio = OIII_flux / (Ha_flux+ 1.34*NII_flux)

# Make a copy of NII flux array
F_NII_corr = np.copy(NII_flux)

# find where NII AON is less than 3
NII_AON_lt_3 = np.where(NII_AON < 3.)

# Where NII AON is less than 3: do 3 * F_NII / AON_NII
F_NII_corr[NII_AON_lt_3] = 3*(NII_flux[NII_AON_lt_3] / NII_AON[NII_AON_lt_3])

# Now calculate the ratio correction: F_[OIII] / F_Ha + (1.34 * corrected F_NII)
ratio_corr = np.copy(OIII_flux) / (np.copy(Ha_flux) + 1.34 * F_NII_corr)


############## SII correction ##############
# Same as for NII, we are correcting the values of SII where SII is detected below a combined AON of 2.5, so as to estimate limits to the values.
HaoNII = Ha_flux / NII_flux
HaoSII = Ha_flux / (SII_I_flux + SII_II_flux)

SII_I_lt_3 = np.where(SII_I_AON < 2.5)
SII_II_lt_3 = np.where(SII_II_AON < 2.5)

SII_I_F_corr = np.copy(SII_I_flux)
SII_II_F_corr = np.copy(SII_II_flux)

SII_I_F_corr[SII_I_lt_3] = 3 * (SII_I_flux[SII_I_lt_3] / SII_I_AON[SII_I_lt_3])
SII_II_F_corr[SII_II_lt_3] = 3 * (SII_II_flux[SII_II_lt_3] / SII_II_AON[SII_II_lt_3])

HaoSII_corr = Ha_flux / (SII_I_F_corr+SII_II_F_corr)


################################ Filter Ha/NII vs Ha/SII data ################################
# F_Ha / F_NII where Ha and NII are detected above AON of 3, and ID == PN
Ha_NII = gand_df["Ha flux"].loc[HaNII_and_cond].values / gand_df["[NII] flux"].loc[HaNII_and_cond].values

# F_Ha / (F_SII I + F_SII II), where Ha and NII are detected above AON of 3, and ID == PN
Ha_SII = gand_df["Ha flux"].loc[HaNII_and_cond].values / (gand_df["[SII]I flux"].loc[HaNII_and_cond].values + gand_df["[SII]II flux"].loc[HaNII_and_cond].values)

# F_Ha / F_NII where Ha and NII are detected above AON of 3, along with combined SII detected above 2.5, and ID == PN
Ha_NII_1 = gand_df["Ha flux"].loc[HaNII_SII_and_cond].values / gand_df["[NII] flux"].loc[HaNII_SII_and_cond].values

# F_Ha / (F_SII I + F_SII II), where Ha and NII are detected above AON of 3, along with combined SII detected above 2.5, and ID == PN
Ha_SII_1 = gand_df["Ha flux"].loc[HaNII_SII_and_cond].values / (gand_df["[SII]I flux"].loc[HaNII_SII_and_cond].values + gand_df["[SII]II flux"].loc[HaNII_SII_and_cond].values)


################################################################################################
###########################   Plotting begins here   ###########################################
################################################################################################


############# Define the diagnostic limit for HII region identification ########################
mag_range = np.arange(-4.75, 0, 0.001)
limit = 10**((-0.37 * mag_range) - 1.16)

################################################################################################
################################  First plot   #################################################
################################################################################################

# Options and switches
f_size = 20  # unify the font size
p_s = 40     # unify the point size 
ann = args.n # True or False flag for annotating the plots with numbers.

fig = plt.figure(figsize=(8,10))
ax0 = plt.subplot(2,1,1)
# Scatter plot of m_5007 (x axis) vs F_[OIII] / (Ha + 1.34*NII), where Ha or NII AON is greater than 3, and ID == PN
# colorof points is set to Ha signal to noise (AON), between 3 and max value of AON_Ha
plt.scatter(gand_df["m 5007"].loc[HaNII_or_cond].values, HaNII_ratio_cond_or, c=gand_df["Ha AON"].loc[HaNII_or_cond], vmin=3, vmax=np.max(Ha_AON[HaNII_or_cond]), s=p_s)


#### NEED TO ADD IN CONDITION of Ha or NII >3
for i in np.where(HaNII_or_cond)[0]:
    if NII_AON[i] < 3:
        # for i where Ha or NII AON is greater than 3, and if AON_NII less than 3 (assumes that at least Ha is detected at AON greater than 3), 
        # then plot errorbars to indidcate corrected value range.
        # plotline1, caplines1 and barlinecols1 is used so as to customise the tips of the errorbars
        plotline1, caplines1, barlinecols1 = plt.errorbar(x=gand_df["m 5007"].iloc[i], y=ratio[i], yerr=np.abs(ratio[i]-ratio_corr[i]), uplims=True, c="k", alpha=0.7, elinewidth=0.8, ls="None", capsize=0)
        caplines1[0].set_marker("_")
        caplines1[0].set_markersize(0)

# Annotate the points on the figure
for n, i in enumerate(gand_df["[OIII] flux"].loc[HaNII_or_cond].index.values):
    # If data point is below limit, annotate the number
    if (galaxy_name == "FCC219") & (i in [52,53,54,63]):
        plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]-0.15, HaNII_ratio_cond_or[n]-0.35), c="r", )
    elif (galaxy_name == "FCC167") & (i in [46,69]):
        plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]-0.15, HaNII_ratio_cond_or[n]-0.35), c="r", )
    if HaNII_ratio_cond_or[n] < 10**((-0.37 * (gand_df["m 5007"].iloc[i]-dM)) - 1.16):
        if ann == True:
            # annotate appropriate scatter point with the correct object number ID (1 -> N). 
            # Ths is to help identify points between the two scatter plots.
            # here we annotate with the number, at m_5007 (x axis) + 0.03, and  F_[OIII]/(Ha+NII) (y axis).
            if i == 35:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]+0.02, HaNII_ratio_cond_or[n]-0.12))
            elif i == 6:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]-0.1, HaNII_ratio_cond_or[n]))
            elif i == 32:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]+0.05, HaNII_ratio_cond_or[n]-0.1))
            else:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gand_df["m 5007"].iloc[i]+0.05, HaNII_ratio_cond_or[n]))


#Limits
plt.xlim(-5.3+dM,-2.+dM)
plt.ylim(0.25, 20)

# Set yscale to log and add colorbar + label
plt.yscale("log")

# colorbar settings
cb = plt.colorbar()
cb.set_label(r"$\rm A_{H\alpha}\ / rN$", fontsize=f_size)
ax = plt.gca()
cb.ax.tick_params(labelsize=f_size)

# SII encircled points
plt.scatter(gand_df["m 5007"].loc[HaNII_SII_or_cond].values, HaNII_ratio_SII, facecolor="None", edgecolor="k", lw=1.2, s=p_s+150, label=r"$A_{[SII]}/rN > 2.5$")
plt.legend(fontsize=15)

# Draw limit lines; horizontal line at y=4, then using condition limit for HII regions.
plt.axhline(4, c="k", ls="--", alpha=0.7)
plt.plot(mag_range+dM, limit, c="k", ls="--", alpha=0.7)

# Labels
plt.xlabel(r"$\rm m_{5007}$", fontsize=f_size)
plt.ylabel(r"$\rm [OIII] \ / \ ([H\alpha + [NII])$", fontsize=f_size)
plt.tick_params(labelsize = 18)
plt.yticks([0.1, 1, 10], [0.1, 1 ,10])




########################################################################################
################################  Second plot   ########################################
########################################################################################


ax1 = plt.subplot(2,1,2)

# Plot Ha / SII (x axis) against Ha / NII, scatter point colour scale is set to AON of NII, where Ha and NII AON is above 3. colorbar bar ranges from 3 to 6
plt.scatter(Ha_SII, Ha_NII, c=gand_df["[NII] AON"].loc[HaNII_and_cond], s=p_s, vmin=3, vmax=6, edgecolors="k", lw=0.8)

# Plot the colorbar, setting the ticks, labels and label size
cb = plt.colorbar(ticks=[3,4,5,6])
cb.set_label(r"$\rm A_{[NII]}\ / rN$", fontsize=f_size)
ax = plt.gca()
cb.ax.tick_params(labelsize=f_size)

# Circle scatter points that have SII above 2.5, along with AON of Ha and NII both above 3
plt.scatter(Ha_SII_1, Ha_NII_1, s=p_s+150, facecolors="None", edgecolors="k", lw=1.2)

# For objects where Ha and NII are detected above AON of 3
# Where either of the SII lines is below AON of 3, plot a onesided error bar to indicate potential range of values.
for i in zip(*np.where(HaNII_and_cond)):
    if (SII_I_AON[i] <3) | (SII_II_AON[i]<3):
        plotline1, caplines1, barlinecols1 = plt.errorbar(x=HaoSII[i], y=HaoNII[i], xerr=np.abs(HaoSII_corr[i]-HaoSII[i]), c="k", xuplims=True ,alpha=0.7, elinewidth=0.8, ls="None", capsize=0)
    caplines1[0].set_marker("|")
    caplines1[0].set_markersize(0)

# if ann arg is True, annotate the plot with the ID number
for n, i in enumerate(gand_df["Ha flux"].loc[HaNII_and_cond].index.values):
    if ann == True:
        if galaxy_name == "FCC167":
            if i == 27:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.1, Ha_NII[n]-0.15))
            elif i == 13:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]+0.06, Ha_NII[n]+0.14))
            elif i == 67:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.1, Ha_NII[n]-0.15))
            elif i == 21:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.16, Ha_NII[n]+0.12))
            else:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.1, Ha_NII[n]+0.12))
        if galaxy_name == "FCC219":
            if i == 54:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.4, Ha_NII[n]-0.32))
            elif i == 34:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.1, Ha_NII[n]+0.2))
            else:
                plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]-0.1, Ha_NII[n]+0.12))
            
        
# Set scale to Log for x and y, then set limtis
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.1,10)
plt.xlim(0.1,10**1.5)
# plt.colorbar()

# Draw out the HII region
plt.plot(np.power(10, [0.5,0.9,0.9,0.5,0.5]), np.power(10, [0.2,0.2,0.7,0.7,0.2]), c="k", alpha=0.8, lw=0.8)
# Draw out the SNR region
plt.plot(np.power(10, [-0.1,0.3,0.3,0.1,0.1,-0.1,-0.1]),np.power(10,[0.05,0.25,0.05,-0.05,-0.5,-0.5,0.05]), c="k", alpha=0.8, lw=0.8)

# Draw line for condition of x axis of 1.3, objects to the left of this line, and reside close to the SNR region are considered SNR
plt.axvline(1.3, c="k", alpha=0.5, ls="--", lw=0.8)

# Annotate the regions
plt.annotate("HII", (4.2,6), fontsize=f_size)
plt.annotate("SNR", (0.78, 0.21), fontsize=f_size)

# Change the size and intervals of the axis ticks
plt.tick_params(labelsize = 18)
plt.xticks([0.1, 1, 10], [0.1, 1, 10])
plt.yticks([0.1, 1, 10], [0.1, 1 ,10])

# Label x and y axis
plt.xlabel(r"$\rm H\alpha \ / \ [SII]$", fontsize=f_size)
plt.ylabel(r"$\rm H\alpha \ / \ [NII]$", fontsize=f_size)


if plt_save == True:
    plt.savefig(f"Plots/{galaxy_name}_contamination_test.pdf", bbox_inches='tight')


###########################################################################################################################
#########################################   End of plotting   #############################################################
###########################################################################################################################
    
    
# Print statements for the index of imposters:
# initial check for imposters using HII check, on objects with Ha alpha AON of 3
m = gand_df["m 5007"].loc[HaNII_and_cond].values - dM
imposters = gand_df.loc[HaNII_and_cond].iloc[HaNII_ratio_cond_and < (10**((-0.37 * m) - 1.16))].index.values
print("First imposter check, PNe: ", imposters)

HII_region_x = [10**0.5, 10**0.9]
HII_region_y = [10**0.2, 10**0.7]

# HII_region_x[0]<Ha_SII.all()<HII_region_y[1]

# SNR first check Ha/SII < 1.3
SNR = gand_df.loc[HaNII_and_cond].iloc[np.where((Ha_SII<1.3) & (Ha_NII<10**0.25) & (Ha_NII>10**-0.5))].index.values
HII_imposter = [i for i in imposters if i not in SNR]

print(f"SNR imposters {SNR}")
print(f"HII imposters {HII_imposter}")


# Plot out brightest PNe with filter Y

p = int(PNe_df.loc[PNe_df["ID"]=="PN"].nsmallest(1, "m 5007").index.values)




def emission_plot_maker(obj_n, obj_t, top_plt_y_range, sub_OIII=2e4, sub_Ha=2e4, shift=0, save_fig=False):
    ind_OIII = [np.argmin(abs(wave-4850)), np.argmin(abs(wave-5100))]
    ind_Ha   = [np.argmin(abs(wave-6500)), np.argmin(abs(wave-6750))]
    
    fig = plt.figure(figsize=(16,10),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    f_ax1 = fig.add_subplot(gs[0, :]) # data, stellar and best-fit plots
    f_ax1.set_title(f"{obj_t}, number {obj_n}", fontsize=f_size)
    f_ax1.plot(wave, AllSpectra[1].data[obj_n][0], c="k", lw=1, alpha=0.8, label="data")
    f_ax1.plot(wave, gandalf_best[1].data[obj_n][0], c="g", lw=1.1, label="best fit", )
    f_ax1.plot(wave, gandalf_best[1].data[obj_n][0] - gandalf_emission[1].data[obj_n][0], c="r", lw=0.7,label="stellar")
    f_ax1.set_xlim(min(wave)-20, max(wave)+20)
    f_ax1.set_ylim(top_plt_y_range[0], top_plt_y_range[1])
    f_ax1.set_xlabel(r"Wavelength $(\AA)$", fontsize=f_size)
    f_ax1.set_ylabel("Flux Density", fontsize=f_size)
    f_ax1.tick_params(axis="both", labelsize=f_size )
    f_ax1.legend(loc="lower right", fontsize=12)
    
    f_ax2 = fig.add_subplot(gs[1, :]) # emissiona nd residual plot
    f_ax2.plot(wave, gandalf_emission[1].data[obj_n][0], lw=1.5, c="blue", label="emission") # emission
    f_ax2.scatter(wave, AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0], c="k", s=1, label="residuals")  # residuals
    f_ax2.axhline(np.std(AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0]), c="k", ls="--", alpha=0.7, label="residual noise") # residual noise line
    f_ax2.set_ylim(-1250, np.max(gandalf_emission[1].data[obj_n][0])*1.25)
    f_ax2.set_xlim(min(wave)-20, max(wave)+20)
    f_ax2.set_xlabel(r"Wavelength $(\AA)$", fontsize=f_size)
    f_ax2.set_ylabel("Flux Density", fontsize=f_size)
    f_ax2.tick_params(axis="both", labelsize=f_size)
    f_ax2.legend(loc="upper right",fontsize=12)
    
    
    f_ax3 = fig.add_subplot(gs[-1, :-1]) # [OIII] region plot
    f_ax3.plot(wave, (AllSpectra[1].data[obj_n][0]) - sub_OIII, c="k", lw=1, alpha=0.8, label="data")
    f_ax3.plot(wave, (gandalf_best[1].data[obj_n][0]) - sub_OIII, c="g", label="best fit", zorder=1)
    f_ax3.plot(wave, (gandalf_best[1].data[obj_n][0] - gandalf_emission[1].data[obj_n][0]) - sub_OIII, c="r", label="stellar", lw=0.7, zorder=2)
    f_ax3.plot(wave, gandalf_emission[1].data[obj_n][0], c="blue", lw=1.5)
    f_ax3.scatter(wave, AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0], c="k", s=2)
    f_ax3.axhline(np.std(AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0]), c="k", ls="--", alpha=0.7)
    f_ax3.set_xlim(4850,5100)
    f_ax3.set_ylim(-1000,np.max(AllSpectra[1].data[obj_n][0][ind_OIII[0]:ind_OIII[1]])-sub_OIII+1e3)
    f_ax3.set_xlabel(r"Wavelength $(\AA)$", fontsize=f_size)
    f_ax3.set_ylabel("Flux Density", fontsize=f_size)
    f_ax3.tick_params(axis="both", labelsize=f_size )
    f_ax3.annotate("[OIII]", (wave[ind_OIII[0]+np.argmax(gandalf_emission[1].data[obj_n][0][ind_OIII[0]:ind_OIII[1]])]+2, np.max(gandalf_emission[1].data[obj_n][0][ind_OIII[0]:ind_OIII[1]])-100), fontsize=12)
    
    
    f_ax4 = fig.add_subplot(gs[-1, -1]) # [NII], Ha, and [SII] region plot
    f_ax4.plot(wave, (AllSpectra[1].data[obj_n][0]) - sub_Ha, c="k", lw=1, alpha=0.8, label="data")
    f_ax4.plot(wave, (gandalf_best[1].data[obj_n][0]) - sub_Ha, c="g", label="best fit", zorder=1)
    f_ax4.plot(wave, (gandalf_best[1].data[obj_n][0] - gandalf_emission[1].data[obj_n][0]) - sub_Ha, c="r", label="stellar", lw=0.7, zorder=2)
    f_ax4.plot(wave, gandalf_emission[1].data[obj_n][0], c="blue", lw=1.5)
    f_ax4.scatter(wave, AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0], c="k", s=2)
    f_ax4.axhline(np.std(AllSpectra[1].data[obj_n][0] - gandalf_best[1].data[obj_n][0]), c="k", ls="--", alpha=0.7)
    f_ax4.set_xlim(6500, 6750)
    f_ax4.set_ylim(-1000, np.max(AllSpectra[1].data[obj_n][0][ind_Ha[0]:ind_Ha[1]])-sub_Ha+1e3)
    f_ax4.set_xlabel(r"Wavelength $(\AA)$", fontsize=f_size)
    f_ax4.set_ylabel("Flux Density", fontsize=f_size)
    f_ax4.tick_params(axis="both", labelsize=f_size)
    
    f_ax4.annotate("[NII]", (6590,1000+shift), fontsize=12)
    f_ax4.annotate(r"$H\alpha$", (6567, 1000+shift), fontsize=12)
    f_ax4.annotate("[SII]", (6715, 1000+shift), fontsize=12)
    
    if plt_save == True:
        plt.savefig(f"Plots/{galaxy_name}_{obj_t}_gandalf_spec.pdf", bbox_inches='tight')


        
# PNe Plot
# FCC167
emission_plot_maker(p, obj_t="PNe", top_plt_y_range=[4e4,8e4], sub_OIII=5e4, sub_Ha=5.8e4, )
# FCC219
# emission_plot_maker(p, obj_t="PNe", top_plt_y_range=[4e4,8e4], sub_OIII=5e4, sub_Ha=5.8e4, )
# SNR plot
# Plot out brightest SNR object with filter Y
if len(SII_II_AON[SNR]) != 0:
    n = SNR[np.argmax(SII_II_AON[SNR])]
    # FCC167
    emission_plot_maker(n, obj_t="SNR", top_plt_y_range=[2e4,5.5e4], sub_OIII=3e4, sub_Ha=3.5e4, )
    # FCC219
#     emission_plot_maker(n, obj_t="SNR", top_plt_y_range=[2e4,5.5e4], sub_OIII=3e4, sub_Ha=3.5e4, )

plt.show()    

if show_plot == False:
    plt.close("all")
