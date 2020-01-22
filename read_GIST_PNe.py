import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from matplotlib.patches import Rectangle, Ellipse, Circle
import matplotlib.gridspec as gridspec

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
    work_dir = f"../../gist_PNe/results/{galaxy_name}MUSEPNeweighted_contamination/{galaxy_name}MUSEPNeweighted"
else:
    work_dir = f"../../gist_PNe/results/{galaxy_name}MUSEPNe_contamination/{galaxy_name}MUSEPNe"
    
# read in PNe data
PNe_df = pd.read_csv(f"exported_data/{galaxy_name}/{galaxy_name}_PNe_df.csv")
m_5007 = PNe_df["m 5007"]

# Open and name the following result files
gandalf_emission = fits.open(f"{work_dir}_gandalf-emission_SPAXEL.fits")
gandalf_best     = fits.open(f"{work_dir}_gandalf-bestfit_SPAXEL.fits")
gandalf_clean    = fits.open(f"{work_dir}_gandalf-cleaned_SPAXEL.fits")
gandalf_results  = fits.open(f"{work_dir}_gandalf_SPAXEL.fits")
AllSpectra       = fits.open(f"{work_dir}_AllSpectra.fits")
ppxf_results     = fits.open(f"{work_dir}_ppxf.fits")
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


Hb_ampl     = np.array([gandalf_results[2].data[i]["Ampl"][11] for i in np.arange(len(gandalf_results[2].data))])
OIII_ampl   = np.array([gandalf_results[2].data[i]["Ampl"][12] for i in np.arange(len(gandalf_results[2].data))])
Ha_ampl     = np.array([gandalf_results[2].data[i]["Ampl"][17] for i in np.arange(len(gandalf_results[2].data))])
NII_ampl    = np.array([gandalf_results[2].data[i]["Ampl"][18] for i in np.arange(len(gandalf_results[2].data))])
SII_I_ampl  = np.array([gandalf_results[2].data[i]["Ampl"][19] for i in np.arange(len(gandalf_results[2].data))])
SII_II_ampl = np.array([gandalf_results[2].data[i]["Ampl"][20] for i in np.arange(len(gandalf_results[2].data))])

# Hb_flux     = np.array([gandalf_results[2].data[i]["Flux"][11] for i in np.arange(len(gandalf_results[2].data))])
# OIII_flux   = np.array([gandalf_results[2].data[i]["Flux"][12] for i in np.arange(len(gandalf_results[2].data))])
# Ha_flux     = np.array([gandalf_results[2].data[i]["Flux"][17] for i in np.arange(len(gandalf_results[2].data))])
# NII_flux    = np.array([gandalf_results[2].data[i]["Flux"][18] for i in np.arange(len(gandalf_results[2].data))])
# SII_I_flux  = np.array([gandalf_results[2].data[i]["Flux"][19] for i in np.arange(len(gandalf_results[2].data))])
# SII_II_flux = np.array([gandalf_results[2].data[i]["Flux"][20] for i in np.arange(len(gandalf_results[2].data))])
Hb_flux     = np.array([gandalf_results[2].data[i]["Ampl"][11]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
OIII_flux   = np.array([gandalf_results[2].data[i]["Ampl"][12]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
Ha_flux     = np.array([gandalf_results[2].data[i]["Ampl"][17]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
NII_flux    = np.array([gandalf_results[2].data[i]["Ampl"][18]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
SII_I_flux  = np.array([gandalf_results[2].data[i]["Ampl"][19]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20
SII_II_flux = np.array([gandalf_results[2].data[i]["Ampl"][20]*np.sqrt(2*np.pi)*(((gandalf_results[2].data[i]["Sigma"][12]*1e3)  / 299792458.0)*5006.77) for i in np.arange(len(m_5007))]) * 1e-20


Hb_AON     = np.array([gandalf_results[2].data[i]["AON"][11] for i in np.arange(len(gandalf_results[2].data))])
OIII_AON   = np.array([gandalf_results[2].data[i]["AON"][12] for i in np.arange(len(gandalf_results[2].data))])
Ha_AON     = np.array([gandalf_results[2].data[i]["AON"][17] for i in np.arange(len(gandalf_results[2].data))])
NII_AON    = np.array([gandalf_results[2].data[i]["AON"][18] for i in np.arange(len(gandalf_results[2].data))])
SII_I_AON  = np.array([gandalf_results[2].data[i]["AON"][19] for i in np.arange(len(gandalf_results[2].data))])
SII_II_AON = np.array([gandalf_results[2].data[i]["AON"][20] for i in np.arange(len(gandalf_results[2].data))])

gandalf_df = pd.DataFrame(index=np.arange(0,len(PNe_df)), columns=("Hb amp","Hb flux", "Hb AON",
             "[OIII] amp", "corr [OIII] amp","[OIII] flux","corr [OIII] flux", "[OIII] AON",
             "Ha amp", "Ha flux", "Ha AON", 
             "[NII] amp", "[NII] flux", "[NII] AON",
             "[SII]I amp", "[SII]I flux", "[SII]I AON",
             "[SII]II amp", "[SII]II flux", "[SII]II AON"))

gandalf_df["Hb amp"]          = Hb_ampl
gandalf_df["Hb flux"]         = Hb_flux
gandalf_df["Hb AON"]          = Hb_AON
gandalf_df["[OIII] amp"]      = OIII_ampl
gandalf_df["[OIII] flux"]     = OIII_flux
gandalf_df["[OIII] AON"]      = OIII_AON
gandalf_df["Ha amp"]          = Ha_ampl
gandalf_df["Ha flux"]         = Ha_flux
gandalf_df["Ha AON"]          = Ha_AON
gandalf_df["[NII] amp"]       = NII_ampl
gandalf_df["[NII] flux"]      = NII_flux
gandalf_df["[NII] AON"]       = NII_AON
gandalf_df["[SII]I amp"]      = SII_I_ampl
gandalf_df["[SII]I flux"]     = SII_I_flux
gandalf_df["[SII]I AON"]      = SII_I_AON
gandalf_df["[SII]II amp"]     = SII_II_ampl
gandalf_df["[SII]II flux"]    = SII_II_flux
gandalf_df["[SII]II AON"]     = SII_II_AON

# put m5007 values into gandalf_df
gandalf_df["m 5007"] = m_5007
gandalf_df["Filter"] = PNe_df["Filter"]
gandalf_df["Filter"]="Y"


# gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv")
# dM = gal_df["dM PNLF"].loc[gal_df["Galaxy"]==galaxy_name].values
dM = 31.229
# set up conditions

# First plot conditions
# Ha or NII over 3, with Filter==Y
HaNII_or_cond = ((gandalf_df["Ha AON"] > 3.) | (gandalf_df["[NII] AON"]>3)) & (gandalf_df["Filter"]=="Y")  
# HaNII_or_cond = (gandalf_df["Ha AON"] > 3.) & (gandalf_df["Filter"]=="Y")  
HaNII_and_cond = ((gandalf_df["Ha AON"] > 3.) & (gandalf_df["[NII] AON"]>3)) & (gandalf_df["Filter"]=="Y")  
# HaNII_and_cond = (gandalf_df["Ha AON"] > 3.) & (gandalf_df["Filter"]=="Y")  

# ( (Ha or NII over 3), and SII over 3) and Filter==Y
HaNII_SII_or_cond = (((gandalf_df["Ha AON"] > 3.) | (gandalf_df["[NII] AON"]>3)) & np.greater_equal((gandalf_df["[SII]I AON"]+gandalf_df["[SII]II AON"])/2, 2.5)) & (gandalf_df["Filter"]=="Y")
# HaNII_SII_or_cond = ((gandalf_df["Ha AON"] > 3.) & np.greater_equal((gandalf_df["[SII]I AON"]+gandalf_df["[SII]II AON"])/2, 2.5)) & (gandalf_df["Filter"]=="Y")

HaNII_SII_and_cond = (((gandalf_df["Ha AON"] > 3.) & (gandalf_df["[NII] AON"]>3)) & np.greater_equal((gandalf_df["[SII]I AON"]+gandalf_df["[SII]II AON"])/2, 2.5)) & (gandalf_df["Filter"]=="Y")
# HaNII_SII_and_cond = ((gandalf_df["Ha AON"] > 3.)  & np.greater_equal((gandalf_df["[SII]I AON"]+gandalf_df["[SII]II AON"])/2, 2.5)) & (gandalf_df["Filter"]=="Y")


######## Filter OIII / Ha+NII vs m_5007 data #############################
ratio_cond_or = gandalf_df["[OIII] flux"].loc[HaNII_or_cond].values / (gandalf_df["Ha flux"].loc[HaNII_or_cond].values + 1.34*gandalf_df["[NII] flux"].loc[HaNII_or_cond].values)
ratio_cond_and = gandalf_df["[OIII] flux"].loc[HaNII_and_cond].values / (gandalf_df["Ha flux"].loc[HaNII_and_cond].values + 1.34*gandalf_df["[NII] flux"].loc[HaNII_and_cond].values)
# ratio_cond = gandalf_df["[OIII] flux"].loc[HaNII_or_cond].values / (gandalf_df["Ha flux"].loc[HaNII_or_cond].values)

ratio_SII = gandalf_df["[OIII] flux"].loc[HaNII_SII_or_cond].values / (gandalf_df["Ha flux"].loc[HaNII_SII_or_cond].values + 1.34*gandalf_df["[NII] flux"].loc[HaNII_SII_or_cond].values)
# ratio_SII = gandalf_df["[OIII] flux"].loc[HaNII_SII_or_cond].values / (gandalf_df["Ha flux"].loc[HaNII_SII_or_cond].values)

# NII correction
ratio = OIII_flux / (Ha_flux+ 1.34*NII_flux)
# ratio = OIII_flux / (Ha_flux)

F_NII_corr = np.copy(NII_flux)
# F_NII_corr = np.copy(Ha_flux)

NII_AON_lt_3 = np.where(NII_AON<3)
# Ha_AON_lt_3 = np.where(Ha_AON<3)

F_NII_corr[NII_AON_lt_3] = 3*(NII_flux[NII_AON_lt_3] / NII_AON[NII_AON_lt_3])
# F_NII_corr[Ha_AON_lt_3] = 3*(Ha_flux[Ha_AON_lt_3] / Ha_AON[Ha_AON_lt_3])

ratio_corr = np.copy(OIII_flux) / (np.copy(Ha_flux)+ 1.34*F_NII_corr)
# ratio_corr = np.copy(OIII_flux) / (np.copy(Ha_flux))


# SII correction ###################
HaoNII = Ha_flux / NII_flux
HaoSII = Ha_flux / (SII_I_flux+SII_II_flux)

SII_I_lt_3 = np.where(SII_I_AON<2.5)
SII_II_lt_3 = np.where(SII_II_AON<2.5)

SII_I_F_corr = np.copy(SII_I_flux)
SII_II_F_corr = np.copy(SII_II_flux)

SII_I_F_corr[SII_I_lt_3] = 3*(SII_I_flux[SII_I_lt_3] / SII_I_AON[SII_I_lt_3])
SII_II_F_corr[SII_II_lt_3] = 3*(SII_II_flux[SII_II_lt_3] / SII_II_AON[SII_II_lt_3])

HaoSII_corr = Ha_flux / (SII_I_F_corr+SII_II_F_corr)



########## Filter Ha/NII vs Ha/SII data ################################
Ha_NII = gandalf_df["Ha flux"].loc[HaNII_and_cond].values / gandalf_df["[NII] flux"].loc[HaNII_and_cond].values
Ha_SII = gandalf_df["Ha flux"].loc[HaNII_and_cond].values / (gandalf_df["[SII]I flux"].loc[HaNII_and_cond].values + gandalf_df["[SII]II flux"].loc[HaNII_and_cond].values)

Ha_NII_1 = gandalf_df["Ha flux"].loc[HaNII_SII_and_cond].values / gandalf_df["[NII] flux"].loc[HaNII_SII_and_cond].values
Ha_SII_1 = gandalf_df["Ha flux"].loc[HaNII_SII_and_cond].values / (gandalf_df["[SII]I flux"].loc[HaNII_SII_and_cond].values + gandalf_df["[SII]II flux"].loc[HaNII_SII_and_cond].values)

# Options and switches
f_size = 20
p_s = 40
ann = args.n

fig = plt.figure(figsize=(8,10))
ax0 = plt.subplot(2,1,1)
plt.scatter(gandalf_df["m 5007"].loc[HaNII_or_cond].values, ratio_cond_or, c=gandalf_df["Ha AON"].loc[HaNII_or_cond], vmin=3, vmax=np.max(Ha_AON[HaNII_or_cond]), s=p_s)


#### NEED TO ADD IN CONDITION of Ha or NII >3
for i in np.squeeze(np.where(HaNII_or_cond)):
    if NII_AON[i] <3:
        plotline1, caplines1, barlinecols1 = plt.errorbar(x=gandalf_df["m 5007"].iloc[i], y=ratio[i], yerr=np.abs(ratio[i]-ratio_corr[i]), uplims=True, c="k", alpha=0.7, elinewidth=0.8, ls="None", capsize=0)
        caplines1[0].set_marker("_")
        caplines1[0].set_markersize(0)
        
for n, i in enumerate(gandalf_df["[OIII] flux"].loc[HaNII_or_cond].index.values):
    if ann == True:
        plt.annotate(str(PNe_df["PNe number"].iloc[i]), (gandalf_df["m 5007"].iloc[i]+0.03, ratio_cond_or[n]))
#Limits
plt.xlim(-5.3+dM,-2.+dM)
plt.ylim(0.25, 20)
# Set yscale to log and add colorbar + label
plt.yscale("log")
cb = plt.colorbar()
cb.set_label(r"$\rm A_{H\alpha}\ / rN$", fontsize=f_size)
ax = plt.gca()
cb.ax.tick_params(labelsize=f_size)
# SII encircled points
plt.scatter(gandalf_df["m 5007"].loc[HaNII_SII_or_cond].values, ratio_SII, facecolor="None", edgecolor="k",lw=1.2, s=p_s+150, label=r"$A_{[SII]}/rN > 2.5$")
plt.legend(fontsize=15)
# Draw regions
plt.axhline(4,c="k", ls="--", alpha=0.7)
x = np.arange(-4.75, 0,0.001)
plt.plot(x+dM, 10**((-0.37 * x) - 1.16), c="k", ls="--", alpha=0.7)
# Labels
plt.xlabel(r"$\rm m_{5007}$", fontsize=f_size)
plt.ylabel(r"$\rm [OIII] \ / \ ([H\alpha + [NII])$", fontsize=f_size)
plt.tick_params(labelsize = 18)
plt.yticks([0.1, 1, 10], [0.1, 1 ,10])


###### Start of second plot ##########
ax1 = plt.subplot(2,1,2)
# Plot here
plt.scatter(Ha_SII, Ha_NII, c=gandalf_df["[NII] AON"].loc[HaNII_and_cond], s=p_s, vmin=3, vmax=6)
cb = plt.colorbar(ticks=[3,4,5,6])
cb.set_label(r"$\rm A_{[NII]}\ / rN$", fontsize=f_size)
ax = plt.gca()
cb.ax.tick_params(labelsize=f_size)
plt.scatter(Ha_SII_1, Ha_NII_1, s=p_s+150, facecolors="None", edgecolors="k",lw=1.2)

for i in zip(*np.where(HaNII_and_cond)):
    if (SII_I_AON[i] <3) | (SII_II_AON[i]<3):
        plotline1, caplines1, barlinecols1 = plt.errorbar(x=HaoSII[i], y=HaoNII[i], xerr=np.abs(HaoSII_corr[i]-HaoSII[i]), c="k", xuplims=True ,alpha=0.7, elinewidth=0.8, ls="None", capsize=0)
    caplines1[0].set_marker("|")
    caplines1[0].set_markersize(0)

for n, i in enumerate(gandalf_df["Ha flux"].loc[HaNII_and_cond].index.values):
    if ann == True:
        plt.annotate(str(PNe_df["PNe number"].iloc[i]), (Ha_SII[n]+0.1, Ha_NII[n]+0.08))
# Set scale to Log for x and y, then set limtis
plt.xscale("log")
plt.yscale("log")
plt.ylim(0.1,10)
plt.xlim(0.1,10**1.5)
# plt.colorbar()

# Draw out the HII region and SNR regions
plt.plot(np.power(10, [0.5,0.9,0.9,0.5,0.5]), np.power(10, [0.2,0.2,0.7,0.7,0.2]), c="k", alpha=0.8, lw=0.8)
plt.plot(np.power(10, [-0.1,0.3,0.3,0.1,0.1,-0.1,-0.1]),np.power(10,[0.05,0.25,0.05,-0.05,-0.5,-0.5,0.05]), c="k", alpha=0.8, lw=0.8)

plt.axvline(1.3, c="k", alpha=0.5, ls="--", lw=0.8)
# Annotate the regions
plt.annotate("HII", (4.2,6), fontsize=f_size)
plt.annotate("SNR", (0.78, 0.21), fontsize=f_size)
plt.tick_params(labelsize = 18)
plt.xticks([0.1, 1, 10], [0.1, 1, 10])
plt.yticks([0.1, 1, 10], [0.1, 1 ,10])

plt.xlabel(r"$\rm H\alpha \ / \ [SII]$", fontsize=f_size)
plt.ylabel(r"$\rm H\alpha \ / \ [NII]$", fontsize=f_size)


if plt_save == True:
    plt.savefig(f"Plots/{galaxy_name}_contamination_test.pdf", bbox_inches='tight')
    
# Print statements for the index of imposters:
# initial check for imposters using HII check, on objects with Ha alpha AON of 3
m = gandalf_df["m 5007"].loc[HaNII_and_cond].values - dM
imposters = gandalf_df.loc[HaNII_and_cond].iloc[ratio_cond_and < (10**((-0.37 * m) - 1.16))].index.values
print("First imposter check, PNe: ", imposters)

HII_region_x = [10**0.5, 10**0.9]
HII_region_y = [10**0.2, 10**0.7]

# HII_region_x[0]<Ha_SII.all()<HII_region_y[1]

# SNR first check Ha/SII < 1.3
SNR = gandalf_df.loc[HaNII_and_cond].iloc[np.where((Ha_SII<1.3) & (Ha_NII<10**0.25) & (Ha_NII>10**-0.5))].index.values
HII_imposter = [i for i in imposters if i not in SNR]

print(f"SNR imposters {SNR}")
print(f"HII imposters {HII_imposter}")


# Plot out brightest PNe with filter Y

p = int(PNe_df.loc[PNe_df["Filter"]=="Y"].nsmallest(1, "m 5007").index.values)




def emission_plot_maker(obj_n, obj_t, sub_OIII=2e4, sub_Ha=2e4, shift=0, save_fig=False):
    ind_OIII = [np.argmin(abs(wave-4850)), np.argmin(abs(wave-5100))]
    ind_Ha   = [np.argmin(abs(wave-6500)), np.argmin(abs(wave-6750))]
    
    fig = plt.figure(figsize=(16,10),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    
    f_ax1 = fig.add_subplot(gs[0, :]) # data, stellar and best-fit plots
    f_ax1.plot(wave, AllSpectra[1].data[obj_n][0], c="k", lw=1, alpha=0.8, label="data")
    f_ax1.plot(wave, gandalf_best[1].data[obj_n][0], c="g", lw=1.1, label="best fit", )
    f_ax1.plot(wave, gandalf_best[1].data[obj_n][0] - gandalf_emission[1].data[obj_n][0], c="r", lw=0.7,label="stellar")
    f_ax1.set_xlim(min(wave)-20, max(wave)+20)
    f_ax1.set_ylim(0,)
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
    f_ax4.annotate("[SII]", (6715, 1500+shift), fontsize=12)
    
    if plt_save == True:
        plt.savefig(f"Plots/{galaxy_name}_{obj_t}_gandalf_spec.pdf", bbox_inches='tight')


        
# PNe Plot
emission_plot_maker(p, obj_t="PNe", sub_OIII=3.2e4, sub_Ha=4e4, )
# SNR plot
# Plot out brightest SNR object with filter Y
if len(SII_II_AON[SNR]) != 0:
    n = SNR[np.argmax(SII_II_AON[SNR])]
    emission_plot_maker(n, obj_t="SNR", sub_OIII=1.2e4, sub_Ha=1.5e4, )

plt.show()    

if show_plot == False:
    plt.close("all")