from functions.PNe_functions import dM_to_D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from scipy.stats import norm

####### Read in
galaxy_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
pd.set_option('display.max_columns', 100)
idx = pd.IndexSlice

gal_names = np.unique(galaxy_df.index.get_level_values(0))
galaxy_loc = [np.unique(galaxy_df.index.get_level_values(0)),["center"]*len(np.unique(galaxy_df.index.get_level_values(0)))]
galaxy_halo = [galaxy_df.loc[idx[:, 'halo'], :].index.get_level_values(0), ["halo"]*len(galaxy_df.loc[idx[:, 'halo'], :].index.get_level_values(0))]
galaxy_mid = [galaxy_df.loc[idx[:, 'middle'], :].index.get_level_values(0), ["middle"]*len(galaxy_df.loc[idx[:, 'middle'], :].index.get_level_values(0))]

gal_cen_tuples = list(zip(*galaxy_loc))
gal_halo_tuples = list(zip(*galaxy_halo))
gal_middle_tuples = list(zip(*galaxy_mid))

# PN_N_filter  = (galaxy_df.loc[idx[:, 'center'], "PNe N"]>20)
PN_N_filter  = (galaxy_df.loc[idx[:, 'center'], "PNe N"]>=5) #&  (galaxy_df.loc[gal_cen_tuples, "Bl dM"].notna()) 


with open("config/galaxy_info.yaml", "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
galaxy_info = [yaml_info[f"{gal_name}_center"] for gal_name in gal_names]

#########

PNLF_dM = galaxy_df[["PNLF dM"]].loc[gal_cen_tuples].values
PNLF_dM_err = galaxy_df[["PNLF dM err up", "PNLF dM err lo"]].loc[gal_cen_tuples].values
Bl_dM = galaxy_df[["Bl dM"]].loc[gal_cen_tuples].values
Bl_dM_err = galaxy_df[["Bl dM err"]].loc[gal_cen_tuples].values

PNLF_Bl_mtch_indx = np.where((np.isnan(PNLF_dM)!=True) & (np.isnan(Bl_dM)!=True))[0]

def weighted_avg(dM, dM_sigma, match_indx):
    dM_weighted_avg = np.average(dM[match_indx], weights=1/dM_sigma[match_indx].mean(axis=1)**2, axis=0)
    dM_weighted_avg_err = np.sqrt(1/(np.nansum(1/dM_sigma[match_indx].mean(axis=1)**2)))

    return dM_weighted_avg, dM_weighted_avg_err

PNLF_weighted_avg, PNLF_weighted_avg_err = weighted_avg(PNLF_dM, PNLF_dM_err, PNLF_Bl_mtch_indx)
SBF_weighted_avg, SBF_weighted_avg_err = weighted_avg(Bl_dM, Bl_dM_err, PNLF_Bl_mtch_indx)

########



filter_to_use = PN_N_filter
PNLF_D = np.empty(len(gal_names))
PNLF_D_err_up = np.empty(len(gal_names))
PNLF_D_err_lo = np.empty(len(gal_names))
for i , gal_l in enumerate(list(gal_cen_tuples)):
    if filter_to_use[i] == True:
        PNLF_D[i]     = 10.**(((galaxy_df.loc[gal_l, "PNLF dM"]) -25.) / 5.)
        # PNLF_D_err[i] = galaxy_df.loc[gal_l, "PNLF dM err"] / (5/(np.log(10) * PNLF_D[i]))
        PNLF_D_err_up[i] = 0.2*np.log(10)*galaxy_df.loc[gal_l, "PNLF dM err up"]*PNLF_D[i]
        PNLF_D_err_lo[i] = 0.2*np.log(10)*galaxy_df.loc[gal_l, "PNLF dM err lo"]*PNLF_D[i]
    elif filter_to_use[i] == False:
        PNLF_D[i]     = np.nan
        PNLF_D_err_up[i] = np.nan
        PNLF_D_err_lo[i] = np.nan

PNLF_D_lt = np.empty(len(gal_names))
PNLF_D_err_lt_up = np.empty(len(gal_names))
PNLF_D_err_lt_lo = np.empty(len(gal_names))
for i, gal_l in enumerate(list(gal_cen_tuples)):
    if filter_to_use[i] == False:
        PNLF_D_lt[i]     = 10.**(((galaxy_df.loc[(gal_l), "PNLF dM"]) -25.) / 5.)
        # PNLF_D_err_lt[i] = galaxy_df.loc[gal_l, "PNLF dM err"] / (5/(np.log(10) * PNLF_D_lt[i]))
        PNLF_D_err_lt_up[i] = 0.2*np.log(10)*galaxy_df.loc[gal_l, "PNLF dM err up"]*PNLF_D_lt[i]
        PNLF_D_err_lt_lo[i] = 0.2*np.log(10)*galaxy_df.loc[gal_l, "PNLF dM err lo"]*PNLF_D_lt[i]
    elif filter_to_use[i] == True:
        PNLF_D_lt[i]  = np.nan
        PNLF_D_err_lt_up[i] = np.nan
        PNLF_D_err_lt_lo[i] = np.nan


##### PNLF vs Bl SBF ###########
plt.figure(figsize=(22,8))

plt.scatter(gal_names, PNLF_D, label=r"PNLF CDF fit", c="g", s=100, zorder=0) #, $\geq$ 20 PNe
Bl_D = 10.**(((galaxy_df.loc[gal_cen_tuples, "Bl dM"]) -25.) / 5.)
plt.scatter(gal_names, Bl_D, label="Blakeslee (SBF)", c="k", alpha=0.7, s=50 ,zorder=1)


plt.errorbar(gal_names, PNLF_D, yerr=[PNLF_D_err_lo, PNLF_D_err_up], ls="None", c="g" ,lw=1.1, capsize=4)
Bl_D_err = galaxy_df.loc[gal_cen_tuples, "Bl dM err"] / (5/(np.log(10) * Bl_D))
plt.errorbar(gal_names, Bl_D, yerr=Bl_D_err, ls="None", c="k", alpha=0.7, capsize=4)


plt.axhline(dM_to_D(PNLF_weighted_avg), c="g", alpha=1, label="Weighted average, PNLF") # PNLF_weighted_avg
plt.axhline(dM_to_D(PNLF_weighted_avg+PNLF_weighted_avg_err), c="g", ls="--", alpha=1,) # PNLF_weighted_avg
plt.axhline(dM_to_D(PNLF_weighted_avg-PNLF_weighted_avg_err), c="g", ls="--", alpha=1,) # PNLF_weighted_avg

plt.axhline(20.0, c="k", alpha=1, label="Cluster average, Blakeslee 2009, SBF")
plt.axhline(20.0+1.4, c="k", ls="--", alpha=1,) # SBF_weighted_avg
plt.axhline(20.0-1.4, c="k", ls="--", alpha=1,) # SBF_weighted_avg


plt.yticks(np.arange(13, 28, 1.0), fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("Distance (Mpc)", fontsize=15)
plt.xlabel("Fornax Galaxy Name", fontsize=15)
plt.ylim(13.,28.)
plt.xlim(-1,21)
plt.legend(loc="upper center", fontsize=14,)# bbox_to_anchor=(1., 1.))

plt.grid(True, ls = '-.', lw = 0.5)
plt.savefig("Plots/F3D_cluster_work/Distance_comparison_Bl.png", bbox_inches='tight', dpi=300)

PNLF_dM = galaxy_df[["PNLF dM"]].loc[gal_cen_tuples].values
PNLF_dM_err = galaxy_df[["PNLF dM err up", "PNLF dM err lo"]].loc[gal_cen_tuples].values

PNLF_Bl_mtch_indx = np.where((np.isnan(PNLF_dM)!=True) )[0] #& (np.isnan(Bl_dM)!=True))[0]

def weighted_avg(dM, dM_sigma, match_indx):
    dM_weighted_avg = np.average(dM[match_indx], weights=1/dM_sigma[match_indx].mean(axis=1)**2, axis=0)
    dM_weighted_avg_err = np.sqrt(1/(np.nansum(1/dM_sigma[match_indx].mean(axis=1)**2)))

    return dM_weighted_avg, dM_weighted_avg_err

PNLF_weighted_avg, PNLF_weighted_avg_err = weighted_avg(PNLF_dM, PNLF_dM_err, PNLF_Bl_mtch_indx)

print(f"Weighted average PNLF distance modulus = {PNLF_weighted_avg[0]:.4f} +/- {PNLF_weighted_avg_err:.4f}")

####### PNLF vs CF2 ##########

plt.figure(figsize=(22,8))

plt.scatter(gal_names, PNLF_D, label=r"PNLF CDF fit", c="g", s=100, zorder=0)
# plt.scatter(gal_names, PNLF_D_lt, label="PNLF CDF fit, < 20 PNe", c="r", s=100, zorder=0)
CF_D = 10.**(((galaxy_df.loc[gal_cen_tuples, "lit dM"]) -25.) / 5.)
plt.scatter(gal_names, CF_D, label="CosmicFlows-3", c="k", alpha=0.7, s=50 ,zorder=1)


plt.errorbar(gal_names, PNLF_D, yerr=[PNLF_D_err_lo, PNLF_D_err_up], ls="None", c="g", lw=1.1, capsize=4)
# plt.errorbar(gal_names, PNLF_D_lt, yerr=[PNLF_D_err_lt_lo, PNLF_D_err_lt_up], ls="None", c="r", lw=1.1, capsize=4)

CF_D_err = galaxy_df.loc[gal_cen_tuples, "lit dM err"] / (5/(np.log(10) * CF_D))
plt.errorbar(gal_names, CF_D, yerr=CF_D_err, ls="None", c="k", alpha=0.7, capsize=4)


plt.axhline(dM_to_D(PNLF_weighted_avg), c="g", alpha=1, label="Weighted average, PNLF") # PNLF_weighted_avg
plt.axhline(dM_to_D(PNLF_weighted_avg+PNLF_weighted_avg_err), c="g", ls="--", alpha=1,) # PNLF_weighted_avg
plt.axhline(dM_to_D(PNLF_weighted_avg-PNLF_weighted_avg_err), c="g", ls="--", alpha=1,) # PNLF_weighted_avg

# plt.axhline(18.7, c="k", alpha=0.7, label="Reported Mean CF-2")
plt.axhline(19.3682, c="k", alpha=1, label="Weighted average CF-3") # CF3_weighted_avg
plt.axhline(18.9264, c="k", ls="--", alpha=1,) # CF3_weighted_avg
plt.axhline(19.8203, c="k", ls="--", alpha=1,) # CF3_weighted_avg

plt.legend(bbox_to_anchor=(1., 1.), fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.ylabel("Distance (Mpc)", fontsize=15)
plt.xlabel("Fornax Galaxy Name", fontsize=15)

plt.legend(loc="upper left", fontsize=13, bbox_to_anchor=(.41, 1.01))
plt.yticks(np.arange(13, 28, 1.0), fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("Distance (Mpc)", fontsize=15)
plt.xlabel("Fornax Galaxy Name", fontsize=15)
plt.ylim(13.,28.)
plt.xlim(-1,21)
plt.grid(True, ls = '-.', lw = 0.5)
plt.savefig("Plots/F3D_cluster_work/Distance_comparison_CF3.png", bbox_inches='tight', dpi=300)


#### PNLF vs SBF histogram, matching ciardullo paper 20212 ########

PN_N_filter  = (galaxy_df.loc[idx[:, 'center'], "PNe N"]>=5) &  (galaxy_df.loc[gal_cen_tuples, "Bl dM"].notna()) 

PNLF_dM_values = galaxy_df.loc[gal_cen_tuples, "PNLF dM"].loc[PN_N_filter].values
PNLF_dM_err = galaxy_df.loc[gal_cen_tuples, ["PNLF dM err up", "PNLF dM err up"]].loc[PN_N_filter].median(axis=1).values
Bl_SBF_dM_values = galaxy_df.loc[gal_cen_tuples, "Bl dM"].loc[PN_N_filter].values
Bl_SBF_dM_err = galaxy_df.loc[gal_cen_tuples, "Bl dM err"].loc[PN_N_filter].values

PNLF_SBF_tension = (PNLF_dM_values - Bl_SBF_dM_values) / np.sqrt(PNLF_dM_err**2 + Bl_SBF_dM_err**2)

plt.figure(figsize=(8,6))
plt.hist(galaxy_df.loc[gal_cen_tuples, "PNLF dM"].loc[PN_N_filter].values - galaxy_df.loc[gal_cen_tuples, "Bl dM"].loc[PN_N_filter].values , bins=np.arange(-0.6, 0.48 , 0.12), ec="black", lw=0.8)
plt.xlabel(r"$\Delta \mu$(PNLF - SBF)", fontsize=18)
plt.ylabel("Number of Galaxies", fontsize=18)
plt.xticks(np.arange(-1.5,1., 0.25),fontsize=14)
plt.yticks(fontsize=16)
plt.xlim(-1.4,0.8)
plt.ylim(0,6)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
mu, std = norm.fit(galaxy_df.loc[gal_cen_tuples, "PNLF dM"].loc[PN_N_filter].values - galaxy_df.loc[gal_cen_tuples, "Bl dM"].loc[PN_N_filter].values,)
median_diff = np.nanmedian(galaxy_df.loc[gal_cen_tuples, "PNLF dM"].loc[PN_N_filter].values) - np.median(galaxy_df.loc[gal_cen_tuples, "Bl dM"].loc[PN_N_filter].values)
p = norm.pdf(x, median_diff, std)
plt.plot(x, 1.96*p, c="k", ls="--", lw=1.2)

print(f"Median PNLF - SBF: {np.nanmedian(galaxy_df.loc[gal_cen_tuples, 'PNLF dM'].loc[PN_N_filter].values - galaxy_df.loc[gal_cen_tuples, 'Bl dM'].loc[PN_N_filter].values)}")
print(f"std PNLF - SBF: {np.nanstd(galaxy_df.loc[gal_cen_tuples, 'PNLF dM'].loc[PN_N_filter].values - galaxy_df.loc[gal_cen_tuples, 'Bl dM'].loc[PN_N_filter].values)}")

plt.figure(figsize=(8,6))
plt.hist(PNLF_SBF_tension, bins=np.arange(-1.75,2,0.25),ec="black", lw=0.8)
# plt.axvline(np.median(PNLF_SBF_tension))
plt.xlabel(r"$(\mu_{PNLF} - \mu_{SBF}) \, / \, \sqrt{\sigma_{PNLF}^{2} + \sigma_{SBF}^{2}}$", fontsize=18) #$\Delta \mu$(PNLF - SBF) /
plt.ylabel("Number of Galaxies", fontsize=18)
plt.yticks(fontsize=16)
plt.ylim(0,6)
# plt.title(f"Median tension between PNLF & SBF: {np.median(PNLF_SBF_tension):.3f}", fontsize=14)
print(f"Median tension PNLF - SBF: {np.nanmedian(PNLF_SBF_tension)}")


plt.savefig("Plots/F3D_cluster_work/tension_PNLF_minus_SBF_mu.png", bbox_inches='tight', dpi=300)