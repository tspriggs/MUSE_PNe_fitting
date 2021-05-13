import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.api import get_objs_combined_axis
import yaml

from astropy.io import ascii
from tqdm import tqdm

from scipy import stats


from matplotlib.ticker import (AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter

from functions.file_handling import paths
from functions.PNLF import MC_PNLF_runner, calc_PNLF,  PNLF_analysis, PNLF_fitter, ecdf

from scipy.optimize import curve_fit

np.random.seed(42)


my_parser = argparse.ArgumentParser()

my_parser.add_argument('--calc_err', action="store_true", default=False, help="Boolean switch for calculating errors every 50th iteration.")
args = my_parser.parse_args()
calc_err = args.calc_err

gal = "FCC083"
loc = "center"
DIR_dict = paths(gal, loc)

galaxy_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
galaxy_names = np.unique(galaxy_df.index.get_level_values(0))

PNe_df = pd.read_csv(f"exported_data/{gal}/{gal}{loc}_PNe_df.csv")
gal_m_5007 = PNe_df["m 5007"].loc[PNe_df["ID"] == "PN"]

with open("config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

galaxy_data = galaxy_info[f"{gal}_{loc}"]

obs_comp_list = [np.load(f"exported_data/{gal}/{gal}center_completeness_ratio.npy") for gal in galaxy_names]

# Change this to alter how many iterations of the simulation you want to run.
# n_sim = 10000

# Input values for distance modulus (dM_in) and the c2 paramter (in_c_2)
dM_in = 31.45
# dM_in = 33.01
in_c_2 = 0.307

mag_step = 0.001
M_star = -4.53
m_5007 = np.arange(22, 34, mag_step)
M_5007 = np.arange(M_star, M_star+12, mag_step)


gal_m_5007_err_up = PNe_df["mag error up"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_lo = PNe_df["mag error lo"].loc[PNe_df["ID"].isin(["PN"])].values
mag_err_list = np.median([gal_m_5007_err_up, gal_m_5007_err_lo],0) # average error

slope, intercept, r_value, p_value, std_err = stats.linregress(gal_m_5007, mag_err_list)


# change this to alter the number of PNe that can be within a simulation sample.
# 5 is the lower limit, if it is set lower, the simulation may fail.
# 161 was chosen to eoncompass the largest observed sample of PNe from the catalogue from Foranx3D work.
# 1 is the step size.

n_sim_per_gal = 20
n_PNe = np.repeat(np.arange(5, 206, 1), n_sim_per_gal*len(obs_comp_list))

n_sim = len(n_PNe)


fit_dM = np.ones((n_sim,3))
fit_c2 = np.ones((n_sim,3))

fit_Mstar = np.ones((n_sim,2))
m_5007_extent = np.ones(n_sim)
brightest_m_5007 = np.ones(n_sim)
MC_boot_KS2_stat_d = np.ones(n_sim)
MC_boot_KS2_stat_p = np.ones(n_sim)
boot_dM_50 = np.ones(n_sim)
boot_dM_16 = np.ones(n_sim)
boot_dM_84 = np.ones(n_sim)
boot_c2_50 = np.ones(n_sim)
boot_c2_16 = np.ones(n_sim)
boot_c2_84 = np.ones(n_sim)
chi_2 = np.ones(n_sim)
chi_r = np.ones(n_sim)
success = np.empty(n_sim)
n_sim_PNe = np.empty(n_sim)
for_marc_10_PNe = []
for_marc_13_PNe = []
complete_limit_mag = np.ones(n_sim)
calc_err_comp_lim = []
calc_err_counter = []

all_slop_intercept = ascii.read("exported_data/for_marc/all_galaxy_slope_intercept.txt")
slope_list = np.array(all_slop_intercept["slope"])
intercept_list = np.array(all_slop_intercept["intercept"])



PNLF = calc_PNLF(M_star+dM_in, M_5007+dM_in, c_2=in_c_2)

obs_indx_to_use = np.tile(np.arange(0,len(obs_comp_list)), int(n_sim/len(obs_comp_list)),)

PNLF_comp_corr_list = [np.array(np.interp(m_5007, M_5007+dM_in, PNLF)*obs_comp) for obs_comp in obs_comp_list]

for k, (n, obs) in tqdm(enumerate(zip(n_PNe, obs_indx_to_use)), total=len(n_PNe), leave=True):

    complete_limit_mag[k] = m_5007[obs_comp_list[obs]>=0.5].max()

    n_sim_pois = np.random.poisson(n)
    while n_sim_pois < 5:
        n_sim_pois = np.random.poisson(n)

    PNe_sample = np.interp(np.random.uniform(0,1,n_sim_pois), np.cumsum(PNLF_comp_corr_list[obs])/np.sum(PNLF_comp_corr_list[obs]), m_5007)
    n_sim_PNe[k] = n_sim_pois





    vary_dict={"dM":True, "M_star":False, "c1":False, "c2":True, "c3":False}
    if vary_dict["c2"] == False:
        PNLF_results = PNLF_analysis(gal, loc, PNe_sample,np.ones_like(PNe_sample), obs_comp_list[obs], M_5007, m_5007, c2_in=0.307, vary_dict=vary_dict, min_stat="KS_1samp", comp_lim=False)
    elif vary_dict["c2"] == True:
        PNLF_results = PNLF_analysis(gal, loc, PNe_sample,np.ones_like(PNe_sample), obs_comp_list[obs], M_5007, m_5007, c2_in=0.300, vary_dict=vary_dict, min_stat="KS_1samp", comp_lim=False)

    if PNLF_results.success == True:
        fit_dM[k,0] = PNLF_results.params["dM"]
        fit_c2[k,0] = PNLF_results.params["c2"]
        chi_2[k] = PNLF_results.chisqr
        chi_r[k] = PNLF_results.redchi
    elif PNLF_results.success == False:
        fit_dM[k,0] = np.nan
        fit_c2[k,0] = np.nan

        chi_2[k] = np.nan
        chi_r[k] = np.nan


    brightest_m_5007[k] = np.min(PNe_sample)


good_fits = ((fit_dM[:,0]>PNLF_results.params["dM"].min+0.1) & (fit_dM[:,0]<PNLF_results.params["dM"].max-0.1)) & \
           ((fit_c2[:,0]<PNLF_results.params["c2"].max-0.1) & (fit_c2[:,0]>PNLF_results.params["c2"].min+0.1)) #& (fit_c2[:,0]!=0.207)
   

print("SIMULATION COMPLETE")
print(f"{n_sim} simulations, across {np.min(n_PNe)} to {np.max(n_PNe)} PNe per sample.")

##### Plotting #####

bins = np.arange(0, n_PNe.max()+10,5)


sim_df = pd.DataFrame({"n_PNe":n_PNe[good_fits], "fit_dM":fit_dM[good_fits,0]-dM_in, "fit_c2":fit_c2[good_fits,0]-in_c_2})
data_cut = pd.cut(sim_df.n_PNe, bins)
grp = sim_df.groupby(by = data_cut).median()
grp_up = sim_df.groupby(by = data_cut).quantile(0.16)
grp_lo = sim_df.groupby(by = data_cut).quantile(0.84)

obsv_PN_N = 75

# For plotting, if the c2 parameter was set to vary, then the figure will have two subplots; the upper plot is delta dM, the lower plot is delta c2.
if vary_dict["c2"] == True:
    fig = plt.figure(figsize=(22,20))
    ax = fig.add_subplot(1,1,1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # delta dM plot is ax1
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(n_PNe[good_fits], fit_dM[good_fits,0]-dM_in, alpha=0.7, label="Data")
    ax1.axhline(0.0, ls="--", c="k", alpha=0.7)
    ax1.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $\mu_{PNLF}$="+f"{round(np.nanmedian(fit_dM[good_fits&(n_PNe==obsv_PN_N),0]),2)} $\pm$ {round(np.nanstd(fit_dM[good_fits&(n_PNe==obsv_PN_N),0]),2)}", fontsize=30, y=1.03)
    ax1.plot(grp.n_PNe, grp.fit_dM, c="r", label="Median",)
    plt.fill_between(grp.n_PNe,  grp_up.fit_dM, grp_lo.fit_dM, color="red", alpha=0.4, linestyle="None", label="1$\sigma$")
    ax1.set_ylabel("$\Delta \ \mu_{PNLF}$", fontsize=30)
    ax1.tick_params(axis='y', labelsize=25 )
    ax1.tick_params(axis='x', labelsize=25 )
    ax1.set_xlim(2,np.max(n_PNe)+5 )
    ax1.set_xticks(np.arange(5, np.max(n_PNe)+5,10))
    ax1.set_ylim(-1.5,1.5)
    ax1.set_yticks(np.arange(-1.5,2.0, 0.5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", length=6)
    ax1.tick_params(which="minor", length=4)
    ax1.legend(loc="upper right",fontsize=20)

    # delta c2 plot is ax2
    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(n_PNe[good_fits], fit_c2[good_fits,0]-in_c_2, alpha=0.7, )
    ax2.axhline(0.0, c="k", ls="--", alpha=0.7)
    ax2.plot(grp.n_PNe, grp.fit_c2, c="r")
    plt.fill_between(grp.n_PNe,  grp_up.fit_c2, grp_lo.fit_c2, color="red", alpha=0.4, linestyle="None")
    ax2.set_ylim(-1.5,1.5)
    ax2.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $c_{2}$=" + f"{round(np.nanmedian(fit_c2[good_fits&(n_PNe==obsv_PN_N),0]),3)} $\pm$ {round(np.nanstd(fit_c2[good_fits&(n_PNe==obsv_PN_N),0]),2)}", fontsize=30, y=1.03)
    ax2.set_ylabel("$\Delta \ c_{2}$", fontsize=30)
    ax2.tick_params(axis='y', labelsize=25 )
    ax2.tick_params(axis='x', labelsize=25 )
    ax2.set_ylim(-1.5,1.5)
    ax2.set_xlim(2,np.max(n_PNe)+5 )
    ax2.set_xticks(np.arange(5, np.max(n_PNe)+5,10))
    ax2.set_yticks(np.arange(-1.5,2.0, 0.5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which="major", length=6)
    ax2.tick_params(which="minor", length=4)

# else if only distance modulus (dM) was set to vary, then single plot in figure, showing delta dM.
else:
    fig = plt.figure(figsize=(22,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(n_PNe[good_fits], fit_dM[good_fits,0]-dM_in, alpha=0.5, label="Data")
    ax1.axhline(0.0, ls="--", c="k", alpha=0.7)
    ax1.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $\mu_{PNLF}$="+f"{round(np.nanmedian(fit_dM[good_fits&(n_PNe==obsv_PN_N),0]),2)} $\pm$ {round(np.nanstd(fit_dM[good_fits&(n_PNe==obsv_PN_N),0]),2)}", fontsize=30, y=1.03)
    ax1.plot(grp.n_PNe, grp.fit_dM, c="r", label="Median",)
    plt.fill_between(grp.n_PNe,  grp_up.fit_dM, grp_lo.fit_dM, color="red", alpha=0.4, linestyle="None", label="1$\sigma$")
    ax1.tick_params(axis='y', labelsize=25 )
    ax1.tick_params(axis='x', labelsize=25 )
    ax1.set_xlim(2,np.max(n_PNe)+5 )
    ax1.set_ylim(-1.5,1.5)
    ax1.set_xticks(np.arange(5, np.max(n_PNe)+5,10))
    ax1.set_yticks(np.arange(-1.5,2.0, 0.5))
    ax1.set_ylabel("$\Delta \ \mu_{PNLF}$", fontsize=25)
    ax1.legend(loc="upper right",fontsize=20)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", length=6)
    ax1.tick_params(which="minor", length=4)


plt.xlabel("Number of PNe", fontsize=25)
plt.subplots_adjust(hspace=0.4)

# depending upon if either both dM and c2 were varied, or just dM, save the plot with a corresponding name.
if vary_dict["c2"] == True:
    plt.savefig(f"Plots/completeness_testing/PNLF_dM_c2_simulations.png", bbox_inches='tight')
elif vary_dict["c2"] == False:
    plt.savefig(f"Plots/completeness_testing/PNLF_dM_simulations.png", bbox_inches='tight')



err_plot_bins = np.arange(0, n_PNe.max()+10,1)

fit_dM[~good_fits, 0] = np.nan
fit_c2[~good_fits, 0] = np.nan

sim_df = pd.DataFrame({"n_PNe":n_PNe, "fit_dM":fit_dM[:,0]-dM_in, "fit_c2":fit_c2[:,0]})
data_cut = pd.cut(sim_df.n_PNe, err_plot_bins)
grp = sim_df.groupby(by = data_cut).median()
grp_up = sim_df.groupby(by = data_cut).quantile(0.16)
grp_lo = sim_df.groupby(by = data_cut).quantile(0.84)

# open up all galaxies and get their N_PNe and dM_err
gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
idx = pd.IndexSlice
gal_names = np.unique(gal_df.index.get_level_values(0))
galaxy_loc = [gal_names,["center"]*len(gal_names)]
gal_cen_tuples = list(zip(*galaxy_loc))

N_obs_PNe = gal_df.loc[gal_cen_tuples, "PNe N"]
PNLF_dM_err = gal_df.loc[gal_cen_tuples, ["PNLF dM err up", "PNLF dM err lo"]].median(axis=1).values

################


plt.figure(figsize=(8,6))
plt.scatter(grp.n_PNe,grp_lo.fit_dM, c="grey", label="Simulations")
plt.scatter(N_obs_PNe, PNLF_dM_err, c=complete_limit_mag[:len(obs_comp_list)], marker="s", label=r"F3D $\mathrm{\delta \, \mu_{PNLF}}$")
cb=plt.colorbar()
cb.set_label("Completeness magntiude", fontsize=16)

def myExpFunc(x, a, b):
    return a * np.power(x, b)

newX = np.logspace(0.7, 2.3, base=10)
popt, pcov = curve_fit(myExpFunc, grp.n_PNe[5:-9].values, grp_lo.fit_dM[5:-9].values)
plt.plot(newX, myExpFunc(newX, *popt), c="k", ls="--")

ax = plt.gca()
ax.loglog()
for axis in [ax.xaxis, ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


plt.ylabel(r"$\mathrm{\delta \, \mu_{PNLF}}$", fontsize=16)
plt.xlabel(r"$\mathrm{N_{PNe}}$", fontsize=16)

plt.yticks([0.1, 0.5, 1.0],fontsize=16)
plt.xticks(fontsize=16)

plt.tick_params(which="minor", length=3 )

plt.legend(fontsize=14)

plt.savefig("Plots/simulations/error_comparison_between_simulations_and_data_log.png", bbox_inches='tight', dpi=300)



plt.figure(figsize=(8,6))
plt.scatter(grp.n_PNe,grp_lo.fit_dM, c="k", label="Simulations")
plt.scatter(N_obs_PNe, PNLF_dM_err, c="r", marker="s", label=r"F3D $\mathrm{\delta \, \mu_{PNLF}}$")

plt.ylabel(r"$\mathrm{\delta \, \mu_{PNLF}}$", fontsize=16)
plt.xlabel(r"$\mathrm{N_{PNe}}$", fontsize=16)

plt.xticks(fontsize=16)

plt.tick_params(which="minor", length=3 )

plt.legend(fontsize=14)

plt.savefig("Plots/simulations/error_comparison_between_simulations_and_data_non_log.png", bbox_inches='tight', dpi=300)

plt.show()



obs_comp_err = np.ones( ( len(obs_comp_list), len(np.arange(5, 206, 1),) ) )

for i in range(len(obs_comp_list)):
    fit_dM_std = np.std([fit_dM[i::len(obs_comp_list),0][j:j+5] for j in np.arange(0,1005, 5)], axis=1)
    obs_comp_err[i,:] = fit_dM_std

plt.figure(figsize=(8,6))
plt.scatter(np.tile(np.arange(5,206,1), len(obs_comp_list)), obs_comp_err, c=np.repeat(complete_limit_mag[:len(obs_comp_list)],201), alpha=0.7, label="Simulation")
cb=plt.colorbar()
cb.set_label("Completeness magntiude", fontsize=16)
plt.scatter(N_obs_PNe, PNLF_dM_err, c="r", marker="s", label=r"F3D $\mathrm{\delta \, \mu_{PNLF}}$")

ax = plt.gca()
ax.loglog()
for axis in [ax.xaxis, ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


plt.ylabel(r"$\mathrm{\delta \, \mu_{PNLF}}$", fontsize=16)
plt.xlabel(r"$\mathrm{N_{PNe}}$", fontsize=16)

plt.yticks([0.1, 0.5, 1.0],fontsize=16)
plt.xticks(fontsize=16)

plt.tick_params(which="minor", length=3 )

plt.legend(fontsize=14)

plt.savefig("Plots/simulations/error_comparison_between_simulations_and_data_log_coloured_by_obs.png", bbox_inches='tight', dpi=300)