import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm

from scipy import stats

from lmfit import Parameters

from matplotlib.ticker import (AutoMinorLocator)

from functions.file_handling import paths
from functions.PNLF import calc_PNLF,  PNLF_analysis


np.random.seed(42)


gal = "FCCtest"
loc = "center"
DIR_dict = paths(gal, loc)

PNe_df = pd.read_csv(f"exported_data/{gal}/{gal}{loc}_PNe_df.csv")
gal_m_5007 = PNe_df["m 5007"].loc[PNe_df["ID"] == "PN"]

with open("config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

galaxy_data = galaxy_info[f"{gal}_{loc}"]

# Change this to alter how many iterations of the simulation you want to run.
n_sim = 2500

# Input values for distance modulus (dM_in) and the c2 paramter (in_c_2)
dM_in = 31.45
in_c_2 = 0.307

mag_step = 0.001
M_star = -4.53
M_5007 = np.arange(M_star, 0.53, mag_step)
m_5007 = np.arange(26, 31, mag_step)
obs_comp = np.load(DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy") 


gal_m_5007_err_up = PNe_df["mag error up"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_lo = PNe_df["mag error lo"].loc[PNe_df["ID"].isin(["PN"])].values
mag_err_list = np.median([gal_m_5007_err_up, gal_m_5007_err_lo],0) # average error

slope, intercept, r_value, p_value, std_err = stats.linregress(gal_m_5007, mag_err_list)
data_shift_err = (slope*gal_m_5007) + intercept


PNLF_sim_result = []

# change this to alter the number of PNe that can be within a simulation sample.
# 5 is the lower limit, if it is set lower, the simulation may fail.
# 161 was chosen to eoncompass the largest observed sample of PNe from the catalogue from Foranx3D work.
# 1 is the step size.
n_PNe = np.random.choice(np.arange(5, 161, 1), n_sim)

PNLF_params = Parameters()
PNLF_params.add("dM", value=31.3, min=30.0, max=33.0 ,vary=True)
PNLF_params.add("c1", value=1, min=0.00, vary=False)
PNLF_params.add("c2", value=0.307, min=-2., max=2.0, vary=False)
PNLF_params.add("c3", value=3., vary=False)
PNLF_params.add("M_star", value=-4.53, min=-4.6, max=-4.3, vary=False)


fit_dM = np.ones((n_sim,3))
fit_c2 = np.ones((n_sim,2))
fit_c1 = np.ones((n_sim,2))
fit_c3 = np.ones(n_sim)
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


PNLF = calc_PNLF(M_star+dM_in, M_5007+dM_in, c_2=in_c_2)
PNLF_comp_corr = np.array(np.interp(m_5007, M_5007+dM_in, PNLF)*obs_comp)

for k, n in tqdm(enumerate(n_PNe), total=n_sim):

    n_sim_pois = np.random.poisson(n)
    while n_sim_pois< 5:
        n_sim_pois = np.random.poisson(n)

    PNLF_sample = np.random.choice(m_5007, n_sim_pois, p=np.array((PNLF_comp_corr) / np.sum(PNLF_comp_corr)))
    data_shift_err = (slope*PNLF_sample) + intercept
    PNe_sample = PNLF_sample 
    n_sim_PNe[k] = n_sim_pois

    vary_dict={"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}
    PNLF_results = PNLF_analysis(gal, loc, PNe_sample, obs_comp, M_5007, m_5007, c2_in=0.307, vary_dict=vary_dict, comp_lim=False)

    fit_dM[k,0] = PNLF_results.params["dM"]
    fit_c3[k] = PNLF_results.params["c3"]
    fit_c2[k,0] = PNLF_results.params["c2"]
    fit_c2[k,1] = PNLF_results.params["c2"].stderr

    fit_Mstar[k,0] = PNLF_results.params["M_star"]
    fit_Mstar[k,1] = PNLF_results.params["M_star"].stderr
    chi_2[k] = PNLF_results.chisqr
    chi_r[k] = PNLF_results.redchi

    brightest_m_5007[k] = np.min(PNe_sample)


bad_fits = ((fit_dM[:,0]>PNLF_results.params["dM"].min+0.1) & (fit_dM[:,0]<PNLF_results.params["dM"].max-0.1)) & \
           ((fit_c2[:,0]<PNLF_results.params["c2"].max-0.1) & (fit_c2[:,0]>PNLF_results.params["c2"].min+0.1))
   

print("SIMULATION COMPLETE")
print(f"{n_sim} simulations, across {np.min(n_PNe)} to {np.max(n_PNe)} PNe per sample.")

##### Plotting #####

bad_fits = ((fit_dM[:,0]>PNLF_results.params["dM"].min+0.1) & (fit_dM[:,0]<PNLF_results.params["dM"].max-0.1)) & \
           ((fit_c2[:,0]<PNLF_results.params["c2"].max-0.1) & (fit_c2[:,0]>PNLF_results.params["c2"].min+0.1))

bins = np.arange(0, n_PNe.max()+5,5)

fit_dM[~bad_fits, 0] = np.nan
fit_c2[~bad_fits, 0] = np.nan

sim_df = pd.DataFrame({"n_PNe":n_PNe, "fit_dM":dM_in-fit_dM[:,0], "fit_c2":in_c_2 - fit_c2[:,0]})
data_cut = pd.cut(sim_df.n_PNe, bins)
grp = sim_df.groupby(by = data_cut).median()
grp_up = sim_df.groupby(by = data_cut).quantile(0.16)
grp_lo = sim_df.groupby(by = data_cut).quantile(0.84)

obsv_PN_N = 65

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
    ax1.scatter(n_PNe[bad_fits], dM_in - fit_dM[bad_fits,0], alpha=0.7, label="Data")
    ax1.axhline(0.0, ls="--", c="k", alpha=0.7)
    ax1.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $\mu_{PNLF}$="+f"{round(np.nanmedian(fit_dM[bad_fits&(n_PNe==obsv_PN_N),0]),2)} $\pm$ {round(np.nanstd(fit_dM[bad_fits&(n_PNe==obsv_PN_N),0]),2)}", fontsize=30, y=1.03)
    ax1.scatter(grp.n_PNe, grp.fit_dM, c="r", label="Median", s=50)
    plt.fill_between(grp.n_PNe,  grp_up.fit_dM, grp_lo.fit_dM, color="red", alpha=0.4, linestyle="None", label="1$\sigma$")
    ax1.tick_params(axis='y', labelsize=30 )
    ax1.tick_params(axis='x', labelsize=30 )
    ax1.set_xticks(np.arange(0, np.max(n_PNe)+10,10),)
    ax1.set_ylabel("$\Delta \ \mu_{PNLF}$", fontsize=30)
    ax1.set_yticks(np.arange(-1.0,1.5, 0.5))
    ax1.legend(loc="upper right",fontsize=20)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", length=6)
    ax1.tick_params(which="minor", length=4)

    # delta c2 plot is ax2
    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(n_PNe[bad_fits], in_c_2 - fit_c2[bad_fits,0], alpha=0.7, )
    ax2.scatter(grp.n_PNe, grp.fit_c2, c="r", s=50)
    plt.fill_between(grp.n_PNe,  grp_up.fit_c2, grp_lo.fit_c2, color="red", alpha=0.4, linestyle="None")
    ax2.axhline(0.0, c="k", ls="--", alpha=0.7)
    ax2.set_ylim(-1.5,2)
    ax2.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $c_{2}$=" + f"{round(np.nanmedian(fit_c2[bad_fits&(n_PNe==obsv_PN_N),0]),3)} \
                    $\pm$ {round(np.nanstd(fit_c2[bad_fits&(n_PNe==obsv_PN_N),0]),2)}", fontsize=30, y=1.03)

    ax2.tick_params(axis='y', labelsize=30 )
    ax2.tick_params(axis='x', labelsize=30 )
    ax2.set_xticks(np.arange(0, np.max(n_PNe)+10,10),)
    ax2.set_ylabel("$\Delta \ c_{2}$", fontsize=30)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which="major", length=6)
    ax2.tick_params(which="minor", length=4)

# else if only distance modulus (dM) was set to vary, then single plot in figure, showing delta dM.
else:
    fig = plt.figure(figsize=(22,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title("At $N_{PNe}=$"+f"{obsv_PN_N},"+" median $\mu_{PNLF}$=" + f"{round(np.nanmedian(fit_dM[bad_fits&(n_PNe==obsv_PN_N),0]),2)} \
                    $\pm$ {round(np.nanstd(fit_dM[bad_fits&(n_PNe==obsv_PN_N),0]),2)}", y=1.03, fontsize=30 )
    ax1.scatter(n_PNe[bad_fits], dM_in - fit_dM[bad_fits,0], alpha=0.7, label="Data",)
    ax1.axhline(0.0, ls="--", c="k", )
    ax1.scatter(grp.n_PNe, grp.fit_dM, c="r", label="Median", s=50)
    plt.fill_between(grp.n_PNe,  grp_up.fit_dM, grp_lo.fit_dM, color="red", alpha=0.4, linestyle="None", label="1$\sigma$")
    ax1.set_ylabel("$\Delta \ \mu_{PNLF}$", fontsize=30)
    ax1.legend(loc="upper right",fontsize=20)
    ax1.tick_params(axis='y', labelsize=30 )
    ax1.tick_params(axis='x', labelsize=30 )
    ax1.set_xticks(np.arange(0, np.max(n_PNe)+10,10),)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", length=6)
    ax1.tick_params(which="minor", length=4)


plt.xlabel("Number of PNe", fontsize=30)
plt.subplots_adjust(hspace=0.4)

# depending upon if either both dM and c2 were varied, or just dM, save the plot with a corresponding name.
if vary_dict["c2"] == True:
    plt.savefig(f"Plots/{gal}_dM_c2_simulations.png", bbox_inches='tight')
elif vary_dict["c2"] == False:
    plt.savefig(f"Plots/{gal}_dM_simulations.png", bbox_inches='tight')
