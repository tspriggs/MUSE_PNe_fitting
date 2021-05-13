import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import yaml

from functions.file_handling import paths 
from functions.PNLF import calc_PNLF_interp_comp, scale_PNLF

# Setup for argparse
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True, help="The name of the galaxy to be analysed.")
my_parser.add_argument("--save", action="store_true", default=False, help="Flag for saving the plots, default is False")
my_parser.add_argument("--show", action="store_true", default=False, help="Flag to decide if the plots made with this script are shown afterwards. Default is False.")
args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy   # galaxy name, format of FCC000
save_plot = args.save       # save plots
show = args.show

loc = "center" 

DIR_dict = paths(galaxy_name, loc)

with open(DIR_dict["YAML"], "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
galaxy_info = yaml_info[f"{galaxy_name}_{loc}"]

PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")

obs_comp = np.load(DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy")

gal_m_5007 = PNe_df["m 5007"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_up = PNe_df["mag error up"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_lo = PNe_df["mag error lo"].loc[PNe_df["ID"].isin(["PN"])].values

step = 0.001
M_star = -4.53
m_5007 = np.arange(22, 34, step)
M_5007 = np.arange(M_star, M_star+12, step)


gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
best_fit_dM = gal_df.loc[(galaxy_name, loc), "PNLF dM"]

dM_err_up = gal_df.loc[(galaxy_name, loc), "PNLF dM err up"] 
# dM_err_lo = MC_dM_50-MC_dM_16
dM_err_lo = gal_df.loc[(galaxy_name, loc), "PNLF dM err lo"] 

PNLF_best_fit, PNLF_interp, PNLF_comp_corr = calc_PNLF_interp_comp(best_fit_dM, 0.307, obs_comp)

PNLF_err_16, PNLF_16_interp, PNLF_comp_corr_16 = calc_PNLF_interp_comp(best_fit_dM-dM_err_lo, 0.307, obs_comp)

PNLF_err_84, PNLF_84_interp, PNLF_comp_corr_84 = calc_PNLF_interp_comp(best_fit_dM+dM_err_up, 0.307, obs_comp)

bw = 0.2

## Plotting of binned PNe and PNLF and completeness corrected PNLF
plt.figure(figsize=(10,7))

# histogram plot of the observed PNe
plt.hist(gal_m_5007, bins=np.arange(min(gal_m_5007), max(gal_m_5007) + bw, bw), ec="black", alpha=0.8, zorder=1, label="PNe") # histogram of gal_m_5007

# Plot the interpolated, PNLF that is using the initial best fit dM value
plt.plot(m_5007, scale_PNLF(gal_m_5007, PNLF_interp, PNLF_comp_corr, bw, step), c="k", label="Best-fit C89 PNLF")

# to show the 1 sigma uncertainty range, use the fillbetween, using scaled upper (16th) and lower (84th) percentile interpolated PNLFS
plt.fill_between(m_5007, scale_PNLF(gal_m_5007, PNLF_16_interp, PNLF_comp_corr, bw, step), scale_PNLF(gal_m_5007, PNLF_84_interp, PNLF_comp_corr, bw, step), \
                 alpha=0.4, color="b", zorder=2, label=r"C89 PNLF 1$\sigma$")

#Plot the completeness corrected, interpolated PNLF form, along with the associated uncertainty regions
plt.plot(m_5007, scale_PNLF(gal_m_5007, PNLF_comp_corr, PNLF_comp_corr, bw, step), c="k", ls="-.", label="Incompleteness-corrected C89 PNLF") 
plt.fill_between(m_5007, scale_PNLF(gal_m_5007, PNLF_comp_corr_16, PNLF_comp_corr, bw, step), scale_PNLF(gal_m_5007, PNLF_comp_corr_84, PNLF_comp_corr, bw, step), \
                 alpha=0.4, color="b",zorder=2) 

plt.xlim(26.0,30.0)
plt.ylim(0, np.max(scale_PNLF(gal_m_5007, PNLF_best_fit, PNLF_comp_corr, bw, step)))



idx = np.where(M_5007 <= M_star+2.5)[0]
N_PNLF = np.sum(PNLF_best_fit[idx]* (len(gal_m_5007) / (np.sum(PNLF_comp_corr)*step))) * step


## Check to see if halo or middle data exists, if so, plot the data overlayed on the central pointing data

if f"{galaxy_name}_halo" in yaml_info:
    halo_DIR_dict = paths(galaxy_name, "halo")
    halo_PNe_df = pd.read_csv(halo_DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")
    halo_m_5007 = halo_PNe_df["m 5007"].loc[halo_PNe_df["ID"].isin(["PN"])].values
    plt.hist(halo_m_5007, bins=np.arange(min(gal_m_5007), max(gal_m_5007) + bw, bw), ec="black", alpha=0.5, 
                    zorder=3, label="Halo PNe", color="green") # histogram of gal_m_5007
    halo_comp = np.load(halo_DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy")

    halo_PNLF, halo_PNLF_interp, halo_PNLF_comp_corr = calc_PNLF_interp_comp(best_fit_dM, 0.307, halo_comp)

    plt.plot(m_5007, scale_PNLF(halo_m_5007, halo_PNLF_comp_corr, halo_PNLF_comp_corr, bw, step), \
             c="green", ls="-.", label="Incompleteness-corrected halo PNLF") 



if f"{galaxy_name}_middle" in yaml_info:
    mid_DIR_dict = paths(galaxy_name, "middle")
    middle_PNe_df = pd.read_csv(mid_DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")
    middle_m_5007 = middle_PNe_df["m 5007"].loc[middle_PNe_df["ID"].isin(["PN"])].values
    plt.hist(middle_m_5007, bins=np.arange(min(gal_m_5007), max(gal_m_5007) + bw, bw), ec="black", alpha=0.5, 
                    zorder=2, label="Middle PNe", color="red") # histogram of gal_m_5007
    mid_comp = np.load(mid_DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy")
    mid_PNLF, mid_PNLF_interp, mid_PNLF_comp_corr = calc_PNLF_interp_comp(best_fit_dM, 0.307, mid_comp)

    plt.plot(m_5007, scale_PNLF(middle_m_5007, mid_PNLF_comp_corr, mid_PNLF_comp_corr, bw, step), \
             c="red", ls="-.", label="Incompleteness-corrected middle PNLF") 


plt.title(f"{galaxy_name}, dM={round(best_fit_dM,2)}$"+"^{+"+f"{round(dM_err_up,2)}"+"}"+f"_{ {-round(dM_err_lo,2)} }$")
plt.xlabel(r"$m_{5007}$", fontsize=15)
plt.ylabel(r"$N_{PNe} \ per \ bin$", fontsize=15)
plt.xlim(26.0,30.0)
plt.ylim(0, np.max(scale_PNLF(gal_m_5007, PNLF_best_fit, PNLF_comp_corr, bw, step)[idx])*1.5)
plt.legend(loc="upper left", fontsize=12)

if show == True:
    plt.show()

if save_plot == True:
    plt.savefig(DIR_dict["PLOT_DIR"]+"_fitted_PNLF_combined_data.png", bbox_inches='tight', dpi=300)
    plt.savefig(DIR_dict["PLOT_DIR"]+"_fitted_PNLF_combined_data.pdf", bbox_inches='tight', dpi=300)
