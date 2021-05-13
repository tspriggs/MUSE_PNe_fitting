import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from astropy.io import fits
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astroquery.simbad import Simbad
from matplotlib.cm import ScalarMappable

# Load up each galaxy and store the central RA and DEC values

# construct a 3D cube, with size X pixels, Y pixles, Z pixels, which covers n arcmin or degrees square in X and Y axis

# First, construct a 2D plane image of RA vs DEC, with galaxy markers coloured by distance.
# Find middle point of cluster. Then shift 
# place galaxy markers at RA and DEC position, though at a distance of Zmin + dM

# Get Cluster centre RA and DEC coords, maybe use NGC1399 as central point.

with open("config/galaxy_info.yaml", "r") as yaml_data:
    yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col="Galaxy")
gal_df = gal_df.loc[gal_df["loc"]=="center"]



gal_centre = "NGC1399"
BL_cen_D = 21.1
Tu_cen_D = 19.95
result_table = Simbad.query_object(gal_centre)
RA_c = result_table["RA"][0]
DEC_c = result_table["DEC"][0]
c_centre_Bl = SkyCoord(Angle(RA_c, u.hourangle), Angle(DEC_c, u.deg), distance=BL_cen_D*u.Mpc, frame="fk5")
c_centre_Tu = SkyCoord(Angle(RA_c, u.hourangle), Angle(DEC_c, u.deg), distance=Tu_cen_D*u.Mpc, frame="fk5")

# sep = c_centre.separation(c_FCC167)
# print(sep.arcmin, "arcmin")
# sep_3D = c_centre.separation_3d(c_FCC167)
# print(sep_3D)


# 2D scatter plot in RA and DEC, with points coloured by distance
gal_names = np.unique(gal_df.index.values)
# F_corr = [yaml_info[f"{gal}_center"]["F_corr"] for gal in gal_names]
# gal_df["F_corr"] = F_corr
PN_lim_idx = gal_df["PNe N"] >=5.
PNLF_dist = 10.**(((gal_df["PNLF dM"]) -25.) / 5.)
# PNLF_dist = 10.**(((gal_df["PNLF dM"]) -25. +(-2.5*np.log10(gal_df["F_corr"]))) / 5.)
PNLF_dM = gal_df["PNLF dM"]# + (-2.5*np.log10(gal_df["F_corr"]))

Bl_dist   = 10.**(((gal_df["Bl dM"]) -25.) / 5.)


RA_list = np.ones(len(gal_names))
DEC_list = np.ones(len(gal_names))
dist_diff = np.ones(len(gal_names))
sep = np.ones(len(gal_names))
sep_3D_PNLF_Bl = np.ones(len(gal_names))
sep_3D_Bl_Bl = np.ones(len(gal_names))
sep_3D_PNLF_Tu = np.ones(len(gal_names))
sep_3D_Bl_Tu = np.ones(len(gal_names))


for i, gal in enumerate(gal_names):
    result_table = Simbad.query_object(gal)
    c_gal_PNLF = SkyCoord(Angle(result_table["RA"][0], u.hourangle), Angle(result_table["DEC"][0], u.deg), distance=PNLF_dist[i]*u.Mpc, frame="fk5")
    c_gal_Bl = SkyCoord(Angle(result_table["RA"][0], u.hourangle), Angle(result_table["DEC"][0], u.deg), distance=Bl_dist[i]*u.Mpc, frame="fk5")
    RA_list[i] = c_gal_PNLF.ra.deg
    DEC_list[i] = c_gal_PNLF.dec.deg
    sep[i] = c_centre_Bl.separation(c_gal_PNLF).arcmin
    sep_3D_PNLF_Bl[i] = c_centre_Bl.separation_3d(c_gal_PNLF).value
    sep_3D_Bl_Tu[i] = c_centre_Tu.separation_3d(c_gal_Bl).value
    sep_3D_Bl_Bl[i] = c_centre_Bl.separation_3d(c_gal_Bl).value
    sep_3D_PNLF_Tu[i] = c_centre_Tu.separation_3d(c_gal_PNLF).value


plt.figure(figsize=(12,8))
# sim_dist = 10.**(((gal_df["Bl dM"]) -25.) / 5.)
N_PNe_gals = gal_df["PNLF N"]
# plt.scatter(RA_list, DEC_list, c=sim_dist, s=70)
# plt.scatter(RA_list, DEC_list, c=round(N_PNe_gals), s=70)
plt.scatter(RA_list, DEC_list, c=round(N_PNe_gals), s=100000/(PNLF_dist**2))
plt.colorbar()
plt.scatter(c_centre_Bl.ra.deg, c_centre_Bl.dec.deg, c="r", s=70)
plt.annotate("NGC1399", (c_centre_Bl.ra.deg+0.05, c_centre_Bl.dec.deg+0.1))
plt.annotate("FCC167", (RA_list[np.where(gal_names=="FCC167")], DEC_list[np.where(gal_names=="FCC167")]+0.05))
plt.annotate("FCC219", (RA_list[np.where(gal_names=="FCC219")]+0.05, DEC_list[np.where(gal_names=="FCC219")]-0.15))
# plt.ylim(-33.45,-37.6)
plt.xlim(56.7,52.45)


# Create 3D scatter plot of RA and DEc in degrees, colour coded in distance.
PNLF_dist_err = 0.2*np.log(10)*gal_df.loc[PN_lim_idx, "PNLF dM err up"]*PNLF_dist[PN_lim_idx]
Bl_dist_err = 0.2*np.log(10)*gal_df.loc[PN_lim_idx, "Bl dM err"]*Bl_dist[PN_lim_idx]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
plt.title("NGC 1399 using Blakeslee dM")

# ax.scatter(RA_list[PN_lim_idx], PNLF_dist[PN_lim_idx], DEC_list[PN_lim_idx], c=PNLF_dist_err[PN_lim_idx], s=50)
ax.scatter(RA_list[PN_lim_idx], PNLF_dist[PN_lim_idx], DEC_list[PN_lim_idx], c=gal_df.loc[PN_lim_idx, "PNLF N"], s=50)
ax.scatter(c_centre_Bl.ra.deg, BL_cen_D, c_centre_Bl.dec.deg, c="r", s=50)
n=np.where(gal_names=="FCC167")

ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dist (Mpc)")
ax.set_zlabel("DEC (deg)")

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
plt.title("PNLF distances, \n Tonry 2001 SBF NGC 1399 (19.95)")

ax.scatter(RA_list[PN_lim_idx], PNLF_dist[PN_lim_idx], DEC_list[PN_lim_idx], c=gal_df.loc[PN_lim_idx, "PNLF N"], s=50)
ax.scatter(c_centre_Tu.ra.deg, 19.95, c_centre_Tu.dec.deg, c="r", s=50)
n=np.where(gal_names=="FCC167")

ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dist (Mpc)")
ax.set_zlabel("DEC (deg)")


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
plt.title(f"Blakeslee SBF distances, \n NGC 1399 @ {BL_cen_D}Mpc")

ax.scatter(RA_list[PN_lim_idx], Bl_dist[PN_lim_idx], DEC_list[PN_lim_idx], c=gal_df.loc[PN_lim_idx, "PNLF N"], s=50)
ax.scatter(c_centre_Tu.ra.deg, BL_cen_D, c_centre_Tu.dec.deg, c="r", s=50)
n=np.where(gal_names=="FCC167")

ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dist (Mpc)")
ax.set_zlabel("DEC (deg)")


plt.figure(figsize=(8,6))
# Using the separation between galaxies to see how PNe are distributed, color scale is log10 mass.
# plt.scatter(sep, gal_df["Rmag"] - gal_df["PNLF dM"], c=np.log10(gal_df["Mass"]))
plt.scatter(sep[PN_lim_idx], gal_df.loc[PN_lim_idx,"lit Rmag"]-gal_df.loc[PN_lim_idx, "PNLF dM"], c=PNLF_dist_err[PN_lim_idx], s=80, edgecolors="k", linewidth=0.8)
cb = plt.colorbar()
cb.ax.set_title("$\sigma_\mathrm{D_{PNLF}}$ (Mpc)", fontsize=22)
cb.ax.tick_params(labelsize=15) 
# cb.set_label("$\sigma_\mathrm{D_{PNLF}}$ (Mpc)", fontsize=20)
plt.xlabel("projection separation (arcmin)", fontsize=20,)
plt.ylabel("M$_{r}$", fontsize=20)
plt.ylim(-24,-17)
plt.tick_params(axis="x", labelsize=15)
plt.tick_params(axis="y", labelsize=15)

plt.savefig("Plots/F3D_cluster_work/tangential_dist_Rmag.png", bbox_inches='tight', dpi=500)


fig =plt.figure(figsize=(26,7))
ax = fig.add_subplot(1,1,1)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

scales = PNLF_dist_err#gal_df["PNLF dM err up"]
norm = plt.Normalize(scales.min(), scales.max())
sm =  ScalarMappable(norm=norm)
sm.set_array([])
cbaxes = fig.add_axes([0.92, 0.1, 0.01, 0.8]) 
cb = plt.colorbar(sm, cax = cbaxes) 
cb.ax.set_title("$\sigma_\mathrm{D_{PNLF}}$ (Mpc)", fontsize=20)

ax1 = fig.add_subplot(1,3,1)
# Using the separation between galaxies to see how PNe are distributed, color scale is log10 mass.
ax1.scatter(sep_3D_Bl_Bl[PN_lim_idx], gal_df.loc[PN_lim_idx, "lit Rmag"] - gal_df.loc[PN_lim_idx, "Bl dM"], c=Bl_dist_err[PN_lim_idx], s=40)
ax1.set_title(f"Blakeslee SBF distances, w.r.t \n Blakeslee SBF NGC1399 ({BL_cen_D}Mpc)", fontsize=18)
ax1.set_xlim(0,4)
ax1.set_ylim(-24.,-17)
ax1.tick_params(axis="x", labelsize=15)
ax1.tick_params(axis="y", labelsize=15)
# plt.xlabel("3D separation in Mpc", fontsize=18)
plt.ylabel("M$_\mathrm{r}$", fontsize=20)

ax2 = fig.add_subplot(1,3,2)
# sep_3D_Bl_Tu
ax2.scatter(sep_3D_Bl_Tu[PN_lim_idx], gal_df.loc[PN_lim_idx, "lit Rmag"] - gal_df.loc[PN_lim_idx, "Bl dM"], c=Bl_dist_err[PN_lim_idx], s=40)
# ax2.scatter(sep_3D_PNLF_Bl, gal_df["Rmag"] - PNLF_dM, c=PNLF_dist_err)
ax2.set_title(f"Blakeslee SBF distances, w.r.t \n Tonry SBF NGC1399 ({Tu_cen_D}Mpc)", fontsize=18)
ax2.set_xlim(0,4)
ax2.set_ylim(-24.,-17)
ax2.tick_params(axis="x", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
plt.xlabel("3D separation in Mpc", fontsize=20, labelpad=15)
# plt.ylabel("M$_{r}$", fontsize=20)

ax3 = fig.add_subplot(1,3,3)
ax3.scatter(sep_3D_PNLF_Tu[PN_lim_idx], gal_df.loc[PN_lim_idx, "lit Rmag"] - PNLF_dM[PN_lim_idx], c=PNLF_dist_err[PN_lim_idx], s=40)
ax3.set_title(f"PNLF distances, w.r.t \n Tonry SBF NGC1399 ({Tu_cen_D})", fontsize=18)
ax3.set_xlim(0,4)
ax3.set_ylim(-24.,-17)
ax3.tick_params(axis="x", labelsize=15)
ax3.tick_params(axis="y", labelsize=15)
# plt.xlabel("3D separation in Mpc", fontsize=18)
# plt.ylabel("M$_{r}$", fontsize=20)



plt.savefig("Plots/F3D_cluster_work/fornax_structure_comparisons.png", bbox_inches='tight', dpi=300)



gal_names = np.unique(gal_df.index.get_level_values(0))
galaxy_loc = [np.unique(gal_df.index.get_level_values(0)),["center"]*len(np.unique(gal_df.index.get_level_values(0)))]
gal_cen_tuples = list(zip(*galaxy_loc))

Fornax_sigma = 300 # m/s

Fornax_LOS_list = []
for gal in gal_names:
    Fornax_LOS_list.append(yaml_info[f"{gal}_center"]["velocity"])

Fornax_LOS_list = np.array(Fornax_LOS_list)
plt.figure(figsize=(8,5))
plt.scatter(sep_3D_PNLF_Tu, (Fornax_LOS_list-1425)/Fornax_sigma, c=gal_df.loc[gal_names, "sigma"])
cb = plt.colorbar()
cb.set_label("$\mathrm{\sigma_{Stars}} \, (km s^{-1})$")
plt.title("3D distance separation")

# for i in range(len(gal_names)):
    # plt.annotate(gal_names[i], (sep_3D_PNLF_Tu[i]+0.005, (Fornax_LOS_list[i]-1425)/Fornax_sigma), fontsize=8 )
plt.ylabel("$V_{LOS, \, F3D}$/ $\sigma_{F3D}$ (km $\, s^{-1}$)")
plt.xlabel("3D separation in Mpc")
plt.savefig("Plots/F3D_cluster_work/LOSV_by_cluster_sigma_vs_3D_sep.png", bbox_inches='tight', dpi=300)

# Projected separation
Fornax_LOS_list = np.array(Fornax_LOS_list)
plt.figure(figsize=(8,5))
plt.scatter(sep, (Fornax_LOS_list-1425)/Fornax_sigma, c=gal_df.loc[gal_names, "sigma"])
cb = plt.colorbar()
cb.set_label("$\mathrm{\sigma_{Stars}} \, (km s^{-1})$")
plt.title("Tangential separation")
# for i in range(len(gal_names)):
    # plt.annotate(gal_names[i], (sep_3D_PNLF_Tu[i]+0.005, (Fornax_LOS_list[i]-1425)/Fornax_sigma), fontsize=8 )
plt.ylabel("$V_{LOS, \, F3D}$/ $\sigma_{F3D}$ (km $\, s^{-1}$)")
plt.xlabel("separation in arcmin")
plt.savefig("Plots/F3D_cluster_work/LOSV_by_cluster_sigma_vs_tang_sep.png", bbox_inches='tight', dpi=300)


# plt.show()