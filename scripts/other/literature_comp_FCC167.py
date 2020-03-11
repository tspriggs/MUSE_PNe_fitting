# Literature comparison FCC167
from astropy.io import ascii, fits
import numpy as np
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import pandas as pd
import matplotlib.pyplot as plt



PNe_df = pd.read_csv("exported_data/FCC167/FCC167center_PNe_df.csv")

x_y_list = np.load("exported_data/FCC167/FCC167center_PNe_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list]) # separate out from the list the list of x coordinates, as well as y coordinates.
y_PNe = np.array([y[1] for y in x_y_list])

# with fits.open(DIR_dict["RAW_DATA"]) as hdu_wcs:
with fits.open(f"/local/tspriggs/Fornax_data_cubes/FCC167center.fits") as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)


c = ascii.read("galaxy_data/FCC167_data/NGC1380_PNe_from_paper.txt", data_start=1)
lit_RA = list(c["RA"])
lit_Dec = list(c["Dec"])
lit_m = np.array(c["mag"])

x_y_lit = np.ones((len(x_PNe),2))
for i in np.arange(0, len(lit_RA)):
    x_y_lit[i] = utils.skycoord_to_pixel(SkyCoord(ra=Angle(lit_RA[i], u.hourangle), dec=Angle(lit_Dec[i], u.deg), frame="fk5"), wcs_obj)

smallest_sep = []
for r,d in zip(lit_RA, lit_Dec):
    sepa = []
    for r1,d1 in zip(PNe_df["Ra (J2000)"], PNe_df["Dec (J2000)"]):
        c1 = SkyCoord(Angle(r, u.hourangle), Angle(d, u.deg), frame="fk5")
        c2 = SkyCoord(Angle(r1, u.hourangle), Angle(d1, u.deg), frame="fk5")
        sepa.append(c1.separation(c2).degree)
    smallest_sep.append(sepa)
    
diff = np.array([np.min(smallest_sep[i]) for i in range(0, len(smallest_sep))])
potential_diff = diff[diff<0.001]
print(np.where(diff<0.001))

lit_index = np.squeeze(np.where(diff<0.001))

indx = np.array(([np.squeeze(np.where(smallest_sep[i] == np.min(smallest_sep[i]))) for i in range(0, len(smallest_sep))]))
#print(indx[[0,2,6,20]])
#print(lit_m[[0,2,6,20]])
PNe_df.loc[indx[np.where(diff<0.001)]]
F3D_index = indx[np.where(diff<0.001)]

matched_lit_m =lit_m[lit_index]
matched_F3D_m = PNe_df["m 5007"].loc[[94, 81, 57, 74]]#indx[np.where(diff<0.001)]]


plt.scatter(matched_F3D_m, matched_lit_m)
plt.plot(np.arange(26.5,29), np.arange(26.5,29))
plt.xlabel("F3D sample")
plt.ylabel("Feldemier sample")

print("Center PNe m5007")
print(matched_lit_m)
print(matched_F3D_m)

print(f"Matched PNe indexes, for plotting: {[94, 81, 57, 74]}")

############################################
#halo
# matched_lit_m =lit_m[[1, 3,  7,  8, 12, 13, 16, 18, 19, 25, 28, 29, 32, 33, 36, 37, 40]]
# matched_F3D_m = PNe_df["m 5007"].loc[[13, 45 , 8 ,21 , 4 ,37 ,25 ,12, 23 ,35 ,46, 38, 34, 36 ,43 ,27, 40]] # center corrected list of indx

# plt.scatter(matched_F3D_m, matched_lit_m)
# plt.plot(np.arange(26.5,29), np.arange(26.5,29))
# plt.xlabel("F3D sample")
# plt.ylabel("Feldemier sample")

# print("Center PNe m5007")
# print(matched_lit_m)
# print(matched_F3D_m)

###########################
mid_matched_lit_m = [26.718, 26.77,  26.821, 26.952, 27.079, 27.079, 27.107, 27.188, 27.201, 27.321, 27.399, 27.411, 27.489, 27.528, 27.747, 27.76,  27.812]
mid_matched_F3D_m = [ 27.574294, 27.296383, 27.438540, 27.477897, 27.868759, 27.446263, 27.561705, 27.451270, 27.629726, 27.897316, 27.784497, 27.767381, 
                      27.985602, 27.848569, 28.111939, 28.104020 ,28.629938]
# with plt.xkcd(2, length=100, randomness=1):
plt.figure(figsize=(10,6))
plt.scatter(mid_matched_F3D_m, mid_matched_lit_m, label="middle matches")
plt.scatter(matched_F3D_m, matched_lit_m, label="Centre matches")
plt.plot(np.arange(26.5,29), np.arange(26.5,29), label="y=x")
plt.plot(np.arange(26.5,29)+0.45, np.arange(26.5,29), ls="--",c="k", label="y=x+0.45")
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$\rm m_{5007, \ F3D}$", fontsize=20)
plt.ylabel(r"$\rm m_{5007, \ Feldemier}$", fontsize=20)

print(f"F3D PNe numbers that matched: {F3D_index}")

plt.savefig("Plots/FCC167/FCC167_lit_comparison.png", bbox_inches='tight')