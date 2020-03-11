# Literature comparison FCC219
from astropy.io import ascii, fits
import numpy as np
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import pandas as pd
import matplotlib.pyplot as plt



PNe_df = pd.read_csv("exported_data/FCC219/FCC219center_PNe_df.csv")

x_y_list = np.load("exported_data/FCC219/FCC219center_PNe_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list]) # separate out from the list the list of x coordinates, as well as y coordinates.
y_PNe = np.array([y[1] for y in x_y_list])

with fits.open(f"/local/tspriggs/Fornax_data_cubes/FCC219center.fits") as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)


# with fits.open(DIR_dict["RAW_DATA"]) as hdu_wcs:
PNe_hdulist = fits.open("galaxy_data/FCC219_data/PNe1404.fit")
list_1404 = PNe_hdulist[1].data
list_1404 = list_1404[-47:]
RA_1404 =    [list_1404[i][7] for i in np.arange(0,len(list_1404))]
Dec_1404 =   [list_1404[i][8] for i in np.arange(0,len(list_1404))]
m5007_1404 = np.array([list_1404[i][4] for i in np.arange(0,len(list_1404))])

x_y_lit = np.ones((len(RA_1404),2))
for i in np.arange(0, len(list_1404)):
    x_y_lit[i] = utils.skycoord_to_pixel(SkyCoord(ra=Angle(RA_1404[i], u.hourangle), dec=Angle(Dec_1404[i], u.deg), frame="fk5"), wcs_obj)

smallest_sep = []
for r,d in zip(RA_1404, Dec_1404):
    sepa = []
    for r1,d1 in zip(PNe_df["Ra (J2000)"], PNe_df["Dec (J2000)"]):
        c1 = SkyCoord(Angle(r, u.hourangle), Angle(d, u.deg), frame="fk5")
        c2 = SkyCoord(Angle(r1, u.hourangle), Angle(d1, u.deg), frame="fk5")
        sepa.append(c1.separation(c2).degree)
    smallest_sep.append(sepa)
    


diff = np.array([np.min(smallest_sep[i]) for i in range(0, len(smallest_sep))])
potential_diff = diff[diff<0.001]

lit_index = np.squeeze(np.where(diff<0.001))

indx = np.array(([np.squeeze(np.where(smallest_sep[i] == np.min(smallest_sep[i]))) for i in range(0, len(smallest_sep))]))

F3D_index = indx[np.where(diff<0.001)]

matched_lit_m = m5007_1404[[2,4,7,38,43]]
matched_F3D_m = PNe_df["m 5007"].loc[[31,36,57,59,61]]

print(lit_index)


plt.scatter(matched_F3D_m, matched_lit_m)
plt.plot(np.arange(26.5,29), np.arange(26.5,29))
plt.xlabel("F3D sample")
plt.ylabel("Feldemier sample")

print("Center PNe m5007")
print(matched_lit_m)
print(matched_F3D_m)

print(f"Matched PNe indexes, for plotting: {F3D_index}")
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
plt.figure(figsize=(10,6))
centre_F3D = [27.67, 27.08, 27.39, 28.21, 27.39]
centre_lit = [26.79, 26.83, 26.98, 27.54, 27.68]

halo_lit   = [27.53, 27.55, 27.7 ]
halo_F3D   = [27.642509, 27.985619, 27.739837]
plt.scatter(halo_F3D, halo_lit, label="halo")
plt.scatter(centre_F3D, centre_lit, label="centre")
plt.plot(np.arange(26.5,29), np.arange(26.5,29))
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r"$\rm m_{5007, \ F3D}$", fontsize=20)
plt.ylabel(r"$\rm m_{5007, \ McMillan}$", fontsize=20)
plt.savefig("Plots/FCC219/FCC219_lit_comparison.png", bbox_inches='tight')