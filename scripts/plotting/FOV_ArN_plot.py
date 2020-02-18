import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

from astropy.io import ascii, fits
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle, Ellipse, Circle

# Setup for argparse
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument("--loc",    action="store", type=str, required=True)
my_parser.add_argument("--save", action="store_true", default=False)
args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy   # galaxy name, format of FCC000
loc = args.loc              # MUSE pointing loc: center, middle, halo
save_plot = args.save       # save plots

from functions.file_handling import paths, open_data, prep_impostor_files

DIR_dict = paths(galaxy_name, loc)

res_data, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy_name, loc, DIR_dict)


# Read in PN dataframe, and using the object column, plot signage of PNe, SNR, HII, impostor and over-luminous.
PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")
x_y_list = np.load(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list.npy")

A_rN_plot = np.load(DIR_dict["EXPORT_DIR"]+"_A_rN_cen.npy")
A_rN_plot_shape = A_rN_plot.reshape(y_data, x_data)

# Get WCS coordinates
with fits.open(DIR_dict["RAW_DAT"]) as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)

plt.figure(figsize=(15,15))
plt.axes(projection=wcs_obj)
plt.imshow(A_rN_plot_shape, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8)
ax = plt.gca()
RA = ax.coords[0]
DEC = ax.coords[1]

# colorbar settings
cb=plt.colorbar(fraction=0.0455, pad=0.04)
cb.set_label("A/rN",fontsize=30)
cb.ax.tick_params(labelsize=22)

# Axis labels and fontsize
plt.xlabel("RA (J2000)", fontsize=30)
plt.ylabel("DEC (J2000)", fontsize=30)
plt.tick_params(labelsize = 22)

# gal and star mask setup
Y, X = np.mgrid[:y_data, :x_data]
xe, ye, length, width, alpha = galaxy_info["gal_mask"]

# if (galaxy_name=="FCC219") & (loc=="center"):
#     plt.ylim(0,440)
#     plt.xlim(0,440);
# if (galaxy_name=="FCC219") & (loc=="halo"):
#     plt.ylim(350,)
# #     plt.xlim(440,);
# elif galaxy_name=="FCC193":
#     plt.ylim(250,)
#     plt.xlim(0,350)
# elif galaxy_name=="FCC161":
#     plt.xlim(0,450)
# elif galaxy_name=="FCC147":
#     plt.xlim(230,)
#     plt.ylim(0,320)
# elif galaxy_name=="FCC083":
#     plt.xlim(0,370)
#     plt.ylim(0,370)
# elif galaxy_name=="FCC310":
#     plt.xlim(0,410)
#     plt.ylim(100,)
# elif galaxy_name=="FCC276":
#     plt.xlim(310,)
# elif galaxy_name=="FCC184":
#     plt.xlim(0,450)
#     plt.ylim(0,450)

elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=False, color="grey", ls="--")
ax.add_artist(elip_gal)

for star in galaxy_info["star_mask"]:
    ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=False, color="grey", ls="--"))


for i, item in enumerate(x_y_list):
    if PNe_df.iloc[i].ID == "PN": # If PN, use black circle
        ax = plt.gca()
        circ = plt.Circle((item[0], item[1]), 5, color="black", lw=1.2, fill=False)
        ax.add_artist(circ)
        ax.annotate(i, (item[0]+6, item[1]-2), color="black", size=15)
    if PNe_df.iloc[i].ID == "OvLu":
        ax.plt.gca()
        rect = plt.Rectangle((item[0], item[1]), 7,7, color="black", fill=False)
        ax.add_artist(rect)

plt.legend((circ, rect), ("PNe", "Over-luminous"))

# Plot arrows
plt.arrow(400,380, 0,30, head_width=5, width=0.5, color="k")
plt.annotate("N", xy=(395, 420), fontsize=25)
plt.arrow(400,380, -20,0, head_width=5, width=0.5, color="k")
plt.annotate("E", xy=(360, 375), fontsize=25)

# plt.show()

if save_plot == True:
    plt.savefig(DIR_dict["PLOT_DIR"]+"_A_rN_circled.png", bbox_inches='tight')
    plt.savefig(DIR_dict["PLOT_DIR"]+"_A_rN_circled.pdf", bbox_inches='tight')


