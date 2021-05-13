import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import yaml

from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import  Ellipse, Circle

from functions.file_handling import reconstructed_image, paths, open_data


# Setup for argparse
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument("--loc",    action="store", type=str, required=False)
my_parser.add_argument("--ann", action="store_true", default=False)
my_parser.add_argument("--save", action="store_true", default=False)
my_parser.add_argument("--show", action="store_true", default=False)
my_parser.add_argument("--matched", action="store", nargs="+" , type=int, default=[])
args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy
loc = args.loc              # galaxy name, format of FCC000
ann = args.ann              # Annotate sources with numbers
save_plot = args.save       # save plots
matched = args.matched
show = args.show

 
def open_relevant_data(gal_name, loc):
    """Open up the relevant dataframe of PNe for a given galaxy and location, along with A/rN map, 
    [x,y] coordinates and the wcs objs.

    Parameters
    ----------
    gal_name : str
        Galaxy Name (format of FCC000 as example)
    loc : str
        pointing location: center, middle or halo

    Returns
    -------
    [type]
        DIR_dict, PNe_df, A_rN_plot_shape, x_y_list, galaxy_info, x_data, y_data, wcs_obj
    """
    DIR_dict = paths(gal_name, loc) # Get directories
    res_cube, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy_name, loc, DIR_dict) # open corresponding data for galaxy and loc
    
    # Read in PN dataframe, and using the object column, plot signage of PNe, SNR, HII, impostor and over-luminous.
    PNe_df = pd.read_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")
    x_y_list = np.load(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list.npy")
    A_rN_plot = np.load(DIR_dict["EXPORT_DIR"]+"_A_rN.npy")

    A_rN_plot_shape = A_rN_plot.reshape(y_data, x_data)
    
    # Get WCS coordinates
    with fits.open(DIR_dict["RAW_DATA"]) as hdu_wcs:
        hdr_wcs = hdu_wcs[1].header
        wcs_obj = WCS(hdr_wcs, naxis=2)

    return DIR_dict, PNe_df, A_rN_plot_shape, x_y_list, galaxy_info, x_data, y_data, wcs_obj


if galaxy_name not in ["FCC083", "FCC147", "FCC148", "FCC161", "FCC184", "FCC190", "FCC193", "FCC219", "FCC310",]:
        
    DIR_dict, PNe_df, A_rN_plot_shape, x_y_list, galaxy_info, x_data, y_data, wcs_obj = open_relevant_data(galaxy_name, loc)

    # set the project of the figure to the RA and DEC of the raw MUSE image
    # then plot the A/rN map for better source contrast against background
    plt.figure(figsize=(12,12))
    plt.axes(projection=wcs_obj)
    plt.imshow(A_rN_plot_shape, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8)

    # store the RA and DEC axis information
    ax = plt.gca()
    RA = ax.coords[0]
    DEC = ax.coords[1]
    # reduce number of ticks to stop crowding of the tick labels
    RA.set_ticks(number=4)

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

    # For FCC276, the xlim and ylim need to be set specifically, as we only have the central pointing.
    if galaxy_name=="FCC276":
        if loc=="center":
            plt.xlim(315,)
            plt.ylim(0,435)

    # elliptical galaxy mask creation, using coordinates from yaml config
    elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=True, alpha=0.5, hatch="\\", color="k", ls="--")
    ax.add_artist(elip_gal)

    # create star mask to mask any stars in the FOV, using coordinates from yaml config
    for star in galaxy_info["star_mask"]:
        ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=True, alpha=0.5, hatch="\\", color="k", ls="--"))

    # add circles to highlight the PNe within the image that are a part of the catalogue. Colour code the PNe of each location
    # C0 for center PNe (cyan/default)
    # red for middle PNe
    # green for halo PNe
    PN_x_y_list = x_y_list[PNe_df.loc[PNe_df["ID"]=="PN"].index.values]
    if loc == "center":
        ax.scatter(PN_x_y_list[:,0], PN_x_y_list[:,1], facecolor="None", edgecolor="C0", lw=1.2, s=250, label="Centre PNe")
    elif loc == "middle":
        ax.scatter(PN_x_y_list[:,0], PN_x_y_list[:,1], facecolor="None", edgecolor="r", lw=1.2, s=250, label="Middle PNe")
    elif loc == "halo":
        ax.scatter(PN_x_y_list[:,0], PN_x_y_list[:,1], facecolor="None", edgecolor="g", lw=1.2, s=250, label="Halo PNe")

    # If any objects ID set to Over-luminous, add a specific label for them and add it to the legend.
    if len(PNe_df.loc[PNe_df["ID"]=="OvLu"].index.values) >=1:
        OvLu_x_y_list = x_y_list[PNe_df.loc[PNe_df["ID"]=="OvLu"].index.values]
        ax.scatter(OvLu_x_y_list[:,0], OvLu_x_y_list[:,1], marker="s", facecolor="None", edgecolor="k", lw=1.2, s=250, label="Over-luminous object")

    # for any literature matching PNe, add a specific type of highlight for them and add to legend.
    if len(matched) > 0:
        matched_x_y_list = x_y_list[matched]
        ax.scatter(matched_x_y_list[:,0], matched_x_y_list[:,1], marker="s", facecolor="None", edgecolor="blue", lw=1.2, s=350, label="Literature-matched PNe")

    plt.legend(loc=2, fontsize=15, labelspacing=1.0)

    # If the ann flag is set to true, annotate the PNe with their ID number (0...n)
    for i, item in enumerate(x_y_list):
        if ann == True:
            ax.annotate(i, (item[0]+6, item[1]-2), color="black", size=15)

    # Plot arrows
    # dictionary for galaxies that need specific N E arrow locations "gal_name":[x,y]
    gal_arrow_coords = {"FCC083":[355,300], "FCC143":[50,40], "FCC147":[270,25], "FCC148":[400,350], "FCC153":[400,350],
                        "FCC170":[300,40], "FCC176":[50,40], "FCC190":[450, 430],"FCC193":[50,280], "FCC249":[300,250], 
                        "FCC276":[720,350], "FCC310":[390,450]}

    # If plotting the center pointing, and galaxy name is in the above dictionary, use the appropriate pre-determined coordinates for the North and East arrow.
    # Otherwise, use the default location set out below.
    if loc=="center":
        if galaxy_name in gal_arrow_coords:
            N_E_x = gal_arrow_coords[galaxy_name][0]
            N_E_y = gal_arrow_coords[galaxy_name][1]
        elif galaxy_name not in gal_arrow_coords:
            N_E_x = 400
            N_E_y = 350

        plt.arrow(N_E_x, N_E_y, 0,30, head_width=5, width=0.5, color="k")
        plt.annotate("N", xy=(N_E_x-5, N_E_y+40), fontsize=25)
        plt.arrow(N_E_x, N_E_y, -20,0, head_width=5, width=0.5, color="k")
        plt.annotate("E", xy=(N_E_x-38, N_E_y-5), fontsize=25)

    # produce a white-light, collpased image of the galaxy.
    data, wave, hdr = reconstructed_image(galaxy_name, loc)
    # calculate the magnitude value of each pixel from the white-light image.
    mag_data = -2.5*np.log10(data*1e-20) -21.10

    # Plot out the contours of the magnitudes calculated above, with range of 15 to 22 mags for all but FCC310 (which will use 15 to 21 instead).
    if galaxy_name != "FCC310":
        CS = plt.contour(mag_data, data=mag_data, levels=range(15,22,1), colors="black", alpha=0.6)
    else:
        CS = plt.contour(mag_data, data=mag_data, levels=range(15,21,1), colors="black", alpha=0.6)

    # manually decided coordinates for the locations of contour labels for the level's value.
    gal_contour_man_locs = {"FCC083":[(180,130), (140,100)],
                            "FCC119":[(170,175), (175,135)],
                            "FCC143":[(230,220),(210,190), (210, 170)],
                            "FCC147":[(390,130), (465,125), (400,50)],
                            "FCC148":[(170,205), (140,190), (140,160)],
                            "FCC153":[(150,230), (230,200),(220,190)],
                            "FCC161":[(220,205), (225,180),(250,130)],
                            "FCC167":[(210,260), (190,90), (350,230)],
                            "FCC170":[(130,150), (100,140),(90,160)],
                            "FCC177":[(200,170),(175,175)],
                            "FCC182":[(190,230), (250,250)],
                            "FCC184":[(150,145),(160,170)],
                            "FCC190":[(300,300),(250,325)],
                            "FCC193":[(210,350),(240,325), (270,350)],
                            "FCC219":[(240,70),(230,200), (200,270)],
                            "FCC249":[(240,70),(260,70), (290,60)],
                            "FCC255":[(255,315),(255,280), (290,300)],
                            "FCC276":[(630,230),(610,260), (560,340)],
                            "FCC277":[(300,230),(240,190), (300,180), (310,150)],
                            "FCC301":[(220,190),(240,190), (260,155)],
                            "FCC310":[(190,250),(210,260)]
                            }

    # Using the above dictionary, place labels on the contours that report the magnitude value of the contour.
    if (galaxy_name in gal_contour_man_locs) & (loc == "center"):
        plt.clabel(CS, fontsize=15, inline=1, fmt='%1.1f', manual=gal_contour_man_locs[galaxy_name])
    # if plotting the contours on the middle or halo pointings, use the default given locations, as unlikely to have source overlap.
    elif loc in ["middle", "halo"]:
        plt.clabel(CS, fontsize=15, inline=True, inline_spacing=8, fmt='%1.1f',)
    else:
        pass

    # if the show flag is set to True, then show the plots.
    if show == True:
        plt.show()

    # if the save_plot flag is set to True, then save the plot in both png and pdf format.
    if save_plot == True:
        plt.savefig(DIR_dict["PLOT_DIR"]+"_A_rN_circled.png", bbox_inches='tight', dpi=300)
        plt.savefig(DIR_dict["PLOT_DIR"]+"_A_rN_circled.pdf", bbox_inches='tight', dpi=300)


################################################################################################################

# If the galaxy name is instead within this list of galaxies, then carry out the plotting procedure, producing a single figure with multiple pointings in.
elif galaxy_name in ["FCC083", "FCC147", "FCC148", "FCC161", "FCC184", "FCC190", "FCC193", "FCC219", "FCC310",]:
    # open up the config file for the galaxy information, used to check if halo and middle data exists
    with open("config/galaxy_info.yaml", "r") as yaml_data:
        yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

    # check if there is a halo pointing for the galaxy, if yes, then change the halo_check flag to True.
    halo_check=False
    if f"{galaxy_name}_halo" in yaml_info:
        halo_check = True

    # check if there is a middle pointing for the galaxy, if yes, then change the mid_check flag to True.
    mid_check=False
    if f"{galaxy_name}_middle" in yaml_info:
        mid_check = True    

    # Open up the relevant data for the central pointing, as everything will be plotted in respect to this FOV / data.
    cen_DIR_dict, cen_PNe_df, cen_map, cen_x_y_list, cen_galaxy_info, cen_x_data, cen_y_data, wcs_obj = open_relevant_data(galaxy_name, "center")

    # set the project of the figure to the RA and DEC of the raw MUSE image
    # then plot the A/rN map for better source contrast against background
    plt.figure(figsize=(12,12))
    plt.axes(projection=wcs_obj)
    plt.imshow(cen_map, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8, zorder=2)

    # store the RA and DEC axis information
    ax = plt.gca()
    RA = ax.coords[0]
    DEC = ax.coords[1]

    # reduce number of ticks to stop crowding of the tick labels
    RA.set_ticks(number=4)
    
    # colorbar settings
    cb=plt.colorbar(fraction=0.03, pad=0.04)
    cb.set_label("A/rN",fontsize=30)
    cb.ax.tick_params(labelsize=22)
    
    # Axis labels and fontsize
    plt.xlabel("RA (J2000)", fontsize=30)
    plt.ylabel("DEC (J2000)", fontsize=30)
    plt.tick_params(labelsize = 22)
    
    # gal and star mask setup
    Y, X = np.mgrid[:cen_y_data, :cen_x_data]
    xe, ye, length, width, alpha = cen_galaxy_info["gal_mask"]

    # elliptical galaxy mask creation, using coordinates from yaml config
    elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=True, alpha=0.5, hatch="\\", color="k", ls="--", zorder=5)
    ax.add_artist(elip_gal)

    # create star mask to mask any stars in the FOV, using coordinates from yaml config
    for star in cen_galaxy_info["star_mask"]:
        ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=True, alpha=0.5, hatch="\\", color="k", ls="--", zorder=5))

    # add circles to highlight the PNe within the image that are a part of the catalogue. Colour code the PNe of each location
    # C0 for center PNe (cyan/default)
    cen_PN_x_y_list = cen_x_y_list[cen_PNe_df.loc[cen_PNe_df["ID"]=="PN"].index.values]
    ax.scatter(cen_PN_x_y_list[:,0], cen_PN_x_y_list[:,1], facecolor="None", edgecolor="C0", lw=1., s=200, label="Centre PNe", zorder=4)

    # If there is middle data, then open the relevant data and plot the A/rN map in the same figure as the central pointing
    if mid_check == True:
        mid_DIR_dict, mid_PNe_df, mid_map, mid_x_y_list, mid_info, mid_x_data, mid_y_data, mid_wcs_obj = open_relevant_data(galaxy_name, "middle")
        plt.imshow(mid_map, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8, zorder=0) # Plot A/rN map
        mid_PN_x_y_list = mid_x_y_list[mid_PNe_df.loc[mid_PNe_df["ID"]=="PN"].index.values] # indexing for only PNe
        ax.scatter(mid_PN_x_y_list[:,0], mid_PN_x_y_list[:,1], facecolor="None", edgecolor="r", lw=1., s=200, label="Middle PNe", zorder=1)
        mid_data, wave, hdr = reconstructed_image(galaxy_name, "middle") # white-light image
        mid_mag_data = -2.5*np.log10(mid_data* cen_galaxy_info["F_corr"]*1e-20) -21.10  # magnitude calculation

    # If there is halo data, then open the relevant data and plot the A/rN map in the same figure as the central pointing
    if halo_check == True:
        halo_DIR_dict, halo_PNe_df, halo_map, halo_x_y_list, halo_info, halo_x_data, halo_y_data, halo_wcs_obj = open_relevant_data(galaxy_name, "halo")
        plt.imshow(halo_map, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8, zorder=0) # Plot A/rN map
        halo_PN_x_y_list = halo_x_y_list[halo_PNe_df.loc[halo_PNe_df["ID"]=="PN"].index.values] # indexing for only PNe
        ax.scatter(halo_PN_x_y_list[:,0], halo_PN_x_y_list[:,1], facecolor="None", edgecolor="g", lw=1., s=200, label="Halo PNe", zorder=1)
        halo_data, wave, hdr = reconstructed_image(galaxy_name, "halo") # white-light image
        halo_mag_data = -2.5*np.log10(halo_data* cen_galaxy_info["F_corr"]*1e-20) -21.10 # magnitude calculation

    # If any objects ID set to Over-luminous, add a specific label for them and add it to the legend.
    if len(cen_PNe_df.loc[cen_PNe_df["ID"]=="OvLu"].index.values) >=1:
        OvLu_x_y_list = cen_x_y_list[cen_PNe_df.loc[cen_PNe_df["ID"]=="OvLu"].index.values]
        ax.scatter(OvLu_x_y_list[:,0], OvLu_x_y_list[:,1], marker="s", facecolor="None", edgecolor="k", lw=1.2, s=250, label="Over-luminous object")

    if galaxy_name in ["FCC161", "FCC193"]:
        plt.legend(loc="upper right", fontsize=15, labelspacing=1.0)
    elif galaxy_name == "FCC184":
        plt.legend(loc="lower right", fontsize=15, labelspacing=1.0)
    else:
        plt.legend(loc=2, fontsize=15, labelspacing=1.0)

    # produce a white-light, collpased image of the galaxy.
    cen_data, wave, hdr = reconstructed_image(galaxy_name, "center")
    # calculate the magnitude value of each pixel from the white-light image.
    cen_mag_data = -2.5*np.log10(cen_data*cen_galaxy_info["F_corr"]*1e-20) -21.10

    # Plot out the contours of the magnitudes calculated above, with range of 15 to 22 mags for all but FCC310 (which will use 15 to 21 instead).
    if galaxy_name != "FCC310":
        cen_CS = plt.contour(cen_mag_data, data=cen_mag_data, levels=range(15,22,1), colors="black", alpha=0.6, zorder=3)
        if halo_check == True:
            halo_CS = plt.contour(halo_mag_data, data=halo_mag_data, levels=range(15,22,1), colors="black", alpha=0.4,)
        if mid_check == True:
            mid_CS = plt.contour(mid_mag_data, data=mid_mag_data, levels=range(15,22,1), colors="black", alpha=0.4)

    else:
        cen_CS = plt.contour(cen_mag_data, data=cen_mag_data, levels=range(15,21,1), colors="black", alpha=0.6, zorder=3)
        if halo_check == True:
            halo_CS = plt.contour(halo_mag_data, data=halo_mag_data, levels=range(15,21,1), colors="black", alpha=0.4)
        if mid_check == True:
            mid_CS = plt.contour(mid_mag_data, data=mid_mag_data, levels=range(15,21,1), colors="black", alpha=0.4)
            
    # manually decided coordinates for the locations of contour labels for the level's value.
    gal_contour_man_locs = {"FCC083":[(180,130), (140,100)],
                            "FCC119":[(170,175), (175,135)],
                            "FCC143":[(230,220),(210,190), (210, 170)],
                            "FCC147":[(390,130), (465,125), (400,50)],
                            "FCC148":[(170,205), (140,190), (140,160)],
                            "FCC153":[(150,230), (230,200),(220,190)],
                            "FCC161":[(220,205), (225,180),(250,130)],
                            "FCC167":[(210,260), (190,90), (350,230)],
                            "FCC170":[(130,150), (100,140),(90,160)],
                            "FCC177":[(200,170),(175,175)],
                            "FCC182":[(190,230), (250,250)],
                            "FCC184":[(160,145),(160,170)],
                            "FCC190":[(300,300),(250,325)],
                            "FCC193":[(210,350),(240,325), (270,350)],
                            "FCC219":[(240,70),(230,200), (200,270)],
                            "FCC249":[(240,70),(260,70), (290,60)],
                            "FCC255":[(255,315),(255,280), (290,300)],
                            "FCC276":[(630,230),(610,260), (560,340)],
                            "FCC277":[(300,230),(240,190), (300,180), (310,150)],
                            "FCC301":[(220,190),(240,190), (260,155)],
                            "FCC310":[(190,250),(210,260)]
                            }

    # Using the above dictionary, place labels on the contours that report the magnitude value of the contour.
    if (galaxy_name in gal_contour_man_locs) & (loc == "center"):
        plt.clabel(cen_CS, fontsize=12, inline=1, fmt='%1.1f', manual=gal_contour_man_locs[galaxy_name])
    else:
        pass
    
    # if the show flag is set to True, then show the plots.
    if show == True:
        plt.show()
    
    # if the save_plot flag is set to True, then save the plot in both png and pdf format.
    if save_plot == True:
        plt.savefig(cen_DIR_dict["PLOT_DIR"]+"_A_rN_circled.png", bbox_inches='tight', dpi=300)
        plt.savefig(cen_DIR_dict["PLOT_DIR"]+"_A_rN_circled.pdf", bbox_inches='tight', dpi=300)
