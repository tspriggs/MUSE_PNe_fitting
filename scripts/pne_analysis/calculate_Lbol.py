import gc
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from astropy.table import Table
from astropy.io import fits, ascii

from functions.file_handling import paths
from functions.PNe_functions import generate_mask
from functions.ppxf_gal_L import ppxf_L_tot

# Read in galaxy dataframe for list of galaxy names.
galaxy_df = pd.read_csv(f"exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
gal_names = galaxy_df.index.get_level_values(0) # get all galaxy names, has duplicates
all_locs = galaxy_df.index.get_level_values(1) # get all locs for gal_names  

# Setup of Argparse, where we can ask for individual galaxies by name, or, by default, all galaxies in galaxy_df will be used.
# dM type argument is for when the literature distance should be used instead.
# app argument is to switch to apperture summation, instead of FOV summation.
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--galaxy', action='store', nargs="+", type=str, required=False, default=gal_names, help="The name of the galaxy to be analysed. The default is to fit all the galaxies.")
my_parser.add_argument('--loc', action='store', nargs="+", type=str, required=False, default=all_locs, help="The pointing location, e.g. center, halo or middle. The default is to fit all the locations of each galaxy.")
my_parser.add_argument('--dM_type', action='store', type=str, required=False, default="PNLF", help="Provide the method of dM you want to calculate Lbol with: PNLF or lit. The default is PNLF.")
my_parser.add_argument('--app', action='store_true', default=False, help="For if you want to sum within an aperture or not, defaults to False.")

args = my_parser.parse_args()

galaxy_selection = args.galaxy 
locs = args.loc
dM_type = args.dM_type
calc_app = args.app


print(f"Proceeding to calculate Lbol for: {galaxy_selection} {locs}")

c = 299792458.0
# loc = "center"

with open("config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)


#############################################################################################################################
def run_Lbol(gal, loc, DIR_dict, dM, gal_params, z, dM_err_up, dM_err_lo, app_sum=False, custom_app=False, custom_app_params=[]):
    """First, mask out regions to be ignored, and or custom masks or appertures. Then integrate 2D IFU data into 1D spectrum.
    Then, run the code associated with calculating the bolometric luminosity of a galaxy's integrated spectrum.

    Parameters
    ----------
    gal : string
        galaxy name.

    loc : string
        galaxy pointing location (center, middle, halo).

    DIR_dict : dict
        dictionary containing useful directories for saving and loading plots or data.

    dM : float
        distance modulus used for Lbol determination.

    gal_params : dict
        [description]

    z : [float]
        Redshift

    dM_err_up : float
        upper error in dM.

    dM_err_lo : float
        lower error in dM.

    app_sum : bool, optional
        Check for if apperture summation should be used or not, by default False.

    custom_app : bool, optional
        Some galaxies need a custom apperture, this is again a check, by default False.

    custom_app_params : list, optional
        If a custom apperture is needed, then please provide the paramters for the apperture, by default an empty list, [].

    Returns
    -------
    [dict]
        Keys: "Lbol", "Lbol_err_up", "Lbol_err_lo", "mag_g", "mag_r", "mag_v", "sigma"
    """
    
    with fits.open(DIR_dict["RAW_DATA"]) as orig_hdulist:
        raw_data_cube = orig_hdulist[1].data * gal_params["F_corr"]
        h1 = orig_hdulist[1].header
        
    s = np.shape(raw_data_cube)
    Y, X = np.mgrid[:s[1], :s[2]]

    if app_sum == False:
        elip_mask = generate_mask([s[1], s[2]], gal_params["gal_mask"], mask_shape="ellipse")
        star_mask_sum = np.sum([generate_mask([s[1], s[2]], star_mask, mask_shape="circle") \
                            for star_mask in gal_params["star_mask"]],0).astype(bool)

        indx_mask = np.where(((np.isnan(raw_data_cube[40, :, :]) == False) & (elip_mask == False) & (star_mask_sum == False)) == True)

    elif app_sum == True:
        # create a 30 arcsecond aperture to extract and sum spectrum from
        if gal_params["centre"] == [1,1]:
            MUSE_x = h1["CRPIX1"]
            MUSE_y = h1["CRPIX2"]
        else:
            MUSE_x = gal_params["centre"][0]
            MUSE_y = gal_params["centre"][1]


        circ_params = [MUSE_x, MUSE_y, 150]
        im_shape = [s[1], s[2]]
        circ_mask = generate_mask(im_shape, circ_params, mask_shape="circle")

        star_mask_sum = np.sum([generate_mask(im_shape, star_mask, mask_shape="circle") \
                            for star_mask in gal_params["star_mask"]],0).astype(bool)

        indx_mask = np.where(((np.isnan(raw_data_cube[40, :, :]) == False) & (circ_mask == False) & (star_mask_sum == False)) == True)


    # For galaxies that require a custom apperture for summing the spectra.
    if custom_app == True:
        xe, ye, length, width, alpha = custom_app_params
        elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + \
        (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1

        star_mask_params = gal_params["star_mask"]
        star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc **
                                2 for xc, yc, rc in star_mask_params], 0).astype(bool)


        indx_mask = np.where(((np.isnan(raw_data_cube[40, :, :]) == False) & (elip_mask == True) & (star_mask_sum == False)) == True)



    print("Collapsing cube now....")    
    plt.figure()
    gal_lin = np.nansum(raw_data_cube[:, indx_mask[0], indx_mask[1]], 1)

    wavelength_data = h1['CRVAL3']+(np.arange(np.shape(raw_data_cube)[0])-h1['CRPIX3'])*h1['CD3_3']

    raw_data_cube = []
    elip_mask = []
    star_mask_sum = []
    total_mask = []
    indx_mask = []
    gc.collect()
    
    if app_sum == False:
        ascii.write([wavelength_data, gal_lin], f"exported_data/integrated_gal_spectra/{gal}_{loc}_FOV_integrated_spectra.txt", names=["Lambda", "Flux"], overwrite=True)

    print("Cube has been collapsed...")
    
    L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_params["velocity"], dist_mod=dM, dM_err=[dM_err_up, dM_err_lo])
    
    gal_lin = []
    gc.collect()
    plt.clf()
    plt.close()
    
    return L_bol

    


def calc_Lbol(gal, loc, dM_type, calc_app):
    galaxy_data = galaxy_info[f"{gal}_{loc}"]
    DIR_dict = paths(gal, loc)
    
    # based on dM_type, select which dM value set to use, PNLF or literature based.
    if dM_type == "PNLF":
        dM = galaxy_df.loc[(gal, "center"), "PNLF dM"]
        dM_err_up = galaxy_df.loc[(gal, "center"), "PNLF dM err up"]
        dM_err_lo = galaxy_df.loc[(gal, "center"), "PNLF dM err lo"]
    
    # else if literature distances are to be used, select the dM values that aren't nans (FCC161 is not in Blakeslee for example)
    elif dM_type == "lit":
        if np.isnan(galaxy_df.loc[(gal, "center"), "Bl dM"].value):
            dM = galaxy_df.loc[(gal, "center"), "lit dM"]
            dM_err_up = galaxy_df.loc[(gal, "center"), "lit dM err"]
            dM_err_lo = galaxy_df.loc[(gal, "center"), "lit dM err"]
        else:
            dM = galaxy_df.loc[(gal, "center"), "Bl dM"]
            dM_err_up = galaxy_df.loc[(gal, "center"), "Bl dM err"]
            dM_err_lo = galaxy_df.loc[(gal, "center"), "Bl dM err"]
        
    z = galaxy_data["velocity"] * 1e3 / c
    

    # If galaxy not in the prescribed list, calculate the Lbol of the spectra, across FOV, with masking applied.
    if (gal not in ["FCC119", "FCC143", "FCC176", "FCC255", "FCC301"]) or (loc in ["middle", "halo"]):

        Lbol_results = run_Lbol(gal, loc, DIR_dict, dM, galaxy_data, z, dM_err_up, dM_err_lo, app_sum=calc_app)
    
    # else if galaxy is in prescibed list, calculate the Lbol using pre-made apertures.
    elif (gal in ["FCC119", "FCC143", "FCC176", "FCC255", "FCC301"]) and (loc == "center"):
        gal_centre_pix = Table.read("exported_data/galaxy_centre_pix.dat", format="ascii")
        gal_indx = np.argwhere(gal_centre_pix["Galaxy"] == gal)
        gal_x = gal_centre_pix[gal_indx]["x_pix"]
        gal_y = gal_centre_pix[gal_indx]["y_pix"]
        if gal == "FCC119":
            app_params = [galaxy_data["centre"][0], galaxy_data["centre"][1], 100, 100, 1]
        elif gal == "FCC143":
            app_params = [gal_x, gal_y, 260, 140, 2.1]
        elif gal == "FCC176":
            app_params = [gal_x, gal_y, 400, 400, 1.0]
        elif gal == "FCC255":
            app_params = [gal_x, gal_y, 400, 200, -0.1]
        elif gal == "FCC301":
            app_params = [gal_x, gal_y, 270, 200, -0.3]

        Lbol_results = run_Lbol(gal, loc, DIR_dict, dM, galaxy_data, z, dM_err_up, dM_err_lo, custom_app=True, custom_app_params=app_params)



    return Lbol_results



for i, (gal, loc) in enumerate(zip(galaxy_selection, locs)):
    print(f"\n Calculating Lbol for {gal} {loc}")
    DIR_dict = paths(gal, loc)
    Lbol_results = calc_Lbol(gal, loc, dM_type, calc_app)
    L_bol_df = pd.DataFrame(data=[[Lbol_results["Lbol"], Lbol_results["Lbol_err_up"]-Lbol_results["Lbol"], Lbol_results["Lbol"]-Lbol_results["Lbol_err_lo"],
                                    Lbol_results["mag_v"], Lbol_results["mag_r"], Lbol_results["sigma"]]],
                                    columns=("Lbol", "Lbol_err_up", "Lbol_err_lo", "mag_v", "mag_r", "sigma"))

    L_bol_df.to_csv(DIR_dict["EXPORT_DIR"]+"_Lbol_df.csv")
    print(f"\n{gal} {loc} completed! \n")


