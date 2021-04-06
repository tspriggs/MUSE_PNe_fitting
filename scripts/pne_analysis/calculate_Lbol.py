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


with open("config/galaxy_info.yaml", "r") as yaml_gal_info_file:
    yaml_gal_info = yaml.load(yaml_gal_info_file, Loader=yaml.FullLoader)

galaxy_names = np.unique([gal.split("_")[0] for gal in yaml_gal_info],0)

# take the first entry in the 
if len(list(yaml_gal_info.keys())[0].split("_")) > 1:
    galaxy_locs = np.unique([gal.split("_")[1] for gal in yaml_gal_info],0)
else:
    galaxy_locs = [""]*len(galaxy_names)


# Setup of Argparse, where we can ask for individual galaxies by name, or, by default, all galaxies in galaxy_df will be used.
# dM type argument is for when the literature distance should be used instead.
# app argument is to switch to apperture summation, instead of FOV summation.
my_parser = argparse.ArgumentParser()
my_parser.add_argument('--galaxy', action='store', nargs="+", type=str, required=False, default=galaxy_names, 
                        help="The name of the galaxy to be analysed. The default is to fit all the galaxies.")
my_parser.add_argument('--loc', action='store', nargs="+", type=str, required=False, default=galaxy_locs, 
                        help="The pointing location, e.g. center, halo or middle. The default is to fit all the locations of each galaxy.")
my_parser.add_argument('--app', action='store_true', default=False, help="For if you want to sum within an aperture or not, defaults to False.")

args = my_parser.parse_args()

galaxy_selection = args.galaxy 
locs = args.loc
calc_app = args.app


print(f"Proceeding to calculate Lbol for: {galaxy_selection} {locs}")

c = 299792458.0

with open("config/galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)


#############################################################################################################################
def run_Lbol(DIR_dict, dM, gal_params, z, dM_err_up, dM_err_lo, app_sum=False, custom_app=False, custom_app_params=[]):
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
        dictionary of galaxy values, from the galaxy_info.yaml config file.

    z : float
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
    dict
        Keys: "Lbol", "Lbol_err_up", "Lbol_err_lo", "mag_g", "mag_r", "mag_v", "sigma"
    """
    
    with fits.open(DIR_dict["RAW_DATA"]) as orig_hdulist:
        raw_data_cube = orig_hdulist[1].data * gal_params["F_corr"]
        h1 = orig_hdulist[1].header
        
    s = np.shape(raw_data_cube)
    Y, X = np.mgrid[:s[1], :s[2]]
    
    # The equation for calclating the wavelength array, from the header of the raw/reduced MUSE cube.
    # wavelength_data = h1['CRVAL3']+(np.arange(np.shape(raw_data_cube)[0])-h1['CRPIX3'])*h1['CD3_3']

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


    raw_data_cube = []
    elip_mask = []
    star_mask_sum = []
    indx_mask = []
    gc.collect()
    
    
    print("Cube has been collapsed...")
    try:
        L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_params["velocity"], dist_mod=dM, dM_err=[dM_err_up, dM_err_lo])
    except AssertionError as error:
        print(error)
        print("Please include the emiles stellar template library in the working directory: MUSE_PNe_fitting/emiles/")
        print("Please See documentation for download link to emiles template library.")
        L_bol = {"Lbol":1, "Lbol_err_up":1, "Lbol_err_lo":1, "mag_g":1, "mag_r":1, "mag_v":1, "sigma":1}


    gal_lin = []
    gc.collect()
    plt.clf()
    plt.close()
    
    return L_bol

    


def calc_Lbol(gal, loc, calc_app=False):
    """Simple function made for running the Lbol calculation, across a number of galaxies, matching criteria and wether or not to use appertures.

    Parameters
    ----------
    gal : str
        Galaxy name, given as FCC000 or NGC0000
    loc : str
        Location of the pointing, used for Fornax3D survey as some galaxies are observed across a center, middle and halo pointings.
    calc_app : bool
        Boolean flag to decide if appertures should be used, or not. Defaults to False.

    Returns
    -------
    dict
        Contains all the information needed from using pPXF to calculate the luminosity, with errors, for a given galaxy.
        Keys: "Lbol", "Lbol_err_up", "Lbol_err_lo", "mag_g", "mag_r", "mag_v", "sigma"
    """
    galaxy_data = galaxy_info[f"{gal}_{loc}"]
    DIR_dict = paths(gal, loc)
    
    # use the PNLF derived dM from running the PNe fitting script.
    dM = PNe_results_df['PNLF dM'][0]
    dM_err_up = PNe_results_df['PNLF dM err up'][0]
    dM_err_lo = PNe_results_df['PNLF dM err lo'][0]
    
    #  Alternatively, you can comment this section out and input your own values and errors for dM.
    #dM = your_dM_value_here
    #dM_err_up = your_dM_upper_error_here
    #dM_err_lo = your_dM_lower_error_here


    z = galaxy_data["velocity"] * 1e3 / c
    
    # Inlcude below examples of how I have run the Lbol routines on Fornax3D data.
    # If galaxy not in the below list, or a middle or halo pointing
    # calculate the bolometric luminosity of the spectra, across FOV, with masking applied.
    if (gal not in ["FCC119", "FCC143", "FCC176", "FCC255", "FCC301"]) or (loc in ["middle", "halo"]):

        Lbol_results = run_Lbol(DIR_dict, dM, galaxy_data, z, dM_err_up, dM_err_lo, app_sum=calc_app)
    
    # else if galaxy is in below list, and is a central pointing, calculate the bolometric luminosity using pre-made,
    # defined apertures, using the app_params argument.
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
    PNe_results_df = pd.read_csv(f"exported_data/{gal}/{gal}{loc}_PN_result_df.csv")
    Lbol_results = calc_Lbol(gal, loc, calc_app)
    L_bol_df = pd.DataFrame(data=[[Lbol_results["Lbol"], Lbol_results["Lbol_err_up"]-Lbol_results["Lbol"], Lbol_results["Lbol"]-Lbol_results["Lbol_err_lo"],
                                    Lbol_results["mag_v"], Lbol_results["mag_r"], Lbol_results["sigma"]]],
                                    columns=("Lbol", "Lbol_err_up", "Lbol_err_lo", "mag_v", "mag_r", "sigma"))

    L_bol_df.to_csv(DIR_dict["EXPORT_DIR"]+"_Lbol_df.csv")
    print(f"\n{gal} {loc} completed! \n")


