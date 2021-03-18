import numpy as np
import pandas as pd
import yaml
import os
import argparse

from functions.MUSE_Models import Moffat
from functions.file_handling import paths, open_data
from functions.PNLF import reconstructed_image 
from functions.PNe_functions import robust_sigma


# Read in galaxy dataframe for list of galaxy names.
with open("config/galaxy_info.yaml", "r") as yaml_gal_info_file:
    yaml_gal_info = yaml.load(yaml_gal_info_file, Loader=yaml.FullLoader)

galaxy_names = np.unique([gal.split("_")[0] for gal in yaml_gal_info],0)


# Setup of Argparse, where we can ask for individual galaxies by name, or, by default, all galaxies in galaxy_info.yaml will be used.
# dM type argument is for when the literature distance should be used instead.
# app argument is to switch to apperture summation, instead of FOV summation.
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', nargs="+", type=str, required=False, default=galaxy_names)

args = my_parser.parse_args()

galaxy_selection = args.galaxy 

def prep_completness_data(galaxy, loc, DIR_dict, galaxy_info, test=False):

    #load up data
    res_data, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy, loc, DIR_dict)
    
    res_data = res_data*galaxy_info["F_corr"]

    if res_data.ndim > 2:
        res_data_list = res_data.reshape(len(wavelength), x_data*y_data)
        res_data_list = np.swapaxes(res_data_list, 1, 0)
    else:
        res_data_list = res_data

    galaxy_image, rec_wave, rec_hdr = reconstructed_image(galaxy, loc)
    image = galaxy_image * galaxy_info["F_corr"]
    # image = galaxy_image.reshape(y_data, x_data)

    c = 299792458.0
    z = galaxy_info["velocity"]*1e3 / c

    # rN = np.array([np.nanstd(res_data_list[i]) for i in range(len(res_data_list))])
    rN = np.array([robust_sigma(res_data_list[i]) for i in range(len(res_data_list))])
    Noise_map = rN.reshape(y_data, x_data)

    Y, X = np.mgrid[:y_data, :x_data]

    # for some galaxies, setout a region to analyse, before masking out bad areas
    if test == True:
        if galaxy == "FCC143":
            xe, ye, length, width, alpha = [200.9-15, 216.9-20, 300, 250, 2.1]
        if galaxy == "FCC177":
            xe, ye, length, width, alpha = [210., 225.0, 175, 400, 1.5]

        elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + \
                    (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1
        star_mask_params = galaxy_info["star_mask"]
        star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc **
                                    2 for xc, yc, rc in star_mask_params], 0).astype(bool)
        total_mask = ((np.isnan(galaxy_image) == False) & (elip_mask == True) & (star_mask_sum == False))

        image[~total_mask] = 0.0
        Noise_map[~total_mask] = 0.0

    # mask out regions
    xe, ye, length, width, alpha = galaxy_info["gal_mask"]
    
    elip_mask_gal = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + \
        (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1
    star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc, yc,
                            rc in galaxy_info["star_mask"]], 0).astype(bool)  # galaxy_data["star_mask"]
    mask_indx = np.array(np.where((elip_mask_gal+star_mask_sum) == True))
    # End of masking

    # Noise data: set masked regions to 0.0
    Noise_map[mask_indx[0], mask_indx[1]] = 0.0
    # image data: set masked regions to 0.0
    image[mask_indx[0], mask_indx[1]] = 0.0
    # iamge data: set any values that are less than 0, equal to 0.
    Noise_map[Noise_map < 0.0] = 0.0
    image[image < 0.0] = 0.0

    return image, Noise_map



def calc_completeness(image, Noise_map, mag, params, peak, n_pixels):
    # Construct the PNe FOV coordinate grid for use when fitting PNe.
    coordinates = [(n, m) for n in range(n_pixels) for m in range(n_pixels)]
    x_fit = np.array([item[0] for item in coordinates])
    y_fit = np.array([item[1] for item in coordinates])

    total_flux = 10**((mag + 13.74) / -2.5)  # Ciardullo's flux calibration

    flux = total_flux*1.0e20

    init_FWHM = params['FWHM']
    init_beta = params['beta']

    sum_init = np.sum(Moffat(1, init_FWHM, init_beta,
                             n_pixels/2, n_pixels/2, x_fit, y_fit))
    input_moff_A = flux / sum_init

    # Make moffat models = F_5007 (x,y)
    Moffat_models = np.array([Moffat(moff_A, init_FWHM, init_beta,
                                     n_pixels/2., n_pixels/2., x_fit, y_fit) for moff_A in input_moff_A])

    # turn moffat model into list of Gaussian amplitudes (A_5007 (x,y))
    Amp_x_y = ((Moffat_models) / (np.sqrt(2*np.pi) * (params["LSF"]/2.35482)))
    # make a list of the maximum amplitude per magnitude
    max_1D_A = np.array([np.max(A) for A in Amp_x_y])

    completeness_ratio = np.zeros(len(mag))

    for i, amp in enumerate(max_1D_A):
        completeness_ratio[i] = (np.nansum(image[((amp / Noise_map) >= peak)]) / np.nansum(image)).astype(np.float128)

    completeness_ratio[np.where(completeness_ratio==np.min(completeness_ratio))] = 0.0

    return np.asarray(completeness_ratio)


step = 0.001
m_5007 = np.arange(26, 31, step)

if os.path.isfile("exported_data/completeness_ratio_df.csv") is False:
    comp_df = pd.DataFrame(columns=("Galaxy", "FWHM", "beta", "LSF"))
    comp_df["Galaxy"] = galaxy_names
    comp_df.set_index("Galaxy", inplace=True)

else:
    comp_df = pd.read_csv("exported_data/completeness_ratio_df.csv", index_col="Galaxy")

for i, gal in enumerate(galaxy_selection):
    for loc in ["center", "halo", "middle"]:
        if f"{gal}_{loc}" in [*yaml_gal_info]:
            galaxy_data = yaml_gal_info[f"{gal}_{loc}"]
            print(f"Calculating {gal}'s {loc} completeness ratio....")
            DIR_dict = paths(gal, loc)
            image, Noise_map = prep_completness_data(gal, loc, DIR_dict, galaxy_data)

            completeness_ratio = calc_completeness(image, Noise_map, m_5007, galaxy_data, 3.0, 9, )

            np.save(DIR_dict["EXPORT_DIR"]+"_completeness_ratio", completeness_ratio)
        else:
            continue
