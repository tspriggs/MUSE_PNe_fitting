import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import ascii, fits
from matplotlib.patches import Rectangle, Ellipse, Circle
from functions.MUSE_Models import Moffat
from functions.PNe_functions import robust_sigma, PNe_spectrum_extractor
from functions.file_handling import open_data
import pdb as pdb

from functions.file_handling import paths

def reconstructed_image(galaxy_name, loc):
    DIR_dict = paths(galaxy_name, loc)
    
    with fits.open(DIR_dict["RAW_DATA"]) as hdu:
        hdr  = hdu[1].header
        s    = np.shape(hdu[1].data)
        wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']   
        cond = (wave >= 4900.0) & (wave <= 5100.0)
        data = np.sum(hdu[1].data[cond,:,:], axis=0)

    return data, wave, hdr


def KS2_test(dist_1, dist_2, conf_lim):
    
    """
    Input: 
          - dist_1 = distribution 1 for test
          - dist_2 = distribution 2 for test
          - conf_lim = confidence limit for consideration (0.05 equates to 5%)
          
    Return:
          - KS2_test = contains the KS D value, along with P value
          
    Purpose:
          - Take in two distributions and run the KS 2 sample test, with a set confidence limit. 
          - Print statements will inform you of the outcome. 
          - Return the KS test statistics.
    """
    
    c_a = np.sqrt(-0.5*np.log(conf_lim))
    print(f"c(a) = {round(c_a, 4)}")
    
    condition = c_a * np.sqrt((len(dist_1) + len(dist_2))/(len(dist_1) * len(dist_2)))
    
    KS2_test = stats.ks_2samp(dist_1, dist_2 )
    
#     print(KS2_test)
    print("\n")
    print("KS2 p-value test")
    if KS2_test[1] < conf_lim:
        print(f"    KS2 sample p-value less than {conf_lim}.")
        print("    Reject the Null hypothesis: The two samples are not drawn from the same distribution.")
    elif KS2_test[1]> conf_lim:
        print(f"    KS2 sample p-value greater than {conf_lim}.")
        print("    We cannot reject the Null hypothesis.")
    
    print("\n")
    print("KS2 D statistic test")
    if KS2_test[0] > condition:
        print(f"    D ({round(KS2_test[0],3)}) is greater than {round(condition,3)}")
        print(f"    The Null hypothesis is rejected. The two samples do not match within a confidence of {conf_lim}.")
    elif KS2_test[0] < condition:
        print(f"    D ({round(KS2_test[0],3)}) is less than {round(condition,3)}")
        print(f"    The Null hypothesis is NOT rejected. The two samples match within a confidence of {conf_lim}.")
        
    print("\n")
    
    return KS2_test



def completeness(galaxy, loc, DIR_dict, mag, params, dM,  peak, n_pixels, c1=0.307):
       
    #load up data
    res_data, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy, loc, DIR_dict)
    
    galaxy_image, rec_wave, rec_hdr = reconstructed_image(galaxy, loc)
    image = galaxy_image.reshape(y_data, x_data)
    
    c = 299792458.0
    z = galaxy_info["velocity"]*1e3 / c
    
    rN = np.array([robust_sigma(res_data[i]) for i in range(len(res_data))])
    Noise_map  = rN.reshape(y_data, x_data)

    # mask out regions
    xe, ye, length, width, alpha = galaxy_info["gal_mask"]
    Y, X = np.mgrid[:y_data, :x_data]
    elip_mask_gal = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1
    star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in galaxy_info["star_mask"]],0).astype(bool) # galaxy_data["star_mask"]
    mask_indx = np.array(np.where((elip_mask_gal+star_mask_sum)==True))
    # End of masking
    
    Noise_map[mask_indx[0], mask_indx[1]] = 0.0 # Noise data: set masked regions to 0.0
    image[mask_indx[0], mask_indx[1]] = 0.0 # image data: set masked regions to 0.0
    image[image<0] = 0 # iamge data: set any values that are less than 0, equal to 0.
        
    # Construct the PNe FOV coordinate grid for use when fitting PNe.
    coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
    x_fit = np.array([item[0] for item in coordinates])
    y_fit = np.array([item[1] for item in coordinates])

    # Setup range of Absolute Magnitudes to be converted to 1D max A values

    Abs_M = np.arange(-4.52,0.04,0.1)

    app_m = Abs_M + dM
#     app_m = np.arange(np.min(mag), 30.0, 0.1)

    total_flux = 10**((app_m + 13.74) / -2.5) # Ciardullo's flux calibration

    flux = total_flux*1.0e20

    init_FWHM = params['M_FWHM']
    init_beta = params['beta']

    sum_init     = np.sum(Moffat(1, init_FWHM, init_beta, n_pixels/2, n_pixels/2, x_fit, y_fit))
    input_moff_A = flux / sum_init

    # Make moffat models = F_5007 (x,y)
    Moffat_models = np.array([Moffat(moff_A, init_FWHM, init_beta, n_pixels/2., n_pixels/2., x_fit, y_fit) for moff_A in input_moff_A])

    # turn moffat model into list of Gaussian amplitudes (A_5007 (x,y))
    Amp_x_y = ((Moffat_models) / (np.sqrt(2*np.pi) * (params["LSF"]/2.35482)))
    # make a list of the maximum amplitude per magnitude
    max_1D_A = np.array([np.max(A) for A in Amp_x_y])

    completeness_ratio = np.zeros(len(app_m)).astype(np.float128)
    
    for i,amp in enumerate(max_1D_A):
        completeness_ratio[i] = (np.nansum(image[((amp / Noise_map) > peak)]) / np.nansum(image)).astype(np.float128)

    ##############
    #    PNLF    #
    ##############
    PNLF = np.exp(c1*Abs_M) * (1-np.exp(3*((-4.52 - Abs_M)))) 
#     PNLF = np.exp(c1*app_m) * (1-np.exp(3*((np.min(mag)-app_m))))
    PNLF[0] = 0.0

    return PNLF, PNLF*completeness_ratio, completeness_ratio, Abs_M, app_m



