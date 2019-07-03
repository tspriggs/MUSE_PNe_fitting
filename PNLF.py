import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii, fits
from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
from MUSE_Models import PNe_spectrum_extractor, data_cube_y_x
import pdb as pdb

def open_data(choose_galaxy):
    # Load in the residual data, in list form
    hdulist = fits.open(choose_galaxy+"_data/"+choose_galaxy+"_residuals_list.fits") # Path to data
    res_hdr = hdulist[0].header # extract header from residual cube

    # Check to see if the wavelength is in the fits fileby checking length of fits file.
    if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
        wavelength = np.exp(hdulist[1].data)
    else:
        wavelength = np.load(galaxy_data["wavelength"])

    # Use the length of the data to return the size of the y and x dimensions of the spatial extent.
    if choose_galaxy == "FCC219" or choose_galaxy =="FCC193" or choose_galaxy =="FCC083":
        x_data, y_data, n_data = data_cube_y_x(len(hdulist[0].data))
    elif choose_galaxy == "FCC161":
        y_data, x_data = 451, 736
    else:
        y_data, x_data, n_data = data_cube_y_x(len(hdulist[0].data))

    return x_data, y_data, hdulist, wavelength

def reconstructed_image(choose_galaxy):
    hdu  = fits.open(choose_galaxy+"_data/"+choose_galaxy+'center.fits')
    data = hdu[1].data
    hdr  = hdu[1].header
    s    = np.shape(data)
    wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

    cond = (wave >= 4900.0) & (wave <= 5100.0)
    data = np.sum(data[cond,:,:],axis=0)

    return data, wave

def completeness(galaxy, mag, params, D, image, peak, gal_mask_params, 
                 star_mask_params, mask=False, c1=0.307):

    def robust_sigma(y, zero=False):
            """
            Biweight estimate of the scale (standard deviation).
            Implements the approach described in
            "Understanding Robust and Exploratory Data Analysis"
            Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

            """
            y = np.ravel(y)
            d = y if zero else y - np.nanmedian(y)

            mad = np.nanmedian(np.abs(d))
            u2 = (d/(9.0*mad))**2  # c = 9
            good = u2 < 1.0
            u1 = 1.0 - u2[good]
            num = y.size * ((d[good]*u1**2)**2).sum()
            den = (u1*(1.0 - 5.0*u2[good])).sum()
            sigma = np.sqrt(num/(den*(den - 1.0)))  # see note in above reference

            return sigma

    n_pixels = 9
    x_data, y_data, hdulist, wavelength = open_data(galaxy)

    rN = []
    for i in range(len(hdulist[0].data)):
        rN.append(robust_sigma(hdulist[0].data[i]))
    rN = np.array(rN)
    Noise_map_cen  = rN.reshape(y_data, x_data)

    # mask out regions where sep masks
    if mask == True:
        xe, ye, length, width, alpha = gal_mask_params
        
        Y, X = np.mgrid[:y_data, :x_data]
        elip_mask_gal = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1
        
        star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in star_mask_params],0).astype(bool)
        
        mask_indx = np.array(np.where((elip_mask_gal+star_mask_sum)==True))
        
        Noise_map_cen[mask_indx[0], mask_indx[1]] = 0.0 #np.nan
        image[mask_indx[0], mask_indx[1]] = 0.0 # np.nan
        image[image<0] = 0
    #x_data, y_data = open_data(galaxy, 'halo')
    #Noise_map_hal  = np.abs(np.std(fits.open(hal_file)[0].data, axis=1))
    #Noise_map_hal  = Noise_map_hal.reshape(y_data, x_data)
    elif mask == False:
        image[image<0] = 0
    # Construct the PNe FOV coordinate grid for use when fitting PNe.
    coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
    x_fit = np.array([item[0] for item in coordinates])
    y_fit = np.array([item[1] for item in coordinates])

    n_pixels = 9

    # Setup range of Absolute Magnitudes to be converted to 1D max A values

    Abs_M = np.arange(-4.52,0.04,0.1) #bins_cens
    dM = 5. * np.log10(D) + 25

    def moffat(amplitude, x_0, y_0, FWHM, beta):
        gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_fit - x_0)**2 + (y_fit - y_0)**2) / gamma**2
        return amplitude * (1 + rr_gg)**(-beta)

    def gaussian(x, amplitude, mean, stddev, bkg, grad):
        return (bkg + grad*x + np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.)) +
                     (np.abs(amplitude)/3.) * np.exp(- 0.5 * (x - (mean - 47.9399))** 2 / (stddev**2.)))

    bins, bins_cens, other = plt.hist(mag, bins=10, edgecolor="black", linewidth=0.8, label="M 5007 > "+str(peak)+"A/rN", alpha=0.5)
    plt.close()

    bins_cens = Abs_M + dM

    app_m = bins_cens


    total_flux = 10**((app_m + 13.74) / -2.5) # Ciardullo's flux calibration

    flux = total_flux*1.0e20

    init_FWHM_cen = params['M_FWHM']
    init_beta_cen = params['beta']

    sum_init_cen     = np.sum(moffat(1, n_pixels/2, n_pixels/2, init_FWHM_cen, init_beta_cen))
    input_moff_A_cen = flux / sum_init_cen

    # Make moffat models = F_5007 (x,y)
    Moffat_models_cen = np.array([moffat(moff_A, n_pixels/2., n_pixels/2., init_FWHM_cen, init_beta_cen) for moff_A in input_moff_A_cen])

    # A_5007 (x,y)
    Amp_x_y_cen = ((Moffat_models_cen) / (np.sqrt(2*np.pi) * 1.19))
    max_1D_A_cen = np.array([np.max(A) for A in Amp_x_y_cen])

    N_data_cen = len(np.nonzero(Noise_map_cen)[0])

    Noise_mask_cen = Noise_map_cen
    ratio_counter_cen = np.zeros(len(app_m)).astype(np.float128)

    M_5007_detlim = -2.5*np.log10(total_flux) - 13.74 - dM
    
    zeros = []
    for i,a in enumerate(max_1D_A_cen):
        ratio_counter_cen[i] = (np.nansum(image[((a / Noise_mask_cen) > peak)])/np.nansum(image)).astype(np.float128)

    ##############
    #    PNLF    #
    ##############

    PNLF = np.exp(c1*Abs_M) * (1-np.exp(3*((-4.52 - Abs_M)))) 
    PNLF[0] = 0.0

    return PNLF, PNLF*ratio_counter_cen, Abs_M



