# PARAMETERS:
#
# galaxy: galaxy name
#
# mag: array of absolute magnitudes (M 5007)
#
# params: A dictionary with the values of M_FWHM and BETA of the PSF
#
# D: Distance in Mpc
#
# image: reconstructed white image of the galaxy
#
# peak: minimum signal-to-noise of the PNLF (typically >3)
#
# region: region of analysis. If there is just one region put the FCCXXX_residuals_list.fits
#         in a new folder called FCCXXX_data/center/

# Opening the residual cube getting with GandALF
def open_data(choose_galaxy,region):
    # Load in the residual data, in list form
    hdulist = fits.open(choose_galaxy+"_data/"+region+"/"+choose_galaxy+"_residuals_list.fits") # Path to data
    res_hdr = hdulist[0].header # extract header from residual cube

    # Check to see if the wavelength is in the fits fileby checking length of fits file.
    if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
        wavelength = np.exp(hdulist[1].data)
    else:
        wavelength = np.load(galaxy_data["wavelength"])

    # Use the length of the data to return the size of the y and x dimensions of the spatial extent.
    if (choose_galaxy == 'FCC153') & (region == "center"):
        y_data, x_data, n_data = data_cube_y_x(len(hdulist[0].data))
    elif (choose_galaxy != 'FCC153') & (region == "center"):
        x_data, y_data, n_data = data_cube_y_x(len(hdulist[0].data))
    if (choose_galaxy == 'FCC153') & (region == "halo"):
        x_data, y_data, n_data = data_cube_y_x(len(hdulist[0].data))
    elif (choose_galaxy != 'FCC153') & (region == "halo"):
        if choose_galaxy == 'FCC177':
            x_data, y_data, n_data = data_cube_y_x(len(hdulist[0].data))
        else:
            y_data, x_data, n_data = data_cube_y_x(len(hdulist[0].data))

    return x_data, y_data, hdulist, wavelength

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

def completeness(galaxy,mag,params,D,image,peak,region):

    n_pixels = 7
    x_data, y_data, hdulist, wavelength = open_data(galaxy, region)

    print('\n- Fitting the residual cube to avoid the wiggles -')
    new_res = []
    for i in range(len(hdulist[0].data)):
        poly = np.polyfit(wavelength,hdulist[0].data[i],6)
        aux = 0
        for j in range(len(poly)):
            aux = aux+poly[j]*wavelength**(len(poly)-j-1)
        new_res.append(hdulist[0].data[i]-aux)
    hdulist[0].data = np.array(new_res)
    print('\nDone!')

    rN = []
    for i in range(len(hdulist[0].data)):
        rN.append(robust_sigma(hdulist[0].data[i]))
    rN = np.array(rN)
    Noise_map_cen  = rN.reshape(y_data, x_data)

    # mask out regions where sep masks
    #Y, X = np.mgrid[:y_data, :x_data]
    #xe = xc
    #ye = yc
    #length= 15
    #width = 15
    #alpha = 0.15
    #elip_mask_gal = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1

    #Noise_map_cen[elip_mask_gal == True] = 0.0

    #x_data, y_data = open_data(galaxy, 'halo')
    #Noise_map_hal  = np.abs(np.std(fits.open(hal_file)[0].data, axis=1))
    #Noise_map_hal  = Noise_map_hal.reshape(y_data, x_data)

    # Construct the PNe FOV coordinate grid for use when fitting PNe.
    coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
    x_fit = np.array([item[0] for item in coordinates])
    y_fit = np.array([item[1] for item in coordinates])

    n_pixels = 7

    # Setup range of Absolute Magnitudes to be converted to 1D max A values

    Abs_M = np.linspace(-4.5,0.0,46) #bins_cens
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
    #bins_cens = bins_cens[:-1] + dM
    bins_cens = Abs_M + dM

    #bins = bins[2:]

    #bins_cens = bins_cens[2:]

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
    #fig, axs = plt.subplots(2,3, figsize=(20, 10))
    #axs = axs.ravel()
    #pdb.set_trace()
    zeros = []
    for i,a in enumerate(max_1D_A_cen):
        temp = np.ones_like(Noise_mask_cen)
        temp[((a / Noise_mask_cen) < peak)] = 0.0

        #zeros.append(len(np.where(temp <= 0.0)[0])

        #if zeros[i]
        ratio_counter_cen[i] = (np.nansum(image[((a / Noise_mask_cen) > peak)])/np.nansum(image)).astype(np.float128)

        #print('A:',a)
        #print('R:', ratio_counter_cen[i])
        #print('Noise:',len(np.where(temp <= 0.0)[0]))
        #print('I:',i)

        #Noise_mask_plot_cen.append(Noise_mask_cen)

    ##############
    #    PNLF    #
    ##############

    #plt.figure(1,figsize=(15,20))
    PNLF = np.exp(0.307*Abs_M) * (1-np.exp(3*((-4.47 - Abs_M)))) #3000 * np.exp(-0.307*(5. * np.log10(18.7) + 25.)) *
    PNLF[0] = 0.0

    return PNLF, PNLF*ratio_counter_cen, Abs_M
