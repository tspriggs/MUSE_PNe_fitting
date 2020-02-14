from astropy.io import fits
import yaml
import numpy as np

def PNe_spectrum_extractor(x, y, n_pix, data, x_d, wave):
    """
    """
    xc = round(x)
    yc = round(y)
    offset = n_pix // 2
    #calculate the x y coordinates of each pixel in n_pix x n_pix square around x,y input coordinates
    y_range = np.arange(yc - offset, (yc - offset)+n_pix, 1, dtype=int)
    x_range = np.arange(xc - offset, (xc - offset)+n_pix, 1, dtype=int)
    ind = [i * x_d + x_range for i in y_range]
    return data[np.ravel(ind)]


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


def uncertainty_cube_construct(data, x_P, y_P, n_pix, x_data, wavelength):
    """
    Extract and construct uncertainty cubes for PNe fitting weighting
    """
    data[data == np.inf] = 0.01
    extract_data = np.array([PNe_spectrum_extractor(x, y, n_pix, data, x_data, wave=wavelength) for x,y in zip(x_P, y_P)])
    array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(wavelength)))
    for p in np.arange(0, len(x_P)):
        list_of_std = np.abs([robust_sigma(dat) for dat in extract_data[p]])
        array_to_fill[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]
        
    return array_to_fill


def calc_chi2(n_PNe, n_pix, n_vary, PNe_spec, wave, F_xy_list, mean_w_list, galaxy_data, G_bkg, G_grad):
    """
    calculate the Chi2 for each fitted PN, setting data that equals zero to 1
    
    Parameters:
        - n_PNe
        - n_pix
        - n_vary
        - PNe_spec
        - wave
        - F_xy_list
        - mean_w_list
        - galaxy_data
        - G_bkg
        - G_grad
        
    Returns:
        - Chi_sqr
        - redchi
        
    """
    gauss_list, redchi, Chi_sqr = [], [], []
    for p in range(n_PNe):
        PNe_n = np.copy(PNe_spec[p])
        flux_1D = np.copy(F_xy_list[p][0])
        A_n = ((flux_1D) / (np.sqrt(2*np.pi) * (galaxy_data["LSF"]// 2.35482)))
        
        def gauss(x, amp, mean, FWHM, bkg, grad):
            stddev = FWHM/ 2.35482
            return ((bkg + grad*x) + np.abs(amp) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.)) +
                    (np.abs(amp)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / (stddev**2.)))
    
        list_of_gauss = [gauss(wave, A, mean_w_list[p][0], galaxy_data["LSF"], G_bkg[p], G_grad[p]) for A in A_n]
        for kk in range(len(PNe_n)):
            temp = np.copy(list_of_gauss[kk])
            idx  = np.where(PNe_n[kk] == 0.0)[0]
            temp[idx] = 0.0
            PNe_n[kk,idx] = 1.0
            list_of_gauss[kk] = np.copy(temp)
        rN   = robust_sigma(PNe_n - list_of_gauss)
        res  = PNe_n - list_of_gauss
        Chi2 = np.sum((res**2)/(rN**2))
        # s    = np.shape(PNe_n)
        redchi.append(Chi2/ ((len(wave) * n_pix**2) - nvarys))
        gauss_list.append(list_of_gauss)
        Chi_sqr.append(Chi2)
    
    return Chi_sqr, redchi


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



def prep_impostor_files(galaxy_name):
    """
    Prepare files for impostor checks, no return, only saved files.
    
    Parameters:
        - galaxy_name - string
    """
    
    ############# WEIGHTED MUSE data PNe ##############
    def PSF_weight(MUSE_p, model_p, r_wls, spaxels=81):
           
        coeff = np.polyfit(r_wls, np.clip(model_p[0, :], -50, 50), 1) # get continuum on first spaxel, assume the same across the minicube
        poly = np.poly1d(coeff)
        tmp = np.copy(model_p)
        for k in np.arange(0,spaxels):
             tmp[k,:] = poly(r_wls)
                
        res_minicube_model_no_continuum = model_p - tmp # remove continuum
        
        # PSF weighted minicube
        sum_model_no_continuum = np.nansum(res_minicube_model_no_continuum, 0)
        weights = np.nansum(res_minicube_model_no_continuum, 1)
        nweights = weights / np.nansum(weights) # spaxel weights
        weighted_spec = np.dot(nweights, MUSE_p) # dot product of the nweights and spectra
    
        return weighted_spec
    
    with fits.open("/local/tspriggs/Fornax_data_cubes/"+galaxy_name+"center.fits") as raw_hdulist:
        raw_data = raw_hdulist[1].data
        raw_hdr = raw_hdulist[1].header
        raw_s = raw_hdulist[1].data.shape # (lambda, y, x)
        full_wavelength = raw_hdr['CRVAL3']+(np.arange(raw_s[0])-raw_hdr['CRPIX3'])*raw_hdr['CD3_3']
        
        if len(raw_hdulist) == 3:
            stat_list = np.copy(raw_hdulist[2].data).reshape(raw_s[0], raw_s[1]*raw_s[2])
            stat_list = np.swapaxes(stat_list, 1,0)
        elif len(raw_hdulist) == 2:
            stat_list = np.ones_like(cube_list)
            
    
    cube_list = np.copy(raw_data).reshape(raw_s[0], raw_s[1]*raw_s[2]) # (lambda, list of len y*x)
    cube_list = np.swapaxes(cube_list, 1,0) # (list of len x*y, lambda)
    
    
    raw_minicubes = np.array([PNe_spectrum_extractor(x,y,n_pixels, cube_list, raw_s[2], full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    # stat_minicubes = np.ones_like(raw_minicubes)
    stat_minicubes = np.array([PNe_spectrum_extractor(x,y,n_pixels, stat_list, raw_s[2], full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    
    sum_raw  = np.nansum(raw_minicubes,1)
    sum_stat = np.nansum(stat_minicubes, 1)
    
    hdu_raw_minicubes = fits.PrimaryHDU(sum_raw,raw_hdr)
    hdu_stat_minicubes = fits.ImageHDU(sum_stat)
    hdu_long_wavelength = fits.ImageHDU(full_wavelength)
    
    raw_hdu_to_write = fits.HDUList([hdu_raw_minicubes, hdu_stat_minicubes, hdu_long_wavelength])
    
    raw_hdu_to_write.writeto(f"exported_data/{galaxy_name}/{galaxy_name}_MUSE_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_MUSE_PNe.fits file saved.")
    
    
    ##### Residual .fits file ################
    residual_hdu = fits.PrimaryHDU(PNe_spectra)
    wavelenth_residual = fits.ImageHDU(wavelength)
    resid_hdu_to_write = fits.HDUList([residual_hdu, wavelenth_residual])
    resid_hdu_to_write.writeto(f"exported_data/{galaxy_name}/{galaxy_name}_residuals_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_residuals_PNe.fits file saved.")
    
    
    ####### 3D model .fits file ##################
    models_hdu = fits.PrimaryHDU(model_spectra_list)
    wavelenth_models = fits.ImageHDU(wavelength)
    model_hdu_to_write = fits.HDUList([models_hdu, wavelenth_models])
    model_hdu_to_write.writeto(f"exported_data/{galaxy_name}/{galaxy_name}_3D_models_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_residuals_PNe.fits file saved.")
    
    weighted_PNe = np.ones((n_PNe, n_pixels**2, len(full_wavelength)))  #N_PNe, spaxels, wavelength length
    
    for p in np.arange(0, n_PNe):
        weighted_PNe[p] = PSF_weight(raw_minicubes[p], model_spectra_list[p], wavelength, n_pixels**2)
    
    sum_weighted_PNe = np.nansum(weighted_PNe, 1)
    
    hdu_weighted_minicubes = fits.PrimaryHDU(sum_weighted_PNe, raw_hdr)
    hdu_weighted_stat = fits.ImageHDU(np.nansum(stat_minicubes,1))
    
    weight_hdu_to_write = fits.HDUList([hdu_weighted_minicubes, hdu_stat_minicubes, hdu_long_wavelength])
    
    weight_hdu_to_write.writeto(f"../../gist_PNe/inputData/{galaxy_name}MUSEPNeweighted.fits", overwrite=True)
    print(f"{galaxy_name}_MUSE_PNe_weighted.fits file saved.")
    

