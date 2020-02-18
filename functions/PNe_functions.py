from astropy.io import fits
import yaml
import numpy as np
from scipy import stats


from functions.MUSE_Models import Gauss

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


def calc_chi2(n_PNe, n_pix, n_vary, PNe_spec, wave, F_xy_list, mean_w_list, galaxy_info, G_bkg, G_grad):
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
        - galaxy_info
        - G_bkg
        - G_grad
        
    Returns:
        - Chi_sqr
        - redchi
        
    """
    c = 299792458.0
    z = galaxy_info["velocity"]*1e3 / c

    gauss_list, redchi, Chi_sqr = [], [], []
    for p in range(n_PNe):
        PNe_n = np.copy(PNe_spec[p])
        flux_1D = np.copy(F_xy_list[p][0])
        A_n = ((flux_1D) / (np.sqrt(2*np.pi) * (galaxy_info["LSF"]// 2.35482)))
    
        list_of_gauss = [Gauss(wave, A, mean_w_list[p][0], galaxy_info["LSF"], G_bkg[p], G_grad[p], z) for A in A_n]
        for kk in range(len(PNe_n)):
            temp = np.copy(list_of_gauss[kk])
            idx  = np.where(PNe_n[kk] == 0.0)[0]
            temp[idx] = 0.0
            PNe_n[kk,idx] = 1.0
            list_of_gauss[kk] = np.copy(temp)
        rN   = robust_sigma(PNe_n - list_of_gauss)
        res  = PNe_n - list_of_gauss
        Chi2 = np.sum((res**2)/(rN**2))
        redchi.append(Chi2/ ((len(wave) * n_pix**2) - n_vary))
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
 

def calc_Lbol_and_mag(DIR_dict, galaxy_info, dist_mod, dM_err=[0.1,0.1]):
    """
    Feature:

    Parameters:
        - DIR_dict      - directory dictionary
        - galaxy_info   - dictionary of galaxy info
        - z             - redshfit
        - dist_mod      - distance modulus
        - dM_err        - list of two elements [dM err upper, dM err lower], or [dM err] 
                            - If one number passed, then it is used for both error bounds.
    Return:
        - L_bol - containing 
    """
    c = 299792458.0
    z = galaxy_info["velocity"]*1e3 / c
    raw_data_cube = DIR_dict["RAW_DATA"]  # read in raw data cube

    xe, ye, length, width, alpha = galaxy_info["gal_mask"]

    with fits.open(DIR_dict["RAW_DATA"]) as orig_hdulist:
        raw_data_cube = np.copy(orig_hdulist[1].data)
        h1 = orig_hdulist[1].header
    
    raw_shape = np.shape(raw_data_cube)
    
    # Setup galaxy and star masks
    Y, X = np.mgrid[:raw_shape[1], :raw_shape[2]]
    elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + \
        (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1

    star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc*rc for xc, yc, rc in star_mask_params], 0).astype(bool)

    # Combine elip_mask and star_mask_sum to make total_mask
    total_mask = ((np.isnan(raw_data_cube[1, :, :]) == False) & (
        elip_mask == False) & (star_mask_sum == False))
    indx_mask = np.where(total_mask == True)

    good_spectra = np.zeros((raw_shape[0], len(indx_mask[0])))

    for i, (y, x) in enumerate(zip(tqdm(indx_mask[0]), indx_mask[1])):
        good_spectra[:, i] = raw_data_cube[:, y, x]

    #for testing at sompoint --> np.nansum(raw_data_cube[:, indx_mask[0], indx_mask[1]], 1)

    print("Collapsing cube now....")

    gal_lin = np.nansum(good_spectra, 1)

    print("Cube has been collapsed...")
    # Check for if error range given, or single number supplied.
    if len(dM_err) > 1:
        L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z,
                       vel=galaxy_info["velocity"], dist_mod=dM, dM_err=[dM_err_up, dM_err_lo])

    elif len(dM_err) == 1: # if dM_err is len 1, use value for both bounds
        L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z,
                           vel=galaxy_info["velocity"], dist_mod=dM, dM_err=[dM_err, dM_err])

    return L_bol
