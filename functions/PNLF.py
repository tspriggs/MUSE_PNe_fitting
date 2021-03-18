import numpy as np
from astropy.io import fits
import yaml
import numpy as np
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt

from functions.MUSE_Models import Gauss

def dM_to_D(dM, dM_err=[]):
    """Distance modulus to distance (Mpc).

    Parameters
    ----------
    dM : [float]
        float value for distance modulus.

    dM_err : list, optional
        list of distance modulus errors: [upper, lower], by default []

    Returns
    -------
    float
        Distance in Mpc, if no error's are given.

    float, float, float
        Distance in Mpc, upper error, then lower error, on distance.
    """
    distance = 10.**((dM -25.) / 5.)

    if len(dM_err)>0:
        distance_err_up = 0.2 * np.log(10) * dM_err[0] * distance
        distance_err_lo = 0.2 * np.log(10) * dM_err[1] * distance
        
        return distance, distance_err_up, distance_err_lo
    
    else:
        return distance

def D_to_dM(distance, distance_err=[]):
    """Distance (Mpc) to distance modulus.

    Parameters
    ----------
    distance : [float]
        Distance in Mpc

    distance_err : list, optional
        list of distance errors: [upper, lower], by default []

    Returns
    -------
    float
        Distance modulus, if no error's are given.

    float, float, float
        Distance modulus, upper error, then lower error.
    """
    dM = 5. * np.log10(distance) + 25

    if len(distance_err)>0:
        dM_err_up = distance_err[0] / (0.2 * np.log10(10) * distance)
        dM_err_lo = distance_err[1] / (0.2 * np.log10(10) * distance)

        return dM, dM_err_up, dM_err_lo

    else:
        return dM


def PNe_minicube_extractor(x, y, n_pix, data, wave):
    """Extract a PNe minicube from a given MUSE residual cube.

    Parameters
    ----------
    x : [float]
        x coordinate

    y : [float]
        y coordinate

    n_pix : [int]
        number of pixels

    data : [multi dimensional array]
        residual data, in list format

    wave : [list]
        wavelength array

    Returns
    -------
    [list]
        PN spectra array of shape: (n_pix*n_pix, len(wave))
    """

    xc = round(x)
    yc = round(y)
    offset = n_pix // 2
    PN = data[:, int(yc-offset):int(yc+offset+1), int(xc-offset):int(xc+offset+1)]
    PN = PN.reshape(len(wave),n_pix*n_pix)
    return np.swapaxes(PN, 1, 0)


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


def uncertainty_cube_construct(data, x_P, n_pix, PN_data, wavelength):
    """Extract and construct uncertainty cubes for PNe fitting weighting usage. Mainly used in the spaxel-by-spaxel fitting script.

    Parameters
    ----------
    data : list / array
        Residual array from emission subtraction (residual = data-model/err) of each spaxel.
    x_P : list
        List of source spaxel x coordinates.
    n_pix : int
        number of pixels wide FOV for each source.
    PN_data : list /array
        array of source spectra.
    wavelength : list
        wavelength array within which we are looking for the emission lines.

    Returns
    -------
    list / array
        residual data cube for each source, as found from spaxel-by-spaxel fitting of residual emission cube.
    """
    data[data == np.inf] = 0.01
    array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(wavelength)))
    for p in np.arange(0, len(x_P)):
        list_of_std = np.abs([robust_sigma(dat) for dat in PN_data[p]])
        array_to_fill[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]
        
    return array_to_fill


def calc_chi2(n_PNe, n_pix, n_vary, PNe_spec, wave, F_xy_list, mean_w_list, galaxy_info, G_bkg, G_grad):
    """calculate the Chi2 for each fitted PN, setting data that equals zero to 1

    Parameters
    ----------
    n_PNe : [int]
        Number of PNe.

    n_pix : [int]
        Number of Pixels in PN minicube.

    n_vary : [int]
        Number of varied parameters.

    PNe_spec : [array, float]
        PNe sepctral data.

    wave : [array, float]
        Wavelength array for spectral data.

    F_xy_list : [list, float]
        List of flux arrays for PNe.

    mean_w_list : [list, float]
        list of mean wavelength positons for PNe.

    galaxy_info : [dict]
        galaxy info dict which contains galaxy information.

    G_bkg : [float]
        Gaussian background value, from fitter, for straight line.

    G_grad : [float]
        Gaussian gradient value, from fitter, for straight line.

    Returns
    -------
    [float]
        chi_sqr, rechi
    """
    c = 299792458.0
    z = galaxy_info["velocity"]*1e3 / c

    gauss_list, redchi, Chi_sqr = [], [], []
    for p in range(n_PNe):
        PNe_n = np.copy(PNe_spec[p])
        flux_1D = np.copy(F_xy_list[p][0])
        A_n = ((flux_1D) / (np.sqrt(2*np.pi) * (galaxy_info["LSF"]/ 2.35482)))
    
        list_of_gauss = [Gauss(wave, A, mean_w_list[p][0], galaxy_info["LSF"], G_bkg[p], G_grad[p], z) for A in A_n]
        for kk in range(len(PNe_n)):
            temp = np.copy(list_of_gauss[kk])
            idx  = np.where(PNe_n[kk] == 0.0)[0]
            temp[idx] = 0.0
            PNe_n[kk,idx] = 1.0
            list_of_gauss[kk] = np.copy(temp)
        rN   = np.std(PNe_n - list_of_gauss)
        res  = PNe_n - list_of_gauss
        Chi2 = np.nansum((res**2)/(rN**2))
        redchi.append(Chi2/ ((len(wave) * n_pix**2) - n_vary))
        gauss_list.append(list_of_gauss)
        Chi_sqr.append(Chi2)
    
    return Chi_sqr, redchi


def KS2_test(dist_1, dist_2, conf_lim, v=True):
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
    
    condition = c_a * np.sqrt((len(dist_1) + len(dist_2))/(len(dist_1) * len(dist_2)))
    
    KS2_test = stats.ks_2samp(dist_1, dist_2 )
    
#     print(KS2_test)
    if v == True:
        print(f"c(a) = {round(c_a, 4)}")
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
 

def LOSV_interloper_check(DIR_dict, galaxy_info, fitted_wave_list, PNe_indx, x_PNe, y_PNe):
    """
    Calculate the PN LOSVD from the systemic velocity, measured from the centre of the galaxy.
    Also returns the indexes of any PN found to be interlopers (vel ratio is outside 3 sigma).

    Parameters:
        - DIR_dict          - directory dictionary
        - galaxy_info       - dictionary of galaxy info
        - fitted_wave_list  - list of fitted mean wavelength's for each object
        - PNe_indx          - indexes of where ID == "PNe"
        - x_PNe             - list of x coordinates of sources
        - y_PNe             - list of y coordinates of sources
        
    Return:
        - PNe_LOS_V         - PNe Line of Sight velocities
        - interlopers       - list of interloper indexes
    """   
    ## Velocity from files
#     with fits.open(DIR_dict["RAW_DIR"]+f"/{galaxy_info['Galaxy name']}center_ppxf_SPAXELS.fits") as hdulist_ppxf:    
    with fits.open(f"/local/tspriggs/re_reduced_F3D/gist_results/{galaxy_info['Galaxy name']}center_center/{galaxy_info['Galaxy name']}center_ppxf.fits") as hdulist_ppxf:

        v_star, s_star = hdulist_ppxf[1].data.V, hdulist_ppxf[1].data.SIGMA
    
    
    with fits.open(DIR_dict["RAW_DIR"]+f"/{galaxy_info['Galaxy name']}center_table.fits") as hdulist_table:
        X_star, Y_star = hdulist_table[1].data.XBIN, hdulist_table[1].data.YBIN
        flux_star = hdulist_table[1].data.FLUX
    
    idx = np.nanargmax(flux_star)
    X_star, Y_star = X_star-X_star[idx], Y_star-Y_star[idx]
    
    # systemic velocity from inner 5 arcsecond circle of galaxy centre
    cond = np.sqrt( (X_star)**2 + (Y_star)**2 ) <= 5.0
    vsys = np.median(v_star[cond]) # v_star may be the number of fitted pixels, need to reshape to be like res cube
    v_star = v_star-vsys
    
    c = 299792458.0
    
    LOS_z = (vsys * 1e3) / c
    
    LOS_de_z = np.array(fitted_wave_list[:,0] / (1 + LOS_z))
        
    PNe_LOS_V = (c * (LOS_de_z - 5006.77) / 5006.77) / 1000. 
        
    gal_centre_pix = Table.read("exported_data/galaxy_centre_pix.dat", format="ascii")
    
    gal_ind = np.where(gal_centre_pix["Galaxy"]==galaxy_info["Galaxy name"])
    gal_x_c = gal_centre_pix["x_pix"][gal_ind]
    gal_y_c = gal_centre_pix["y_pix"][gal_ind]
    
    xpne, ypne = (x_PNe[PNe_indx]-gal_x_c)*0.2, (y_PNe[PNe_indx]-gal_y_c)*0.2
    
    # Estimating the velocity dispersion of the PNe along the LoS
    def sig_PNe(X_star, Y_star, v_stars, sigma, x_PNe, y_PNe, vel_PNe):
    
        d_PNe_to_skin = np.zeros(len(x_PNe))
        Vs_PNe = np.ones(len(x_PNe)) # Velocity of the closest star
        Ss_PNe = np.ones(len(x_PNe)) # Sigma for each PNe
        i_skin_PNe = []
    
        """ To estimate the velocity dispersion for PNe we need to
        extract the sigma of the closest stars for each PNe """
    
        for i in range(len(x_PNe)):
            r_tmp = np.sqrt((X_star-x_PNe[i])**2+(Y_star-y_PNe[i])**2)
            d_PNe_to_skin[i] = min(r_tmp)
            i_skin_PNe.append(r_tmp.argmin())
    
        Vs_PNe  = v_stars[i_skin_PNe]
        Ss_PNe  = sigma[i_skin_PNe]
        rad_PNe = np.sqrt(x_PNe**2+y_PNe**2)
        k = np.where(d_PNe_to_skin > 1.0)
    
        return rad_PNe, (vel_PNe-Vs_PNe)/Ss_PNe, k, Vs_PNe, Ss_PNe
    
    rad_PNe, vel_ratio, k, Vs_PNe, Ss_PNe  = sig_PNe(X_star, Y_star, v_star, s_star, xpne, ypne, PNe_LOS_V[PNe_indx])
    # rad_PNe, vel_ratio, k  = sig_PNe(X_star, Y_star, v_star, s_star, xpne, ypne, PNe_df["V (km/s)"].loc[PNe_df["Filter"]=="Y"])
    # rad_PNe, vel_ratio, k  = sig_PNe(X_star, Y_star, v_star, s_star, xpne, ypne, PNe_df["PNe_LOS_V"])
    
    # Filter interlopers by PN that have vel ratio's outside a 3 sigma range
    interlopers = np.where((vel_ratio<-3) | (vel_ratio>3))
    # for inter in interlopers:
    #     PNe_df.loc[PNe_df["PNe number"]==inter, "Filter"] = "N"
        
    #print(stats.norm.fit(PNe_df["V (km/s)"].loc[PNe_df["Filter"]=="Y"].values))
    
    return PNe_LOS_V, interlopers, vel_ratio, vsys


def generate_mask(img_shape, mask_params=[], mask_shape=""):
    """Create either elliptical or circular mask based on parameters and input mask shape choice.

    Parameters
    ----------
    img_shape : list
        [Y axis length, X axis length]  (e.g. [y_data, x_data]).

    mask_params : list, optional
        Parameters for the mask; 5 for ellipse, 3 for circles. Normally stored in the 'galaxy_info.yml' file, by default [].

    mask_shape : string, optional
        Choose from "ellipse" or "circle". by default "".
        
    Returns
    -------
    bool, mask
        Returned mask has the same dimensions as input img_shape.
    """

    Y, X = np.mgrid[:img_shape[0], :img_shape[1]] # [Y, X]
    mask = []
    if (mask_shape == "ellipse") & (len(mask_params) == 5):
        xe, ye, length, width, alpha = mask_params

        mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + \
            (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1

    elif (mask_shape == "circle") & (len(mask_params) == 3):
        xc, yc, rc = mask_params

        mask = (Y - yc)**2 + (X - xc)**2 <= rc*rc

    else:
        print("Check your spelling for mask_params!")

    return mask


def plot_single_spec(n, DIR_dict, wavelength, PNe_spectra, model_spectra_list, i):
    """Simple function to plot the integrated spectrum of a given PNe, showing the residual data, the best-fit model, and the residual of data minus best-fit model.

    Parameters
    ----------
    n : int
        integer number for the PNe to be plotted
    DIR_dict : dict
        Dictionary of useful directories for file opening and saving.
    wavelength : list / array
        lambda wavelength range array.
    PNe_spectra : list / array
        Residual data list of the PNe spectrum to be plotted
    model_spectra_list : list / array
        best-fit model of the PNe to be plotted
    i : int
        integer value designating the source as the ith brightest of those plotted, used in the filename.
    """

    res = PNe_spectra[n] - model_spectra_list[n]
    plt.figure(figsize=(20,8))
    plt.plot(wavelength, np.sum(PNe_spectra[n],0), label="data")
    plt.plot(wavelength, np.sum(res, 0), label="residuals")
    plt.plot(wavelength, np.sum(model_spectra_list[n], 0), label="model")
    plt.title(f"PNe {n}")
    plt.legend(fontsize=18)
    plt.savefig(DIR_dict["PLOT_DIR"]+f"_{i}_brightest_single_spec.png", bbox_inches='tight')