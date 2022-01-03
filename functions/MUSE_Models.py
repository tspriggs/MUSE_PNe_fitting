import numpy as np
from lmfit import Parameters

def Gauss(lam, amp, mean, FWHM, bkg=0.0, grad=0.0, z=0.0):
    """Gaussian equation for [OIII] 4959 and 5007 emission lines

    Parameters
    ----------
    lam : array
        lambda wavelength range array.

    amp : float
        amplitude of Gaussian model at 5007 A.

    mean : float
         position in wavelngth of the 5007 A line, may shift due to redshfit.

    FWHM : float
        Full Width Half Maximum, in spectral pixels.

    bkg : float, optional
        straight line background value. default is 0.0.

    grad : float, optional
        straight line gradient value. default is 0.0.

    z : float, optional
        Redshift value. default is 0.0


    Returns
    -------
    [array]
        twin peaked Gaussian model for 4959 and 5007 A emission lines.
    """
    stddev = FWHM / 2.35482
    return ((bkg + grad*lam ) + np.abs(amp) * np.exp(- 0.5 * (lam - mean) ** 2 / (stddev**2.)) + \
            (np.abs(amp)/3) * np.exp(- 0.5 * (lam - (mean - 47.9399*(1+z))) ** 2 / (stddev**2.)))

# Moffat model function
def Moffat(amp, FWHM, beta, x, y, x_2D, y_2D):
    """Moffat function for 2D flux distribution of [OIII] emission.

    Parameters
    ----------
    amp : float
        amplitude of Moffat function.
    FWHM : float
        Full Width Half Maximum, in pixels.
    beta : float
        weight given to the wings of the distribution.
    x : float
        x location of centre point.
    y : float
        y location of centre point.
    x_2D : float
        x array of matrix sized n_pixel x n_pixel.
    y_2D : float
        y array of matrix sized n_pixel x n_pixel.

    Returns
    -------
    [array]
        2D moffat distribution
    """
    alpha = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / alpha**2.
    return amp * (2 * ((beta -1)/(alpha**2))) * ((1 + rr_gg)**(-beta))

def generate_3D_fit_params(wave=5007, FWHM=4.0, FWHM_err=0.1, beta=2.5, beta_err=0.3, LSF=2.81, em_dict=None, \
                            vary_LSF=False, vary_PSF=False, z=0.0, n_pixels=9):
    """Use initial guess values to form the LMfit parameters for 3D fitting of PNe. Also decide whether to vary certain parameters through boolean flags.

    Parameters
    ----------
    wave : int, optional
        Wavelength value, in Angstrom, to start from, by default 5007 A.

    FWHM : float, optional
        Moffat function Full Width Half Maximum (FWHM), by default 4.0.

    FWHM_err : float, optional
        Error in Moffat FWHM parameter, derived from PSF fititng, by default 0.1.

    beta : float, optional
        Moffat function beta paramter, controlling the kurtosis (tail) of distribution, by default 2.5.

    beta_err : float, optional
        Error in Moffat beta parameter, derived from PSF fititng, by default 0.3.

    LSF : float, optional
        Line Spread Function used within the Gaussian component of model, in Angstrom, by default 2.81.

    em_dict : dict, optional
        Dictionary of emission lines to be fit, mainly [OIII] doublet here, by default None.

    vary_LSF : bool, optional
        Boolean switch for whether or not the LSF paramter should be varied, by default False.

    vary_PSF : bool, optional
        Boolean switch for whether or not the PSF parameters should be varied, by default False.
    
    Returns
    -------
    [dict]
        LMfit paramter instance, using initial guess values, limits and varying required parameters.
    """    

    params = Parameters()
    # loop through emission dictionary to add different element parameters
    for em in em_dict:
        #Amplitude params for each emission
        params.add('Amp_2D_{}'.format(em), value=em_dict[em][0], min=0.00001, max=1e5, expr=em_dict[em][1])
        #Wavelength params for each emission
        if em_dict[em][2] == None:
            params.add("wave_{}".format(em), value=wave, min=wave-15., max=wave+15.)
        else:
            params.add("wave_{}".format(em), expr=em_dict[em][2].format(z))

    params.add("x_0", value=(n_pixels/2.), min=(n_pixels/2.) -3, max=(n_pixels/2.) +3)
    params.add("y_0", value=(n_pixels/2.), min=(n_pixels/2.) -3, max=(n_pixels/2.) +3)
    params.add("LSF", value=LSF, vary=vary_LSF, min=LSF-1, max=LSF+1)
    params.add("M_FWHM", value=FWHM, min=FWHM - FWHM_err, max=FWHM + FWHM_err, vary=vary_PSF)
    params.add("beta", value=beta, min=beta - beta_err, max=beta + beta_err, vary=vary_PSF)
    params.add("Gauss_bkg",  value=1.0, vary=True)
    params.add("Gauss_grad", value=0.0001, min=-2, max=2, vary=True)

    return params

# Multi wavelength analysis model
def PNe_3D_fitter(params, lam, x_2D, y_2D, emission_dict):
    """[summary]

    Parameters
    ----------
    params : dict
        LMfit parameter dictionary object.
    lam : list / array
        Wavelength array over which to look for the emission lines of interest.
    x_2D : list / array
        x array of matrix sized n_pixel x n_pixel.
    y_2D : list / array
        y array of matrix sized n_pixel x n_pixel.
    emission_dict : dict
        Dictionary of emission line names and wavelength positions that are to be fitted for.

    Returns
    -------
    list / array
        model_spectra, [max amplitude of model, flux distribution array, amplitude distribution array, model_spectra]
    """
    # loop through emission dict and append to Amp 2D and wave lists
    amp_2D_list = [params["Amp_2D_{}".format(em)] for em in emission_dict]
    x_0 = params['x_0']
    y_0 = params['y_0']
    LSF = params["LSF"]
    M_FWHM = params["M_FWHM"]
    beta = params["beta"]
    wave_list = [params["wave_{}".format(em)] for em in emission_dict]
    G_bkg = params["Gauss_bkg"]
    G_grad = params["Gauss_grad"]
    #G_curve = params["Gauss_curve"]
          
    # Use Moffat function to return array of fluxes for each emission line's amplitude.
    flux_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0, x_2D, y_2D) for A in amp_2D_list])
    
    # calculate 1D Gaussian standard deviation from FWHM
    gauss_std = LSF / 2.35482 # LSF

    # Convert flux to amplitude
    amp_xy = np.array([F / (np.sqrt(2*np.pi) * gauss_std) for F in flux_xy])

    def gauss(Amp_1D, wave):
        return np.array([(G_bkg + G_grad*lam) + A * np.exp(- 0.5 * (lam - wave)** 2 / gauss_std**2.) for A in Amp_1D])

    model_spectra = np.sum(np.array([gauss(A, w) for A,w in zip(amp_xy, wave_list)]),0) # sum up each emission cube to construct integrated model.
    
    return model_spectra, [np.max(amp_xy[0]), flux_xy, amp_xy, model_spectra]


def PNe_residuals_3D(params, l, x_2D, y_2D, data, error, emission_dict, list_to_append_data):
    """Function used to evaluate the residual array for 3D modelling of PNe emission lines. Here, we also extract useful information from the model, including maximum amplitude, amplitude and flux array, and the model itself.

    Parameters
    ----------
    params : dict
        LMfit parameter dictionary object.
    l : list / array
        wavelength array for fitting the emission lines in.
    x_2D : list / array
        x array of matrix sized n_pixel x n_pixel.
    y_2D : list / array
        y array of matrix sized n_pixel x n_pixel.
    data : list / array
        List of the residual spectra for each source that we are to fit. This is used in the residual evaluation (data-model / err)
    error : list / array
        associated error for each PNe, used in the weighting of the residual calculation (data-model / err)
    emission_dict : dict
        Dictionary of emission line names and wavelength positions that are to be fitted for. (e.g. [OIII] @ 4959 & 5007 Angstroms)
    list_to_append_data : list
        An empty list to append the useful information from the PNe_3D_fitter function call.

    Returns
    -------
    list / array
        residual function: data - model / err, which is masked for values of zero.
    """
    # Run the 3D fitting model, returning both the model and useful information.
    model, useful_info = PNe_3D_fitter(params, l, x_2D, y_2D, emission_dict)
    # Append the residual (data-model), along with useful information, to the appropriate list.
    list_to_append_data.clear()
    list_to_append_data.append(data-model)
    list_to_append_data.append(useful_info)
    # Some sources may be at the edge of the field of fiew (FOV), and this may have arrays of 0.0. find such pixels and make a mask array.
    zero_mask = np.where(((data[:,0]!=0) & (data[:,-1]!=0)) & (np.isnan(data[:,0])==False))

    
    # return the object function, for LMfit's minimisation routine, which is weighted by the errors and has the zero mask applied.
    return (data[zero_mask]- model[zero_mask]) / error[zero_mask]

def spaxel_by_spaxel(params, lam, data, error, z):
    """Using a Gaussian double peaked model, fit the [OIII] lines at 4959 and 5007 Angstrom,
    found within Stellar continuum subtracted spectra, from MUSE. For use with LMfit minimizer.

    Parameters
    ----------
    params : dict
        LMfit paramter class
    lam : list / array
        wavelength array
    data : list / array
        PNe residual spectra
    error : list / array
        error array for the spectra
    z : float
        Redshift value

    Returns
    -------
    list / array
        object function as residual array: (data - model) / error
    """
    amp = params["Amp"]
    mean = params["wave"]
    FWHM = params["FWHM"]
    bkg = params["Gauss_bkg"]
    grad = params["Gauss_grad"]

    model = Gauss(lam, amp, mean, FWHM, bkg, grad, z)

    residual = (data - model) / error
    return residual

