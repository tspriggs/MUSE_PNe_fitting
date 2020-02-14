import numpy as np


# Moffat model function
def Moffat(Amp, FWHM, beta, x, y, x_2D, y_2D):
        alpha = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / alpha**2.
        return Amp * (2 * ((beta -1)/(alpha**2))) * ((1 + rr_gg)**(-beta))

    
# Multi wavelength analysis model
def PNe_3D_fitter(params, l, x_2D, y_2D, data, emission_dict):
    # loop through emission dict and append to Amp 2D and wave lists
    Amp_2D_list = [params["Amp_2D_{}".format(em)] for em in emission_dict]
    x_0 = params['x_0']
    y_0 = params['y_0']
    LSF = params["LSF"]
    M_FWHM = params["M_FWHM"]
    beta = params["beta"]
    wave_list = [params["wave_{}".format(em)] for em in emission_dict]
    G_bkg = params["Gauss_bkg"]
    G_grad = params["Gauss_grad"]
    G_curve = params["Gauss_curve"]
          
    # Use Moffat function to return array of fluxes for each emission line's amplitude.
    flux_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0, x_2D, y_2D) for A in Amp_2D_list])
    
    # calculate 1D Gaussian standard deviation from FWHM
    gauss_std = LSF / 2.35482 # LSF

    # Convert flux to amplitude
    amp_xy = np.array([F / (np.sqrt(2*np.pi) * gauss_std) for F in flux_xy])

    def gauss(Amp_1D, wave):
        return np.array([(G_bkg + (G_grad * l) + (G_curve*l**2)) + A * np.exp(- 0.5 * (l - wave)** 2 / gauss_std**2.) for A in Amp_1D])

    model_spectra = np.sum(np.array([gauss(A, w) for A,w in zip(amp_xy, wave_list)]),0) # sum up each emission cube to construct integrated model.
    
    return model_spectra, [np.max(amp_xy[0]), flux_xy, amp_xy, model_spectra]


def PNe_residuals_3D(params, l, x_2D, y_2D, data, error, PNe_number, emission_dict, list_to_append_data):
    model, useful_info = PNe_3D_fitter(params, l, x_2D, y_2D, data, emission_dict)
    list_to_append_data.clear()
    list_to_append_data.append(data-model)
    list_to_append_data.append(useful_info)    
    zero_mask = np.where(data[:,0]!=0)
    
    return (data[zero_mask]- model[zero_mask]) / error[zero_mask]


def PSF_residuals_3D(PSF_params, l, x_2D, y_2D, data, err, z):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]
    LSF = PSF_params["LSF"]
    
    def gen_model(x, y, moffat_amp, FWHM, beta, g_LSF, Gauss_bkg, Gauss_grad, wave, z, x_2D, y_2D):
        F_OIII_xy = Moffat(A, FWHM, beta, x_0, y_0, x_2D, y_2D)
        
        Gauss_std = g_LSF / 2.35482 # LSF

        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))

        model_spectra = (Gauss_bkg + Gauss_grad * l) + [(Amp * np.exp(- 0.5 * (l - wave)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (l - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

        return model_spectra

    models = {}
    for k in np.arange(0, len(data)):
        models["model_{:03d}".format(k)] = gen_model(PSF_params["x_{:03d}".format(k)], PSF_params["y_{:03d}".format(k)],
                                                       PSF_params["moffat_amp_{:03d}".format(k)], FWHM, beta, LSF,
                                                       PSF_params["gauss_grad_{:03d}".format(k)], PSF_params["gauss_bkg_{:03d}".format(k)],
                                                       PSF_params["wave_{:03d}".format(k)], z, x_2D, y_2D)
    
    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - models["model_{:03d}".format(m)]) / err[m])
    
    if len(resid) > 1.:
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    else:
        return resid["resid_000"]


def spaxel_by_spaxel(params, x, data, error, spec_num, z, data_residuals):
    """
    Using a Gaussian double peaked model, fit the [OIII] lines at 4959 and 5007 Angstrom, found within Stellar continuum subtracted spectra, from MUSE.
    Inputs:
        Params - Using the LMfit python package, contruct the parameters needed and read them in:
                Amplitude of [OIII] at 5007 A.
                mean wavelength position of [OIII] 5007 A peak.
                FWHM of Gaussian profiles.
                Gaussian backrgound level of residuals.
                Gaussian gradient of background residuals.
        x - Wavelength array
        data - read in sprectrum by spectrum of data via list form.
        error - associated errors for each spectrum.
        spec_num - from enumerate, just the index number of spectrum, for storing value sin np array.

    Returns -  (Data - model) / error   for chi square minimiser.
    """
    Amp = params["Amp"]
    wave = params["wave"]
    FWHM = params["FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    Gauss_std = FWHM / 2.35482 # FWHM to Standard Deviation calculation.

    model = ((Gauss_bkg + Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - wave)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (x - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.))

    # Saves both the Residual noise level of the fit, alongside the 'data residual' (data-model) array from the fit.
#     list_of_rN[spec_num] = robust_sigma(data - model)
#     list_of_model[spec_num] = model
    data_residuals[spec_num] = data - model

    return (data - model) / error

