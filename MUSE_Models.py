import numpy as np

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

    # Moffat model
    def Moffat(Amp, FWHM, beta, x, y):
        alpha = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / alpha**2.
        return Amp * (2 * ((beta -1)/(alpha**2))) * ((1 + rr_gg)**(-beta))
                
    # Use Moffat function to return array of fluxes for each emission's amplitude.
    F_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0) for A in Amp_2D_list])
    
    # 1D Gaussian standard deviation from FWHM
    G_std = LSF / 2.35482 # LSF

    # Convert flux to amplitude
    A_xy = np.array([F / (np.sqrt(2*np.pi) * G_std) for F in F_xy])

    def Gauss(Amp_1D, wave):
        return np.array([(G_bkg + (G_grad * l) + G_curve*l**2) + A * np.exp(- 0.5 * (l - wave)** 2 / G_std**2.) for A in Amp_1D])

    model_spectra = np.sum(np.array([Gauss(A, w) for A,w in zip(A_xy, wave_list)]),0) # sum up each emission cube to construct integrated model.
    
    return model_spectra, [np.max(A_xy[0]), F_xy, A_xy, model_spectra]

def PNe_residuals_3D(params, l, x_2D, y_2D, data, error, PNe_number, emission_dict, list_to_append_data):
    model, useful_info = PNe_3D_fitter(params, l, x_2D, y_2D, data, emission_dict)
    list_to_append_data.clear()
    list_to_append_data.append(data-model)
    list_to_append_data.append(useful_info)    
    zero_mask = np.where(data[:,0]!=0)
    #signal_cut = (useful_info[2][0]/robust_sigma(data[:50]))>=2
    #list_to_append_data.append(signal_cut)
    #model[signal_cut==False] = np.zeros_like(len(l))
    
    return (data[zero_mask]- model[zero_mask]) / error[zero_mask]
#     return (np.sum(data[zero_mask], 0) - np.sum(model[zero_mask], 0)) / np.sum(error[zero_mask], 0)

def PSF_residuals_3D(PSF_params, l, x_2D, y_2D, data, err, z):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]
    LSF = PSF_params["LSF"]
    
    def gen_model(x, y, moffat_amp, FWHM, beta, g_LSF, Gauss_bkg, Gauss_grad, wave, z):
        alpha = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2.) / alpha**2.
        F_OIII_xy = moffat_amp *(2 * ((beta -1)/(alpha**2))) * (1. + rr_gg)**(-beta)

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
                                                       PSF_params["wave_{:03d}".format(k)], z)
    
    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - models["model_{:03d}".format(m)]) / err[m])
    
    if len(resid) > 1.:
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    else:
        return resid["resid_000"]


def PNextractor(x, y, n_pix, data, wave=None, dim=1):
    x = round(x)
    y = round(y)
    offset = n_pix // 2
    # select the spectra of interest
    from_data = data[int(y - offset):int(y - offset + n_pix), int(x - offset):int(x - offset + n_pix)]
    if dim == 1:
        return from_data.reshape(n_pix**2)
    if dim == 2:
        return from_data.reshape(n_pix**2, len(wave))


def PNe_spectrum_extractor(x, y, n_pix, data, x_d, wave):
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