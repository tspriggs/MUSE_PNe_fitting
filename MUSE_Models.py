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

    # Moffat model
    def Moffat(Amp, FWHM, beta, x, y):
        r_d = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / r_d**2.
        return Amp * ((1 + rr_gg)**(-beta))

    # Use Moffat function to return array of fluxes for each emission's amplitude.
    F_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0) for A in Amp_2D_list])

    # 1D Gaussian standard deviation from FWHM
    G_std = LSF / 2.35482 # LSF

    # Convert flux to amplitude
    A_xy = np.array([F / (np.sqrt(2*np.pi) * G_std) for F in F_xy])

    def Gauss(Amp_1D, wave):
        return np.array([(G_bkg + (G_grad * l)) + A * np.exp(- 0.5 * (l - wave)** 2 / G_std**2.) for A in Amp_1D])

    model_spectra = np.sum(np.array([Gauss(A, w) for A,w in zip(A_xy, wave_list)]),0) # sum up each emission cube to construct integrated model.
    
    return model_spectra, [np.max(A_xy[0]), F_xy, A_xy, model_spectra]

def PNe_residuals_3D(params, l, x_2D, y_2D, data, error, PNe_number, emission_dict, list_to_append_data):
    model = PNe_3D_fitter(params, l, x_2D, y_2D, data, emission_dict)
    list_to_append_data.clear()
    list_to_append_data.append(data-model[0])
    list_to_append_data.append(model[1])    
    zero_mask = np.where(data[:,0]!=0)
    
    return (data[zero_mask] - model[0][zero_mask]) / error[zero_mask]

def PSF_residuals_3D(PSF_params, l, x_2D, y_2D, data, err, z):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]
    LSF = PSF_params["LSF"]
    zero_mask = np.where(data[:,0]!=0)
    
    def gen_model(x, y, moffat_amp, FWHM, beta, LSF, Gauss_bkg, Gauss_grad, wave, z):
        gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2.) / gamma**2.
        F_OIII_xy = moffat_amp * (1. + rr_gg)**(-beta)

        Gauss_std = LSF / 2.35482 # LSF

        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))

        model_spectra = (Gauss_bkg + Gauss_grad * l) + [(Amp * np.exp(- 0.5 * (l - wave)** 2 / Gauss_std**2.) +
             (Amp/3.) * np.exp(- 0.5 * (l - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

        return model_spectra

    models = {}
    for k in np.arange(0, len(data[zero_mask])):
        models["model_{:03d}".format(k)] = gen_model(PSF_params["x_{:03d}".format(k)], PSF_params["y_{:03d}".format(k)],
                                                       PSF_params["moffat_amp_{:03d}".format(k)], FWHM, beta, LSF,
                                                       PSF_params["gauss_grad_{:03d}".format(k)], PSF_params["gauss_bkg_{:03d}".format(k)],
                                                       PSF_params["wave_{:03d}".format(k)], z)
    
    resid = {}
    for m in np.arange(0, len(data[zero_mask])):
        resid["resid_{:03d}".format(m)] = ((data[zero_mask][m] - models["model_{:03d}".format(m)][zero_mask]) / err[m][zero_mask])
    
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


def data_cube_y_x(n):
    nsqrt = np.ceil(np.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val-=1

    if int(val) > int(val2):
       return int(val2), int(val), n
    else:
       return int(val), int(val2), n
