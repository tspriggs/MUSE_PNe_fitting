import numpy as np

def MUSE_3D_OIII(params, l, x_2D, y_2D, data, model_2D):
    Amp_2D = params['Amp_2D']
    #Amp_2 = params["Amp_2"]
    x_0 = params['x_0']
    y_0 = params['y_0']
    M_FWHM = params["M_FWHM"]
    #G_FWHM = params["G_FWHM"]
    #G_FWHM_2 = params["G_FWHM_2"]
    beta = params["beta"]
    mean = params["mean"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]
    
    if model_2D == "Moffat":
        #Moffat model
        gamma = M_FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x_0)**2. + (y_2D - y_0)**2) / gamma**2.
        F_OIII_xy = Amp_2D * ((1 + rr_gg)**(-beta))
    elif model_2D == "Gauss":
        sigma_2D = G_FWHM / 2.3548
        F_OIII_xy = Amp_2D * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_2D**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_2D**2.))))
    elif model_2D == "Gauss_2":
        sigma_1 = G_FWHM / 2.3548
        sigma_2 = G_FWHM_2 / 2.3548
        Gauss_1 = Amp_2D * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_1**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_1**2.))))
        Gauss_2 = Amp_2 * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_2**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_2**2.))))
        F_OIII_xy = Gauss_1 + Gauss_2
    
    # 1D Gaussian standard deviation from FWHM
    Gauss_std = 2.81 / 2.35482
    
    # Convert flux to amplitude
    A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))
    
    #Construct model gaussian profiles for each amplitude value in cube
    model_spectra = [(Gauss_bkg + (Gauss_grad * l) + Amp * np.exp(- 0.5 * (l - mean)** 2 / Gauss_std**2.) +
             (Amp/3.) * np.exp(- 0.5 * (l - (mean - 47.9399))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]
    
    if model_2D == "Moffat" or model_2D == "Gauss":
        return model_spectra, [np.max(A_OIII_xy), F_OIII_xy, A_OIII_xy]
    elif model_2D == "Gauss_2":
        return model_spectra, [np.max(A_OIII_xy), F_OIII_xy, A_OIII_xy, Gauss_1, Gauss_2]


def MUSE_3D_residual(params, l, x_2D, y_2D, data, error, model_2D, PNe_number, list_to_append_data):
    model = MUSE_3D_OIII(params, l, x_2D, y_2D, data, model_2D )
    list_to_append_data.clear()
    list_to_append_data.append(np.std(data - model[0]))
    list_to_append_data.append(data-model[0])
    list_to_append_data.append(model[1])

    return (data - model[0]) / (error)

### Dev

def MUSE_3D_OIII_sum(params, l, x_2D, y_2D, data, model_2D):
    Amp_2D = params['Amp_2D']
    Amp_2 = params["Amp_2"]
    x_0 = params['x_0']
    y_0 = params['y_0']
    M_FWHM = params["M_FWHM"]
    G_FWHM = params["G_FWHM"]
    G_FWHM_2 = params["G_FWHM_2"]
    beta = params["beta"]
    mean = params["mean"]
    Gauss_FWHM = params["Gauss_FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]
    
    if model_2D == "Moffat":
        #Moffat model
        gamma = M_FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x_0)**2. + (y_2D - y_0)**2) / gamma**2.
        F_OIII_xy = Amp_2D * ((1 + rr_gg)**(-beta))
    elif model_2D == "Gauss":
        sigma_2D = G_FWHM / 2.3548
        F_OIII_xy = Amp_2D * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_2D**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_2D**2.))))
    elif model_2D == "Gauss_2":
        sigma_1 = G_FWHM / 2.3548
        sigma_2 = G_FWHM_2 / 2.3548
        Gauss_1 = Amp_2D * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_1**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_1**2.))))
        Gauss_2 = Amp_2 * np.exp(-1. * ((((np.array(x_2D) - x_0)**2)/(2*sigma_2**2.)) + (((np.array(y_2D) - y_0)**2.)/(2.*sigma_2**2.))))
        F_OIII_xy = Gauss_1 + Gauss_2
    
    # 1D Gaussian standard deviation from FWHM
    Gauss_std = np.sqrt(2.81**2 + Gauss_FWHM**2) / 2.35482
    
    # Convert Moffat flux to amplitude
    A_OIII_xy = (np.sum(F_OIII_xy, 0) / (np.sqrt(2*np.pi) * Gauss_std))
    
    #Construct model gaussian profiles for each amplitude value in cube
    model_spectra = ((Gauss_bkg + (Gauss_grad * l)) + A_OIII_xy * np.exp(- 0.5 * (l - mean)** 2 / Gauss_std**2.) + 
                    (A_OIII_xy/3.) * np.exp(- 0.5 * (l - (mean - 47.9399))** 2 / Gauss_std**2.))
        
    return model_spectra, [np.max(A_OIII_xy), F_OIII_xy]


def MUSE_3D_residual_sum(params, l, x_2D, y_2D, data, error, model_2D, PNe_number, list_to_append_data):
    model = MUSE_3D_OIII_sum(params, l, x_2D, y_2D, data, model_2D )
    list_to_append_data.clear()
    list_to_append_data.append(np.std(data - model[0]))
    list_to_append_data.append(model[1])

    return (np.sum(data,0) - model[0]) #/ np.std(np.sum(data,0))

### Dev End

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


def PSF_residuals(PSF_params, l, x_2D, y_2D, data, err, A_rN):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]
    Gauss_FWHM = PSF_params["Gauss_FWHM"]
    
    def generate_model(x, y, moffat_amp, FWHM, beta, Gauss_FWHM, Gauss_bkg, Gauss_grad, mean):
        gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2.) / gamma**2.
        F_OIII_xy = moffat_amp * (1. + rr_gg)**(-beta)
        
        comb_FWHM = np.sqrt(2.81**2. + Gauss_FWHM**2.)
        Gauss_std = comb_FWHM / 2.35482
        
        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))
        
        model_spectra = [(Gauss_bkg + (Gauss_grad * l) + np.abs(Amp) * np.exp(- 0.5 * (l - mean)** 2. / Gauss_std**2.) +
             (np.abs(Amp)/3.) * np.exp(- 0.5 * (l - (mean - 47.9399))** 2. / Gauss_std**2.)) for Amp in A_OIII_xy]
        
        return model_spectra
    
    list_of_models = {}
    for k in np.arange(0, len(data)):
        list_of_models["model_{:03d}".format(k)] = generate_model(PSF_params["x_{:03d}".format(k)], PSF_params["y_{:03d}".format(k)], PSF_params["moffat_amp_{:03d}".format(k)], 
                            FWHM, beta, Gauss_FWHM, PSF_params["gauss_grad_{:03d}".format(k)], PSF_params["gauss_bkg_{:03d}".format(k)], PSF_params["mean_{:03d}".format(k)])
    
    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - list_of_models["model_{:03d}".format(m)]) / err[m]) *A_rN[m]
    
    if len(resid) > 1.:
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    else:
        return resid["resid_000"]