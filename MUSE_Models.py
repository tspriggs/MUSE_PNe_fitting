import numpy as np
import matplotlib.pyplot as plt

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

# Multi wavelength analysis model
def MUSE_3D_OIII_multi_wave(params, l, x_2D, y_2D, data):
    # n_lines is a dictionary of wavelengths with offsets, read from this and make models with offsets
    Amp_2D_list = [params['Amp_2D_OIII_5007'], params['Amp_2D_OIII_4959'], params['Amp_2D_Hb'], 
                   params['Amp_2D_Ha'], params['Amp_2D_NII_1'], params['Amp_2D_NII_2']]
    x_0 = params['x_0']
    y_0 = params['y_0']
    M_FWHM = params["M_FWHM"]
    beta = params["beta"]
    wave_list = [params["wave_OIII_5007"], params["wave_OIII_4959"], params["wave_Hb"], 
                 params["wave_Ha"], params["wave_NII_1"], params["wave_NII_2"]]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    #Moffat model
    def Moffat(Amp, FWHM, b, x, y):
        gamma = FWHM / (2. * np.sqrt(2.**(1./b) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / gamma**2.
        return Amp * ((1 + rr_gg)**(-b))
    
    F_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0) for A in Amp_2D_list])
    
    # 1D Gaussian standard deviation from FWHM
    Gauss_std = 2.81 / 2.35482

    # Convert flux to amplitude
    A_xy = np.array([F / (np.sqrt(2*np.pi) * Gauss_std) for F in F_xy])
    
    def Gauss(Amp_1D, wave):
        model = np.array([(Gauss_bkg + (Gauss_grad * l)) + A * np.exp(- 0.5 * (l - wave)** 2 / Gauss_std**2.) for A in Amp_1D])
        return model
    
    model_spectra = np.sum(np.array([Gauss(A, w) for A,w in zip(A_xy, wave_list)]),0)

    return model_spectra, [F_xy, A_xy, model_spectra]
# end dev

def MUSE_3D_residual(params, l, x_2D, y_2D, data, error, model_2D, PNe_number, list_to_append_data):
    #model = MUSE_3D_OIII(params, l, x_2D, y_2D, data, model_2D )
    model = MUSE_3D_OIII_multi_wave(params, l, x_2D, y_2D, data)
    list_to_append_data.clear()
    list_to_append_data.append(data-model[0])
    list_to_append_data.append(model[1])

    return (data - model[0]) / error  #(data - model[0]) / (error)

def Gaussian_1D_res(params, x, data, error, spec_num):
    Amp = params["Amp"]
    mean = params["mean"]
    FWHM = params["FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    Gauss_std = FWHM / 2.35482
    model = ((Gauss_bkg + Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - mean)** 2 / Gauss_std**2.) +
             (Amp/3.) * np.exp(- 0.5 * (x - (mean - 47.9399))** 2 / Gauss_std**2.))

    return (data - model) / error

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


def PSF_residuals(PSF_params, l, x_2D, y_2D, data, err):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]

    def generate_model(x, y, moffat_amp, FWHM, beta, Gauss_bkg, Gauss_grad, mean):
        gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2.) / gamma**2.
        F_OIII_xy = moffat_amp * (1. + rr_gg)**(-beta)

        Gauss_std = 2.81 / 2.35482

        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))

        model_spectra = [(Gauss_bkg + (Gauss_grad * l) + np.abs(Amp) * np.exp(- 0.5 * (l - mean)** 2. / Gauss_std**2.) +
             (np.abs(Amp)/3.) * np.exp(- 0.5 * (l - (mean - 47.9399))** 2. / Gauss_std**2.)) for Amp in A_OIII_xy]

        return model_spectra

    list_of_models = {}
    for k in np.arange(0, len(data)):
        list_of_models["model_{:03d}".format(k)] = generate_model(PSF_params["x_{:03d}".format(k)], PSF_params["y_{:03d}".format(k)],
                                                                  PSF_params["moffat_amp_{:03d}".format(k)], FWHM, beta,
                                                                  PSF_params["gauss_grad_{:03d}".format(k)], PSF_params["gauss_bkg_{:03d}".format(k)], 
                                                                  PSF_params["mean_{:03d}".format(k)])

    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - list_of_models["model_{:03d}".format(m)]) / err[m]) 

    if len(resid) > 1.:
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    else:
        return resid["resid_000"]


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
    return int(val), int(val2), n