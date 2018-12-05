import numpy as np
import matplotlib.pyplot as plt

# Multi wavelength analysis model

def MUSE_3D_OIII_multi_wave(params, l, x_2D, y_2D, data, emission_dict):
    # loop through emission dict and append to Amp 2D and wave lists
    Amp_2D_list = [params["Amp_2D_{}".format(em)] for em in emission_dict]
    x_0 = params['x_0']
    y_0 = params['y_0']
    M_FWHM = params["M_FWHM"]
    beta = params["beta"]
    wave_list = [params["wave_{}".format(em)] for em in emission_dict]
    G_bkg = params["Gauss_bkg"]
    G_grad = params["Gauss_grad"]

    # Moffat model
    def Moffat(Amp, FWHM, b, x, y):
        gamma = FWHM / (2. * np.sqrt(2.**(1./b) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2) / gamma**2.
        return Amp * ((1 + rr_gg)**(-b))

    F_xy = np.array([Moffat(A, M_FWHM, beta, x_0, y_0) for A in Amp_2D_list])

    # 1D Gaussian standard deviation from FWHM
    G_std = 2.81 / 2.35482

    # Convert flux to amplitude
    A_xy = np.array([F / (np.sqrt(2*np.pi) * G_std) for F in F_xy])

    def Gauss(Amp_1D, wave):
        return np.array([(G_bkg + (G_grad * l)) + A * np.exp(- 0.5 * (l - wave)** 2 / G_std**2.) for A in Amp_1D])

    model_spectra = np.sum(np.array([Gauss(A, w) for A,w in zip(A_xy, wave_list)]),0)

    return model_spectra, [np.max(A_xy[0]), F_xy, A_xy, model_spectra]

def MUSE_3D_residual(params, l, x_2D, y_2D, data, error, PNe_number, spec, emission_dict, list_to_append_data):
    if spec == "short":
        model = MUSE_3D_OIII(params, l, x_2D, y_2D, data )
    elif spec == "full":
        model = MUSE_3D_OIII_multi_wave(params, l, x_2D, y_2D, data, emission_dict)
    list_to_append_data.clear()
    list_to_append_data.append(data-model[0])
    list_to_append_data.append(model[1])

    return (data - model[0]) / error

def PSF_residuals(PSF_params, l, x_2D, y_2D, data, err):
    FWHM = PSF_params['FWHM']
    beta = PSF_params["beta"]

    def generate_model(x, y, moffat_amp, FWHM, beta, Gauss_bkg, Gauss_grad, wave):
        gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
        rr_gg = ((x_2D - x)**2. + (y_2D - y)**2.) / gamma**2.
        F_OIII_xy = moffat_amp * (1. + rr_gg)**(-beta)

        Gauss_std = 2.81 / 2.35482

        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))

        model_spectra = [((Gauss_bkg + Gauss_grad * l) + Amp * np.exp(- 0.5 * (l - wave)** 2 / Gauss_std**2.) +
             (Amp/3.) * np.exp(- 0.5 * (l - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

        return model_spectra

    list_of_models = {}
    for k in np.arange(0, len(data)):
        list_of_models["model_{:03d}".format(k)] = generate_model(PSF_params["x_{:03d}".format(k)], PSF_params["y_{:03d}".format(k)],
                                                                  PSF_params["moffat_amp_{:03d}".format(k)], FWHM, beta,
                                                                  PSF_params["gauss_grad_{:03d}".format(k)], PSF_params["gauss_bkg_{:03d}".format(k)],
                                                                  PSF_params["wave_{:03d}".format(k)])

    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - list_of_models["model_{:03d}".format(m)]) / err[m])

    if len(resid) > 1.:
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    else:
        return resid["resid_000"]


#def Gaussian_1D_res(params, x, data, error, spec_num):
#    Amp = params["Amp"]
#    wave = params["wave"]
#    FWHM = params["FWHM"]
#    Gauss_bkg = params["Gauss_bkg"]
#    Gauss_grad = params["Gauss_grad"]
#
#    Gauss_std = FWHM / 2.35482
#    return ((Gauss_bkg + Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - wave)** 2 / Gauss_std**2.) +
#             (Amp/3.) * np.exp(- 0.5 * (x - (wave - 47.9399))** 2 / Gauss_std**2.))
#
#    #return (data - model) / error
#
#def MUSE_1D_residual(params, l, data, error, spec_num, list_to_append):
#    model = Gaussian_1D_res(params, l, data, error, spec_num)
#    list_to_append.clear()
#    list_to_append.append(data - model)
#    list_to_append.append(np.std(data - model))
#
#    return (data - model)/ error


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
