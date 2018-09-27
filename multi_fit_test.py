import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.io import ascii, fits
from astropy import wcs
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import sep
from photutils import CircularAperture
from IPython.display import display
from MUSE_Models import MUSE_3D_OIII, MUSE_3D_residual, PNextractor, data_cube_y_x

print("read in data")

hdulist = fits.open("FCC277_data/FCC277_emission_cube.fits")
#hdulist_stellar = fits.open("FCC277_data/FCC277_Gandalf_stellar_cube.fits")
raw_data = hdulist[0].data
#raw_data_stellar = hdulist_stellar[0].data
hdr = hdulist[0].header

#full_wavelength = np.exp(lnl[0].data)
#np.save("exported_data/FCC277/wavelength", full_wavelength)
full_wavelength = np.load("exported_data/FCC277/wavelength.npy")
# wavelength[382] = 4940.976676810172
# wavelength[542] = 5090.878263040583
wavelength = full_wavelength#[382:543]

y_data, x_data, n_data = data_cube_y_x(len(raw_data))
#y_data = 442 # hdr["NAXIS2"]
#x_data = 449 # hdr["NAXIS1"]

coordinates = [(n,m) for n in range(x_data) for m in range(y_data)]

z = 0.00547

x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

oo = np.loadtxt('FCC277_data/FCC277_xy_ima_yngoodpixels.txt', skiprows=1)
fit_these = oo[:,3]

raw_data_list = raw_data#[:,382:543]
#raw_data_list_stellar = raw_data_stellar[:, 382:543]

raw_data_list_fitted = np.squeeze(np.where(fit_these == 1))

raw_data_list_for_fit = raw_data_list[raw_data_list_fitted]

print("starting fit")

def Gaussian_1D_res_full(params, l, data, error, spec_num):
    Amp_list = [params["Amp_OIII_5007"], params["Amp_OIII_4959"], params["Amp_Hb"],
                params["Amp_Ha"], params["Amp_NII_1"], params["Amp_NII_2"]]
    wave_list = [params["wave_OIII_5007"], params["wave_OIII_4959"], params["wave_Hb"],
                params["wave_Ha"], params["wave_NII_1"], params["wave_NII_2"]]
    FWHM = params["FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    Gauss_std = FWHM / 2.35482
    model = np.sum([((Gauss_bkg + Gauss_grad * l) + A * np.exp(- 0.5 * (l - w)** 2 / Gauss_std**2.)) for A,w in zip(Amp_list, wave_list)],0)


    list_of_rN[spec_num] = np.std(data - model)
    list_of_residuals[spec_num] = data - model

    return (data - model) / error

params = Parameters()
params.add("Amp_OIII_5007",value=70., min=0.001)
params.add("Amp_OIII_4959",expr="Amp_OIII_5007/3")
params.add("Amp_Hb",value=10., min=0.001)
params.add("Amp_Ha",value=10., min=0.001)
params.add("Amp_NII_1",value=5., min=0.001)
params.add("Amp_NII_2",value=5., min=0.001)

params.add("wave_OIII_5007", value=5035., min=5000., max=5080.)
params.add("wave_OIII_4959", expr="wave_OIII_5007 - 47.9399")
params.add("wave_Hb", expr="wave_OIII_5007 - 145.518 * (1+{0})".format(z))
params.add("wave_Ha", expr="wave_OIII_5007 + 1556.375 * (1+{0})".format(z))
params.add("wave_NII_1", expr="wave_OIII_5007 + 1541.621 * (1+{0})".format(z))
params.add("wave_NII_2", expr="wave_OIII_5007 + 1577.031 * (1+{0})".format(z))

params.add("FWHM", value=2.81, vary=False) # LSF
params.add("Gauss_bkg", value=0.001)#, min=-500., max=500.)
params.add("Gauss_grad", value=0.001)

list_of_std = np.array([np.abs(np.std(spec)) for spec in raw_data_list])
input_errors = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0,len(list_of_std))]

list_of_rN = np.zeros(len(raw_data_list))
list_of_residuals = np.zeros((len(raw_data_list),len(wavelength)))
list_of_residuals_from_fitter = np.zeros((len(raw_data_list),len(wavelength)))
best_fit_A = np.zeros((len(raw_data_list),6))
best_fit_mean = np.zeros((len(raw_data_list),6))

for spec_n in raw_data_list_fitted:
    results = minimize(Gaussian_1D_res_full, params, args=(wavelength, raw_data_list[spec_n], input_errors[spec_n], spec_n), nan_policy="propagate")
    best_fit_A[spec_n] = [results.params["Amp_OIII_5007"], results.params["Amp_OIII_4959"], results.params["Amp_Hb"], results.params["Amp_Ha"], results.params["Amp_NII_1"], results.params["Amp_NII_2"]]
    best_fit_mean[spec_n] = [results.params["wave_OIII_5007"], results.params["wave_OIII_4959"], results.params["wave_Hb"], results.params["wave_Ha"], results.params["wave_NII_1"], results.params["wave_NII_2"]]
    list_of_residuals_from_fitter[spec_n] = results.residual


print("fit complete")

gauss_A_OIII_5007 = [A[0] for A in best_fit_A]
gauss_A_OIII_4959 = [A[1] for A in best_fit_A]
gauss_A_Hb = np.array([A[2] for A in best_fit_A])
gauss_A_Ha = [A[3] for A in best_fit_A]
gauss_A_N_II_1 = [A[4] for A in best_fit_A]
gauss_A_N_II_2 = [A[5] for A in best_fit_A]

A_rN = np.array([A / rN for A,rN in zip(gauss_A_OIII_5007, list_of_rN)])
A_rN_shape = A_rN.reshape(y_data,x_data)

Gauss_F = np.array(gauss_A_OIII_5007) * np.sqrt(2*np.pi) * 1.19
Gauss_F_shape = Gauss_F.reshape(y_data, x_data)

np.save("exported_data/FCC277/best_fit_A_multi", best_fit_A)
