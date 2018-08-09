import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.io import ascii, fits
from astropy.table import Table
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import lmfit
import pandas as pd
from MUSE_Models import MUSE_3D_OIII, MUSE_3D_residual, Gauss_1D_residual, PNextractor, PSF_residuals
from ppxf import robust_sigma

#First load in the relevant data
hdulist = fits.open("FCC167_data/FCC167_OIII_line_center.fits") # Path to data
hdr = hdulist[0].header # extract header from .fits file
raw_data = hdulist[0].data # extract data from .fits file
y_data = hdr["NAXIS2"] # read y and x dimension values from the header
x_data = hdr["NAXIS1"]
wavelength = np.exp(hdr['CRVAL3']+np.arange(hdr["NAXIS3"])*hdr['CDELT3']) # construct wavelength from header data

# swap axes to y,x,wavelength - THIS MAY NO BE NEEDED
raw_data_list = np.array(raw_data).reshape(len(wavelength), x_data*y_data)
raw_data_list = np.swapaxes(raw_data_list, 1, 0)
# Check for nan values
raw_data_cube = raw_data_list.reshape(y_data, x_data, len(wavelength))

# constants
n_pixels= 13
z = 0.006261 # read from header?
c = 299792458.0 # speed of light
D = 18.7

coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

flatten = lambda l: [item for sublist in l for item in sublist] # flatten command

#Run 1D fit of the spectra and save relevant outputs
## potentially check to see if 1D needs to be run, or load from files with an input() call

check_for_1D_fit = input("Do you want to run the 1D fitter?: (y/n)")

if check_for_1D_fit == "y":
    # Run 1D fitter
    list_of_std = np.array([np.abs(np.std(spec)) for spec in raw_data_list])
    input_errors = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0,len(list_of_std))]
    # setup numpy arrays for storage
    best_fit_A = np.zeros((len(raw_data_list),2))
    list_of_rN = np.zeros(len(raw_data_list))
    data_residuals = np.zeros((len(raw_data_list),len(wavelength)))
    obj_residuals = np.zeros((len(raw_data_list),len(wavelength)))
    # setup LMfit paramterts
    params = Parameters()
    params.add("Amp",value=50., min=0.001, max=300.)
    params.add("mean", value=5035., min=5000., max=5070.)
    params.add("FWHM", value=2.81, vary=False) # Line Spread Function LSF
    params.add("Gauss_bkg", value=0.001, min=-500., max=500.)
    params.add("Gauss_grad", value=0.001)

    for i, spectra in enumerate(raw_data_list):
        fit_results = minimize(Gaussian_1D_res, params, args=(wavelength, spectra, input_errors[i], i), nan_policy="propagate")
        best_fit_A[i] = [results.params["Amp"], results.params["Amp"].stderr]
        list_of_residuals_from_fitter[i] = results.residual

    gauss_A = [A[0] for A in best_fit_A]
    A_err = [A[1] for A in best_fit_A]
    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    Gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

    np.save("exported_data/FCC167/A_rN_cen", A_rN)
    np.save("exported_data/FCC167/gauss_A_cen", gauss_A)
    np.save("exported_data/FCC167/gauss_A_err_cen", A_err)
    np.save("exported_data/FCC167/gauss_F_cen", Gauss_F)
    np.save("exported_data/FCC167/list_of_resids_min_data", list_of_residuals)
    np.save("exported_data/FCC167/list_of_resids_min_obj", list_of_residuals_from_fitter)
    np.save("exported_data/FCC167/rN", list_of_rN)
    
    print("Cube fitted, data saved.")
    # DETECT PNE here?


elif check_for_1D_fit == "n":
    # load from saved files
    #np.load("exported_data/") # read in data

    x_y_list = np.load("exported_data/FCC167/sep_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])

    # Retrieve the respective spectra for each PNe source
    PNe_spectra = np.array([PNextractor(x, y, n_pixels, raw_data_cube, wave=wavelength, dim=2.0) for x,y in zip(x_PNe, y_PNe)])

    # create Pandas data frame for values
    PNe_df = pd.DataFrame(columns=("PNe number", "Total Flux", "Flux error", "V (km/s)", "m 5007", "M 5007", "M 5007 error","A/rN"))
    PNe_df["PNe number"] = np.arange(1,len(x_PNe)+1)

    # Objective residual
    obj_residual_cube = np.load("exported_data/FCC167/list_of_resids_min_obj.npy")
    obj_residual_cube[obj_residual_cube==np.inf] = 0.01
    obj_residual_cube_shape = obj_residual_cube.reshape(y_data, x_data, len(wavelength))
    PNe_uncertainty = np.array([PNextractor(x, y, n_pixels, obj_residual_cube_shape, wave=wavelength, dim=2) for x,y in zip(x_PNe, y_PNe)])

    obj_error_cube = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength)))

    for p in np.arange(0, len(x_PNe)):
        list_of_std = [np.abs(np.std(spec)) for spec in PNe_uncertainty[p]]
        obj_error_cube[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]

    # Data residual
    residual_cube = np.load("exported_data/FCC167/list_of_resids_min.npy")
    residual_cube[residual_cube==np.inf] = 0.01
    residual_cube_shape = residual_cube.reshape(y_data, x_data, len(wavelength))
    PNe_uncertainty = np.array([PNextractor(x, y, n_pixels, residual_cube_shape, wave=wavelength, dim=2) for x,y in zip(x_PNe, y_PNe)])

    error_cube = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength)))

    for p in np.arange(0, len(x_PNe)):
        list_of_std = [np.abs(np.std(spec)) for spec in PNe_uncertainty[p]]
        error_cube[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]
    
    print("Files loaded.")

        

fit_FWHM = 4.0
fit_beta = 2.5

#run initial 3D fit on selected objects
# LMfit initial parameters
PNe_params = Parameters()
def gen_params(wave=5007, FWHM, beta)
    PNe_params.add('Amp_2D', value=100., min=0.01)
    PNe_params.add('x_0', value=(n_pixels/2.), min=0.01, max=n_pixels)
    PNe_params.add('y_0', value=(n_pixels/2.), min=0.01, max=n_pixels)
    PNe_params.add("M_FWHM", value=FWHM, vary=False)
    PNe_params.add("beta", value=beta, vary=False) #1.46
    PNe_params.add("mean", value=wave, min=wave-40., max=wave+40.)
    PNe_params.add("Gauss_bkg",  value=0.001)
    PNe_params.add("Gauss_grad", value=0.001)

    
# generate parameters with values
gen_params(wave=5035., FWHM=fit_FWHM, beta=fit_beta)

# useful value storage setup
total_Flux = np.zeros(len(x_PNe))
list_of_rN = np.zeros(len(x_PNe))
A_OIII_list = np.zeros(len(x_PNe))
F_OIII_xy_list = np.zeros((len(x_PNe), len(PNe_spectra[0])))
M_amp_list = np.zeros(len(x_PNe))
mean_wave_list = np.zeros(len(x_PNe))
list_of_fit_residuals = np.zeros((len(x_PNe), n_pixels*n_pixels* len(wavelength)))

# error lists
moff_A_err = np.zeros(len(x_PNe))
x_0_err = np.zeros(len(x_PNe))
y_0_err = np.zeros(len(x_PNe))
mean_wave_err = np.zeros(len(x_PNe))
Gauss_bkg_err = np.zeros(len(x_PNe))
Gauss_grad_err = np.zeros(len(x_PNe))

FWHM_list = np.zeros(len(x_PNe))
list_of_x = np.zeros(len(x_PNe))
list_of_y = np.zeros(len(x_PNe))
Gauss_bkg = np.zeros(len(x_PNe))
Gauss_grad = np.zeros(len(x_PNe))

model_2D = "Moffat"
#model_2D = "Gauss"
#model_2D = "Gauss_2"

f = FloatProgress(min=0, max=len(x_PNe), description="Fitting progress", )
display(f)

for PNe_num in np.arange(0, len(x_PNe)):
    useful_stuff = []
    #run minimizer fitting routine
    fit_results = minimize(MUSE_3D_residual, PNe_params, args=(wavelength, x_fit, y_fit, PNe_spectra[PNe_num], error_cube[PNe_num], model_2D, PNe_num, useful_stuff), nan_policy="propagate")
    # Store values in numpy arrays
    if model_2D == "Moffat" or model_2D == "Gauss":
        PNe_df.loc[PNe_num, "Total Flux"] = np.sum(useful_stuff[1][1]) * 1e-20
    if model_2D == "Gauss_2":
        PNe_df.loc[PNe_num, "Total Flux"] = (np.sum(useful_stuff[1][2]) + (0.68 * np.sum(useful_stuff[1][3]))) * 1e-20
    list_of_fit_residuals[PNe_num] = useful_stuff[0]
    #list_of_rN[PNe_num] = np.std(useful_stuff)
    A_OIII_list[PNe_num] = useful_stuff[1][0]
    F_OIII_xy_list[PNe_num] = useful_stuff[1][1]
    M_amp_list[PNe_num] = fit_results.params["Amp_2D"]
    list_of_x[PNe_num] = fit_results.params["x_0"]
    list_of_y[PNe_num] = fit_results.params["y_0"]
    mean_wave_list[PNe_num] = fit_results.params["mean"]
    Gauss_bkg[PNe_num] = fit_results.params["Gauss_bkg"]
    Gauss_grad[PNe_num] = fit_results.params["Gauss_grad"]
    #save errors
    moff_A_err[PNe_num] = fit_results.params["Amp_2D"].stderr
    x_0_err[PNe_num] = fit_results.params["x_0"].stderr
    y_0_err[PNe_num] = fit_results.params["y_0"].stderr
    mean_wave_err[PNe_num] = fit_results.params["mean"].stderr
    Gauss_bkg_err[PNe_num] = fit_results.params["Gauss_bkg"].stderr
    Gauss_grad_err[PNe_num] = fit_results.params["Gauss_grad"].stderr
    f.value+=1.

#Apply circular aperture to total flux
Y_circ, X_circ = np.mgrid[:n_pixels, :n_pixels]
#if model_2D == "Moffat":
r = PNe_params["M_FWHM"]
# elif model_2D == "Gauss":
#     r = round(0.75* PNe_params["G_FWHM"])
# elif model_2D == "Gauss_2":
#     r = round(0.75* np.abs(PNe_params["G_FWHM_2"]))
for i in np.arange(0, len(x_PNe)):
    circ_mask = (Y_circ-list_of_y[i])**2 + (X_circ-list_of_x[i])**2 > r*r
    flux_n = np.array(F_OIII_xy_list[i]) # copy list of fluxes
    flux_2D = flux_n.reshape(n_pixels, n_pixels) #reshape
    flux_2D[circ_mask==True] = 0.0 # set mask = False areas to 0.0
    PNe_df.loc[i, "Total Flux"] = np.sum(flux_2D) * 1e-20

# Signal to noise and Magnitude calculations
list_of_rN = np.std(list_of_fit_residuals, 1)
A_by_rN = A_OIII_list / list_of_rN
PNe_df["A/rN"] = A_by_rN

de_z_means = mean_wave_list / (1 + z)

PNe_df["V (km/s)"] = (c * (de_z_means - 5007.) / 5007.) / 1000.

def log_10(x):
    return np.log10(x)

PNe_df["m 5007"] = -2.5 * PNe_df["Total Flux"].apply(log_10) - 13.74
dM =  5. * np.log10(D) + 25   # 31.63
PNe_df["M 5007"] = PNe_df["m 5007"] - dM

#Plotting
plt.figure(1, figsize=(12,10))
plt.axvline(-4.5, color="k", ls="dashed")
info = plt.hist(PNe_df["M 5007"].loc[PNe_df["A/rN"]>2], bins=10, edgecolor="black", linewidth=0.8, label="M 5007 >2 * A/rN", alpha=0.5)
plt.xlim(-5,0)
#plt.title("Absolute Magnitude Histogram", fontsize=24)
plt.xlabel("$M_{5007}$", fontsize=24)
plt.ylabel("N Sources", fontsize=24)
plt.savefig("Plots/FCC167/M5007_histogram.png")
bins_cens = info[1][:-1]

# Run PSF fit using objective residuals




#determine PSF values and feed back into 3D fitter

#Fit PNe with updated PSF

#Run the rest of the analysis
