import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.io import ascii, fits
from astropy.table import Table
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import lmfit
import pandas as pd
from MUSE_Models import MUSE_3D_OIII, MUSE_3D_residual, Gauss_1D_residual, PNextractor, PSF_residuals, data_cube_y_x
from ppxf import robust_sigma

#First load in the relevant data
hdulist = fits.open("FCC167_data/FCC167_OIII_line_center.fits") # Path to data
hdr = hdulist[0].header # extract header from .fits file
raw_data = hdulist[0].data # extract data from .fits file

y_data, x_data, n_data = data_cube_y_x(len(raw_data))

#y_data = hdr["NAXIS2"] # read y and x dimension values from the header
#x_data = hdr["NAXIS1"]
#wavelength = np.exp(hdr['CRVAL3']+np.arange(hdr["NAXIS3"])*hdr['CDELT3']) # construct wavelength from header data
full_wavelength = np.load("exported_data/galaxy/wavelength.npy")

# swap axes to y,x,wavelength - THIS MAY NO BE NEEDED
#raw_data_list = np.array(raw_data).reshape(len(wavelength), x_data*y_data)
#raw_data_list = np.swapaxes(raw_data_list, 1, 0)
# Check for nan values
raw_data_cube = raw_data_list.reshape(y_data, x_data, len(wavelength))

non_zero_index = np.squeeze(np.where(raw_data_list[:,0] != 0.))

# constants
n_pixels= 13
c = 299792458.0 # speed of light

z = 0.006261 # read from header?
D = 18.7 # Distance

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
    params.add("Amp",value=70., min=0.001, max=500.)
    params.add("mean", value=5030., min=5000., max=5080.)
    params.add("FWHM", value=2.81, vary=False) # Line Spread Function LSF
    params.add("Gauss_bkg", value=0.001, min=-500., max=500.)
    params.add("Gauss_grad", value=0.001)

    for i in non_zero_index:
        fit_results = minimize(Gaussian_1D_res, params, args=(wavelength, raw_data_list[i], input_errors[i], i), nan_policy="propagate")
        best_fit_A[i] = [results.params["Amp"], results.params["Amp"].stderr]
        obj_residuals[i] = results.residual

    gauss_A = [A[0] for A in best_fit_A]
    A_err = [A[1] for A in best_fit_A]
    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    Gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

    np.save("exported_data/FCC167/A_rN_cen", A_rN)
    np.save("exported_data/FCC167/gauss_A_cen", gauss_A)
    np.save("exported_data/FCC167/gauss_A_err_cen", A_err)
    np.save("exported_data/FCC167/gauss_F_cen", Gauss_F)
    np.save("exported_data/FCC167/list_of_resids_min_data", data_residuals)
    np.save("exported_data/FCC167/list_of_resids_min_obj", obj_residuals)
    np.save("exported_data/FCC167/rN", list_of_rN)
    
    print("Cube fitted, data saved.")
    # DETECT PNE here?


elif check_for_1D_fit == "n":
    # load from saved files
    #np.load("exported_data/") # read in data

    x_y_list = np.load("exported_data/ galaxy /sep_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])

    # Retrieve the respective spectra for each PNe source
    PNe_spectra = np.array([PNextractor(x, y, n_pixels, raw_data_cube, wave=wavelength, dim=2.0) for x,y in zip(x_PNe, y_PNe)])

    # create Pandas data frame for values
    PNe_df = pd.DataFrame(columns=("PNe number", "Total Flux", "Flux error", "V (km/s)", "m 5007", "M 5007", "M 5007 error","A/rN"))
    PNe_df["PNe number"] = np.arange(1,len(x_PNe)+1)

    # Objective Residual Cube
    obj_residual_cube = np.load("exported_data/FCC255/list_of_resids_min_obj.npy")
    
    # Data Residual Cube
    residual_cube = np.load("exported_data/FCC255/list_of_resids_min.npy")
    
    def uncertainty_cube_construct(data, x_P, y_P, n_pix):
        data[data == np.inf] = 0.01
        data_shape = data.reshape(y_data, x_data, len(wavelength))
        extract_data = np.array([PNextractor(x, y, n_pix, data_shape, wave=wavelength, dim=2) for x,y in zip(x_P, y_P)])
        array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(wavelength)))
        for p in np.arange(0, len(x_P)):
            list_of_std = [np.abs(np.std(spec)) for spec in extract_data[p]]
            array_to_fill[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]
      
        return array_to_fill
    
    error_cube = uncertainty_cube_construct(residual_cube, x_PNe, y_PNe, n_pixels)
    obj_error_cube = uncertainty_cube_construct(obj_residual_cube, x_PNe, y_PNe, n_pixels)
    
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
    PNe_params.add("beta", value=beta, vary=False)
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


f = FloatProgress(min=0, max=len(x_PNe), description="Fitting progress", )
display(f)

for PNe_num in np.arange(0, len(x_PNe)):
    useful_stuff = []
    #run minimizer fitting routine
    fit_results = minimize(MUSE_3D_residual, PNe_params, args=(wavelength, x_fit, y_fit, PNe_spectra[PNe_num], error_cube[PNe_num], model_2D, PNe_num, useful_stuff), nan_policy="propagate")
    # Store values in numpy arrays
    PNe_df.loc[PNe_num, "Total Flux"] = np.sum(useful_stuff[1][1]) * 1e-20
    list_of_fit_residuals[PNe_num] = useful_stuff[0]
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

# Signal to noise and Magnitude calculations
list_of_rN = np.array([np.std(PNe_res) for PNe_res in list_of_fit_residuals])
A_by_rN = A_OIII_list / list_of_rN
PNe_df["A/rN"] = A_by_rN

de_z_means = mean_wave_list / (1 + z)

PNe_df["V (km/s)"] = (c * (de_z_means - 5007.) / 5007.) / 1000.

def log_10(x):
    return np.log10(x)

PNe_df["m 5007"] = -2.5 * PNe_df["Total Flux"].apply(log_10) - 13.74
dM =  5. * np.log10(D) + 25   # 31.63
PNe_df["M 5007"] = PNe_df["m 5007"] - dM

plt.figure(1, figsize=(12,10))
bins, bins_cens, other = plt.hist(PNe_df["m 5007"].loc[PNe_df["A/rN"]>2], bins=10, edgecolor="black", linewidth=0.8, label="m 5007 > 2 * A/rN", alpha=0.5)
plt.xlim(26.0,30.0)
plt.xlabel("$m_{5007}$", fontsize=24)
plt.ylabel("N Sources", fontsize=24)
#plt.legend(fontsize=15)
#plt.savefig("Plots/ galaxy /M5007_histogram.png")
bins_cens = bins_cens[:-1]

# Run PSF fit using objective residuals

PSF_choose = input("Use Brightest PNe? (Y/N)")
if PSF_choose == "Y":
    sel_PNe = PNe_df.nlargest(1, "A/rN").index.values
elif PSF_choose == "N":
    # Devise system for PNe choise based upon low background (radial?)
    #sel_PNe = [0,3]#,62]#63,26]#[ 28]#, 29]
print(sel_PNe)

selected_PNe = PNe_spectra[sel_PNe]
selected_PNe_err = obj_error_cube[sel_PNe] 
PSF_params = Parameters()

def model_params(p, n, amp, mean):
    PSF_params.add("moffat_amp_{:03d}".format(n), value=amp, min=0.001)
    PSF_params.add("x_{:03d}".format(n), value=n_pixels/2., min=0.001, max=n_pixels)
    PSF_params.add("y_{:03d}".format(n), value=n_pixels/2., min=0.001, max=n_pixels)
    PSF_params.add("mean_{:03d}".format(n), value=mean, min=5000., max=5070.)
    PSF_params.add("gauss_bkg_{:03d}".format(n), value=0.001)
    PSF_params.add("gauss_grad_{:03d}".format(n), value=0.001)


for i in np.arange(0,len(sel_PNe)):
        model_params(p=PSF_params, n=i, amp=200.0, mean=5030.0)    
    
PSF_params.add('FWHM', value=4.0, min=0.01, max=12., vary=True)
PSF_params.add("beta", value=4.0, min=0.01, max=12., vary=True) 
PSF_params.add("Gauss_FWHM", value=0.00000, min=0.000001, max=3.0, vary=False) # LSF, instrumental resolution.

PSF_results = minimize(PSF_residuals, PSF_params, args=(wavelength, x_fit, y_fit, selected_PNe, selected_PNe_err, np.ones(len(sel_PNe))), nan_policy="propagate")




#determine PSF values and feed back into 3D fitter

#Fit PNe with updated PSF

#Run the rest of the analysis
