import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from astropy.io import ascii, fits
from astropy.table import Table
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import yaml
import pandas as pd
from MUSE_Models import MUSE_3D_OIII, MUSE_3D_residual, Gaussian_1D_res, PNextractor, PSF_residuals, data_cube_y_x

with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data)
    
choose_galaxy = input("Please type which Galaxy you want to analyse, use FCC000 format: ")
galaxy_data = galaxy_info[choose_galaxy]

#First load in the relevant data
hdulist = fits.open(galaxy_data["emission cube"]) # Path to data
hdr = hdulist[0].header # extract header from .fits file
raw_data = hdulist[0].data # extract data from .fits file
raw_data_list = raw_data
#raw_data_list_short = raw_data[:,382:543]

y_data, x_data, n_data = data_cube_y_x(len(raw_data))

#y_data = hdr["NAXIS2"] # read y and x dimension values from the header
#x_data = hdr["NAXIS1"]
#wavelength = np.exp(hdr['CRVAL3']+np.arange(hdr["NAXIS3"])*hdr['CDELT3']) # construct wavelength from header data
full_wavelength = np.load(galaxy_data["wavelength"])
#wavelength = full_wavelength[382:543]

# Reshape data into y, x and wavelength
raw_data_cube = raw_data_list.reshape(y_data, x_data, len(full_wavelength))

# create an list of indices where there is spectral data to fit.
non_zero_index = np.squeeze(np.where(raw_data_list[:,0] != 0.))

# constants
n_pixels= 13
c = 299792458.0 # speed of light

z = galaxy_data["z"] # Redshift
D = galaxy_data["Distance"] # Distance in Mpc

coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

#Run 1D fit of the spectra and save relevant outputs
## potentially check to see if 1D needs to be run, or load from files with an input() call

check_for_1D_fit = input("Do you want to run the 1D fitter?: (y/n)")

if check_for_1D_fit == "y":
    # Run 1D fitter
    print("Starting 1D fit")
    list_of_std = np.array([np.abs(np.std(spec)) for spec in raw_data_list])
    input_errors = [np.repeat(list_of_std[i], len(full_wavelength)) for i in np.arange(0,len(list_of_std))]
    # setup numpy arrays for storage
    best_fit_A = np.zeros((len(raw_data_list),2))
    list_of_rN = np.zeros(len(raw_data_list))
    data_residuals = np.zeros((len(raw_data_list),len(full_wavelength)))
    obj_residuals = np.zeros((len(raw_data_list),len(full_wavelength)))
    # setup LMfit paramterts
    params = Parameters()
    params.add("Amp",value=70., min=0.001, max=500.)
    params.add("mean", value=galaxy_data["wave start"], 
               min=galaxy_data["wave start"]-40,
               max=galaxy_data["wave start"]+40)
    params.add("FWHM", value=2.81, vary=False) # Line Spread Function
    params.add("Gauss_bkg", value=0.001, min=-500., max=500.)
    params.add("Gauss_grad", value=0.001)

    for i in non_zero_index:
        fit_results = minimize(Gaussian_1D_res, params, args=(full_wavelength, raw_data_list[i], input_errors[i], i), nan_policy="propagate")
        best_fit_A[i] = [results.params["Amp"], results.params["Amp"].stderr]
        obj_residuals[i] = results.residual

    gauss_A = [A[0] for A in best_fit_A]
    A_err = [A[1] for A in best_fit_A]
    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    Gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/A_rN_cen", A_rN)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_A_cen", gauss_A)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_A_err_cen", A_err)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_F_cen", Gauss_F)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/list_of_resids_min_data", data_residuals)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/list_of_resids_min_obj", obj_residuals)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/rN", list_of_rN)
    
    print("Cube fitted, data saved.")
    # DETECT PNE here?


# if n, then fit PNe

print("Starting PNe analysis with initial PSF guess")
# load from saved files

x_y_list = np.load("exported_data/"+ galaxy_data["Galaxy name"] +"/sep_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list])
y_PNe = np.array([y[1] for y in x_y_list])

# Retrieve the respective spectra for each PNe source
PNe_spectra = np.array([PNextractor(x, y, n_pixels, raw_data_cube, wave=full_wavelength, dim=2.0) for x,y in zip(x_PNe, y_PNe)])

# create Pandas data frame for values
PNe_df = pd.DataFrame(columns=("PNe number", "Ra (J2000)", "Dec (J2000)", "[OIII] Flux", "[OIII]/Hb", "Flux error", "V (km/s)", "m 5007", "M 5007", "M 5007 error","A/rN", "rad D"))
PNe_df["PNe number"] = np.arange(1,len(x_PNe)+1)

# Objective Residual Cube
obj_residual_cube = np.load("exported_data/"+ galaxy_data["Galaxy name"] +"/list_of_resids_min_obj.npy")

# Data Residual Cube
residual_cube = np.load("exported_data/"+ galaxy_data["Galaxy name"] +"/list_of_resids_min.npy")

def uncertainty_cube_construct(data, x_P, y_P, n_pix):
    data[data == np.inf] = 0.01
    data_shape = data.reshape(y_data, x_data, len(full_wavelength))
    extract_data = np.array([PNextractor(x, y, n_pix, data_shape, wave=full_wavelength, dim=2) for x,y in zip(x_P, y_P)])
    array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(full_wavelength)))
    for p in np.arange(0, len(x_P)):
        list_of_std = [np.abs(np.std(spec)) for spec in extract_data[p]]
        array_to_fill[p] = [np.repeat(list_of_std[i], len(full_wavelength)) for i in np.arange(0, len(list_of_std))]
  
    return array_to_fill

error_cube = uncertainty_cube_construct(residual_cube, x_PNe, y_PNe, n_pixels)
obj_error_cube = uncertainty_cube_construct(obj_residual_cube, x_PNe, y_PNe, n_pixels)

print("Files loaded.")


#run initial 3D fit on selected objects
# LMfit initial parameters
PNe_multi_params = Parameters()

emission_dict = galaxy_data["emissions"]

def gen_params(wave=5007, FWHM=4.0, beta=2.5, em_dict=None):
    # loop through emission dictionary to add different element parameters 
    for em in em_dict:
        #Amplitude params for each emission
        PNe_multi_params.add('Amp_2D_{}'.format(em), value=emission_dict[em][0], min=0.01, expr=emission_dict[em][1])
        #Wavelength params for each emission
        if emission_dict[em][2] == None:
            PNe_multi_params.add("wave_{}".format(em), value=wave, min=wave-40., max=wave+40.)
        else:
            PNe_multi_params.add("wave_{}".format(em), expr=emission_dict[em][2].format(z))
    
    PNe_multi_params.add('x_0', value=(n_pixels/2.), min=0.01, max=n_pixels)
    PNe_multi_params.add('y_0', value=(n_pixels/2.), min=0.01, max=n_pixels)
    PNe_multi_params.add("M_FWHM", value=FWHM, vary=False)
    PNe_multi_params.add("beta", value=beta, vary=False)   
    PNe_multi_params.add("Gauss_bkg",  value=0.00001)
    PNe_multi_params.add("Gauss_grad", value=0.00001)

# generate parameters with values
gen_params(wave=galaxy_data["wave start"], em_dict=emission_dict)

# useful value storage setup
total_Flux = np.zeros((len(x_PNe),len(emission_dict)))
A_2D_list = np.zeros((len(x_PNe),len(emission_dict)))
F_xy_list = np.zeros((len(x_PNe), len(emission_dict), len(PNe_spectra[0])))
emission_amp_list = np.zeros((len(x_PNe),len(emission_dict)))
model_spectra_list = np.zeros((len(x_PNe), n_pixels*n_pixels, len(full_wavelength)))
mean_wave_list = np.zeros((len(x_PNe),len(emission_dict)))
residuals_list = np.zeros(len(x_PNe))
list_of_fit_residuals = np.zeros((len(x_PNe), n_pixels*n_pixels, len(full_wavelength)))

# error lists
moff_A_err = np.zeros((len(x_PNe), len(emission_dict)))
x_0_err = np.zeros((len(x_PNe), len(emission_dict)))
y_0_err = np.zeros((len(x_PNe), len(emission_dict)))
mean_wave_err = np.zeros((len(x_PNe), len(emission_dict)))
Gauss_bkg_err = np.zeros((len(x_PNe), len(emission_dict)))
Gauss_grad_err = np.zeros((len(x_PNe), len(emission_dict)))

FWHM_list = np.zeros(len(x_PNe))
list_of_x = np.zeros(len(x_PNe))
list_of_y = np.zeros(len(x_PNe))
Gauss_bkg = np.zeros(len(x_PNe))
Gauss_grad = np.zeros(len(x_PNe))

def log_10(x):
    return np.log10(x)

def run_minimiser(parameters):
    for PNe_num in np.arange(0, len(x_PNe)):
        useful_stuff = []
        #run minimizer fitting routine
        multi_fit_results = minimize(MUSE_3D_residual, PNe_multi_params, args=(full_wavelength, x_fit, y_fit, PNe_spectra[PNe_num], error_cube[PNe_num], PNe_num, "full", emission_dict, useful_stuff), nan_policy="propagate")
        total_Flux[PNe_num] = np.sum(useful_stuff[1][1],1) * 1e-20
        list_of_fit_residuals[PNe_num] = useful_stuff[0]
        A_2D_list[PNe_num] = useful_stuff[1][0]
        F_xy_list[PNe_num] = useful_stuff[1][1]
        model_spectra_list[PNe_num] = useful_stuff[1][3]
        emission_amp_list[PNe_num] = [multi_fit_results.params["Amp_2D_{}".format(em)] for em in emission_dict]
        mean_wave_list[PNe_num] = [multi_fit_results.params["wave_{}".format(em)] for em in emission_dict]   
        list_of_x[PNe_num] = multi_fit_results.params["x_0"]
        list_of_y[PNe_num] = multi_fit_results.params["y_0"]
        Gauss_bkg[PNe_num] = multi_fit_results.params["Gauss_bkg"]
        Gauss_grad[PNe_num] = multi_fit_results.params["Gauss_grad"]
        #save errors
        moff_A_err[PNe_num] = [multi_fit_results.params["Amp_2D_{}".format(em)] for em in emission_dict]
        mean_wave_err[PNe_num] = [multi_fit_results.params["wave_{}".format(em)] for em in emission_dict]
        x_0_err[PNe_num] = multi_fit_results.params["x_0"].stderr
        y_0_err[PNe_num] = multi_fit_results.params["y_0"].stderr
        Gauss_bkg_err[PNe_num] = multi_fit_results.params["Gauss_bkg"].stderr
        Gauss_grad_err[PNe_num] = multi_fit_results.params["Gauss_grad"].stderr

    # Signal to noise and Magnitude calculations
    list_of_rN = np.array([np.std(PNe_res) for PNe_res in list_of_fit_residuals])
    PNe_df["A/rN"] = A_2D_list[:,0] / list_of_rN # Using OIII amplitude
    
    de_z_means = mean_wave_list[:,0] / (1 + z) # de redshift OIII wavelength position
    
    PNe_df["V (km/s)"] = (c * (de_z_means - 5007.) / 5007.) / 1000.
    
    PNe_df["[OIII] Flux"] = total_Flux[:,0] #store total OIII 5007 line flux
    
    PNe_df["[OIII]/Hb"] = PNe_df["[OIII] Flux"] / total_Flux[:,2] # store OIII/Hb ratio
    
    def log_10(x):
        return np.log10(x)
    
    PNe_df["m 5007"] = -2.5 * PNe_df["[OIII] Flux"].apply(log_10) - 13.74
    dM =  5. * np.log10(D) + 25.   # 31.63
    PNe_df["M 5007"] = PNe_df["m 5007"] - dM
    
    Dist_est = 10.**(((PNe_df["m 5007"].min() + 4.5) -25.) / 5.)
    print("Distance Estimate from PNLF: ", Dist_est, "Mpc")
    
    PNe_table = Table([np.arange(0,len(x_PNe)), np.round(x_PNe), np.round(y_PNe), PNe_df["[OIII] Flux"], PNe_df["[OIII]/Hb"], PNe_df["m 5007"], PNe_df["M 5007"]], 
                      names=("PNe number", "x", "y", "[OIII] Flux", "[OIII]/Hb", "m 5007", "M 5007"))
    ascii.write(PNe_table, "FCC277_PNe_table.txt", format="tab", overwrite=True)
    ascii.write(PNe_table, "FCC277_PNe_table_latex.txt", format="latex", overwrite=True)
    print("Table Created and saved.")


print("Running fitter")
run_minimiser(PNe_multi_params)

# Run PSF fit using objective residuals
#plt.figure(1, figsize=(12,10))
#bins, bins_cens, other = plt.hist(PNe_df["m 5007"].loc[PNe_df["A/rN"]>2], bins=10, edgecolor="black", linewidth=0.8, label="m 5007 > 2 * A/rN", alpha=0.5)
#plt.xlim(26.0,30.0)
#plt.xlabel("$m_{5007}$", fontsize=24)
#plt.ylabel("N Sources", fontsize=24)
#plt.legend(fontsize=15)
#plt.savefig("Plots/"+ galaxy_data["Galaxy name"]+"/M5007_histogram.png")
#bins_cens = bins_cens[:-1]

use_brightest = input("Use Brightest PNe? (y/n) ")
if use_brightest == "y":
    sel_PNe = PNe_df.nlargest(2, "A/rN").index.values
#elif use_brightest == "n":
    # Devise system for PNe choise based upon low background (radial?)

print(sel_PNe)

selected_PNe = PNe_spectra[sel_PNe] # Select PNe from the PNe minicubes
selected_PNe_err = obj_error_cube[sel_PNe] # Select associated errors from the objective error cubes
PSF_params = Parameters()

def model_params(p, n, amp, wave):
    PSF_params.add("moffat_amp_{:03d}".format(n), value=amp, min=0.001)
    PSF_params.add("x_{:03d}".format(n), value=n_pixels/2., min=0.001, max=n_pixels)
    PSF_params.add("y_{:03d}".format(n), value=n_pixels/2., min=0.001, max=n_pixels)
    PSF_params.add("wave_{:03d}".format(n), value=wave, min=wave-40., max=wave+40.)
    PSF_params.add("gauss_bkg_{:03d}".format(n), value=0.001)
    PSF_params.add("gauss_grad_{:03d}".format(n), value=0.001)

for i in np.arange(0,len(sel_PNe)):
        model_params(p=PSF_params, n=i, amp=200.0, wave=galaxy_data["wave start"])    
    
PSF_params.add('FWHM', value=4.0, min=0.01, max=12., vary=True)
PSF_params.add("beta", value=4.0, min=0.01, max=12., vary=True) 

print("Fitting for PSF")
PSF_results = minimize(PSF_residuals, PSF_params, args=(full_wavelength, x_fit, y_fit, selected_PNe, selected_PNe_err), nan_policy="propagate")

#determine PSF values and feed back into 3D fitter

fitted_FWHM = PSF_results.params["FWHM"].value
fitted_beta = PSF_results.params["beta"].value

#Fit PNe with updated PSF
gen_params(wave=galaxy_data["wave start"],FWHM=fitted_FWHM, beta=fitted_beta) # set params up with fitted FWHM and beta values
print("Fitting with PSF")
run_minimiser(PNe_multi_params) # run fitting section again with new values

# Plot out each full spectrum with fitted peaks
for o in np.arange(0, len(x_PNe)):
    plt.figure(figsize=(30,10))
    plt.plot(full_wavelength, np.sum(PNe_spectra[o],0), alpha=0.7, c="k") # data
    plt.plot(full_wavelength, np.sum(model_spectra_list[o],0), c="r") # model
    plt.axhline(residuals_list[o])
    plt.xlabel("Wavelength ($\AA$)", fontsize=18)
    plt.ylabel("Flux Density ($10^{-20}$ $erg s^{-1}$ $cm^{-2}$ $\AA^{-1}$ $arcsec^{-2}$)", fontsize=18)
    plt.savefig("Plots/"+ galaxy_data["Galaxy name"] +"/full_spec_fit_PNe_{}".format(o))
    plt.clf()


#Run the rest of the analysis

print("ta da")