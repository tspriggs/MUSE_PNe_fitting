import sys
import yaml
import lmfit
import argparse
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mcerp import *
from tqdm import tqdm
from astropy.table import Table
from scipy.stats import norm, chi2
from astropy.io import ascii, fits
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters

from functions.ppxf_gal_L import ppxf_L_tot
from functions.PNLF import reconstructed_image, completeness, KS2_test
from functions.MUSE_Models import PNe_residuals_3D, PSF_residuals_3D
from functions.PNe_functions import PNe_spectrum_extractor, robust_sigma, uncertainty_cube_construct, calc_chi2, calc_Lbol_and_mag
from functions.file_handling import paths, open_data, prep_impostor_files

# Let's use a logging package to store stuff, instead of printing.....


# Read in arguments from command line
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument("--loc",    action="store", type=str, required=True)
my_parser.add_argument("--fit_psf", action="store_true", default=False)
my_parser.add_argument("--Lbol", action="store_true", default=False)
args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy   # galaxy name, format of FCC000
loc = args.loc              # MUSE pointing loc: center, middle, halo
fit_PSF = args.fit_psf
calc_Lbol = args.Lbol

DIR_dict = paths(galaxy_name, loc)

# Load in the residual data, in list form
res_data, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy_name, loc, DIR_dict)

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

# Read from the yaml
emission_dict = galaxy_info["emissions"]

z = galaxy_info["velocity"]*1e3 / c

# Construct the PNe FOV coordinate grid for use when fitting PNe.
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])


# load from saved files

# Read in list of x and y coordinates of detected sources for 3D fitting.
x_y_list = np.load(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list]) # separate out from the list the list of x coordinates, as well as y coordinates.
y_PNe = np.array([y[1] for y in x_y_list])

n_PNe = len(x_PNe)

# Retrieve the respective spectra for each PNe source, from the list of spectra data file, using a function to find the associated index locations of the spectra for a PNe.
PNe_spectra = np.array([PNe_spectrum_extractor(x, y, n_pixels, res_data, x_data, wave=wavelength) for x,y in zip(x_PNe, y_PNe)])

# create Pandas dataframe for storage of values from the 3D fitter.
PNe_df = pd.DataFrame(columns=("PNe number", "Ra (J2000)", "Dec (J2000)", "V (km/s)", "m 5007", "m 5007 error", "M 5007", "[OIII] Flux", "M 5007 error", "A/rN", "redchi", "ID"))
PNe_df["PNe number"] = np.arange(0, n_PNe)
PNe_df["ID"] = "PN" # all start as PN

with fits.open(DIR_dict["RAW_DATA"]) as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)

for i in np.arange(0, n_PNe):
    Ra_Dec = utils.pixel_to_skycoord(x_PNe[i],y_PNe[i], wcs_obj).to_string("hmsdms", precision=2).split()
    PNe_df.loc[i,"Ra (J2000)"] = Ra_Dec[0]
    PNe_df.loc[i,"Dec (J2000)"] = Ra_Dec[1]

# Read in Objective Residual Cube .fits file.

with fits.open(DIR_dict["EXPORT_DIR"]+"_resids_obj.fits") as obj_residual_cube:
    obj_error_cube = uncertainty_cube_construct(obj_residual_cube[0].data, x_PNe, y_PNe, n_pixels, x_data, wavelength)

with fits.open(DIR_dict["EXPORT_DIR"]+"_resids_data.fits") as data_residual_cube:
    error_cube = uncertainty_cube_construct(data_residual_cube[0].data, x_PNe, y_PNe, n_pixels, x_data, wavelength)


##################################################
# This is the start of the setup for the 3D fitter.
# Initialise the paramters for 3D fitting.
PNe_multi_params = Parameters()

# extract dictionary of emissions from Galaxy_info.yaml file.
emission_dict = galaxy_info["emissions"]

# Function to generate the parameters for the 3D model and fitter. Built to be able to handle a primary emission ([OIII] here).
# Buil to fit for other emissions lines, as many as are resent in the emission dictionary.
def gen_params(wave=5007, FWHM=4.0, FWHM_err=0.1, beta=2.5, beta_err=0.3, LSF=2.81, em_dict=None, vary_LSF=False, vary_PSF=False):
    # loop through emission dictionary to add different element parameters 
    for em in em_dict:
        #Amplitude params for each emission
        PNe_multi_params.add('Amp_2D_{}'.format(em), value=emission_dict[em][0], min=0.00001, max=1e5, expr=emission_dict[em][1])
        #Wavelength params for each emission
        if emission_dict[em][2] == None:
            PNe_multi_params.add("wave_{}".format(em), value=wave, min=wave-25., max=wave+25.)
        else:
            PNe_multi_params.add("wave_{}".format(em), expr=emission_dict[em][2].format(z))
    
    PNe_multi_params.add("x_0", value=(n_pixels/2.), min=(n_pixels/2.) -3, max=(n_pixels/2.) +3)
    PNe_multi_params.add("y_0", value=(n_pixels/2.), min=(n_pixels/2.) -3, max=(n_pixels/2.) +3)
    PNe_multi_params.add("LSF", value=LSF, vary=vary_LSF, min=LSF-1, max=LSF+1)
    PNe_multi_params.add("M_FWHM", value=FWHM, min=FWHM - FWHM_err, max=FWHM + FWHM_err, vary=vary_PSF)
    PNe_multi_params.add("beta", value=beta, min=beta - beta_err, max=beta + beta_err, vary=vary_PSF)   
    PNe_multi_params.add("Gauss_bkg",  value=0.001, vary=True)#1, min=-200, max=500)
    PNe_multi_params.add("Gauss_grad", value=0.0001, vary=True)#1, min=-2, max=2)
    
# storage setup
total_Flux = np.zeros((n_PNe, len(emission_dict)))
A_2D_list = np.zeros((n_PNe, len(emission_dict)))
F_xy_list = np.zeros((n_PNe, len(emission_dict), len(PNe_spectra[0])))
moff_A = np.zeros((n_PNe,len(emission_dict)))
model_spectra_list = np.zeros((n_PNe, n_pixels*n_pixels, len(wavelength)))
mean_wave_list = np.zeros((n_PNe,len(emission_dict)))
residuals_list = np.zeros(n_PNe)
list_of_fit_residuals = np.zeros((n_PNe, n_pixels*n_pixels, len(wavelength)))
chi_2_r = np.zeros((n_PNe))
list_of_x = np.zeros(n_PNe)
list_of_y = np.zeros(n_PNe)
Gauss_bkg = np.zeros(n_PNe)
Gauss_grad = np.zeros(n_PNe)

# error lists
moff_A_err = np.zeros((n_PNe, len(emission_dict)))
x_0_err = np.zeros((n_PNe, len(emission_dict)))
y_0_err = np.zeros((n_PNe, len(emission_dict)))
mean_wave_err = np.zeros((n_PNe, len(emission_dict)))
Gauss_bkg_err = np.zeros((n_PNe, len(emission_dict)))
Gauss_grad_err = np.zeros((n_PNe, len(emission_dict)))




# Define a function that contains all the steps needed for fitting, including the storage of important values, calculations and pandas assignment.
def run_minimiser(parameters):
    for PNe_num in tqdm(np.arange(0, n_PNe)):
        useful_stuff = []        
        PNe_minimizer       = lmfit.Minimizer(PNe_residuals_3D, PNe_multi_params, fcn_args=(wavelength, x_fit, y_fit, PNe_spectra[PNe_num], error_cube[PNe_num], PNe_num, emission_dict, useful_stuff), nan_policy="propagate")
        multi_fit_results   = PNe_minimizer.minimize()
        total_Flux[PNe_num] = np.sum(useful_stuff[1][1],1) * 1e-20
        list_of_fit_residuals[PNe_num] = useful_stuff[0]
        A_2D_list[PNe_num]  = useful_stuff[1][0]
        F_xy_list[PNe_num]  = useful_stuff[1][1]
        model_spectra_list[PNe_num] = useful_stuff[1][3]
        moff_A[PNe_num]     = [multi_fit_results.params["Amp_2D_{}".format(em)] for em in emission_dict]
        mean_wave_list[PNe_num]     = [multi_fit_results.params["wave_{}".format(em)] for em in emission_dict]   
        chi_2_r[PNe_num]    = multi_fit_results.redchi
        list_of_x[PNe_num]  = multi_fit_results.params["x_0"]
        list_of_y[PNe_num]  = multi_fit_results.params["y_0"]
        Gauss_bkg[PNe_num]  = multi_fit_results.params["Gauss_bkg"]
        Gauss_grad[PNe_num] = multi_fit_results.params["Gauss_grad"]
        #save errors
        moff_A_err[PNe_num]     = [multi_fit_results.params["Amp_2D_{}".format(em)].stderr for em in emission_dict]
        mean_wave_err[PNe_num]  = [multi_fit_results.params["wave_{}".format(em)].stderr for em in emission_dict]
        x_0_err[PNe_num]        = multi_fit_results.params["x_0"].stderr
        y_0_err[PNe_num]        = multi_fit_results.params["y_0"].stderr
        Gauss_bkg_err[PNe_num]  = multi_fit_results.params["Gauss_bkg"].stderr
        Gauss_grad_err[PNe_num] = multi_fit_results.params["Gauss_grad"].stderr

    # Signal to noise and Magnitude calculations
    list_of_rN = np.array([robust_sigma(PNe_res) for PNe_res in list_of_fit_residuals])
    PNe_df["A/rN"] = A_2D_list[:,0] / list_of_rN # Using OIII amplitude
    
    # chi square analysis
    fit_nvary = multi_fit_results.nvarys
    Chi_sqr, redchi = calc_chi2(n_PNe, n_pixels, fit_nvary, PNe_spectra, wavelength, F_xy_list, mean_wave_list, galaxy_info, Gauss_bkg, Gauss_grad)
    
    PNe_df['Chi2']   = Chi_sqr
    PNe_df["redchi"] = redchi
    
    # velocity
    de_z_means = np.array(mean_wave_list[:,0] / (1 + z)) # de redshift OIII wavelength position
    
    PNe_df["V (km/s)"] = (c * (de_z_means - 5006.77) / 5006.77) / 1000.    
    
    PNe_df["fitted_mean_wave"] = mean_wave_list[:,0]

    PNe_df["[OIII] Flux"] = total_Flux[:,0] #store total OIII 5007 line flux
        
    if "hb" in emission_dict:
        PNe_df["[OIII]/Hb"] = PNe_df["[OIII] Flux"] / total_Flux[:,2] # store [OIII]/Hb ratio

    if "ha" in emission_dict:
        PNe_df["Ha Flux"] = total_Flux[:, 1]
    
    PNe_df["m 5007"] = -2.5 * np.log10(PNe_df["[OIII] Flux"].values) - 13.74
    
    

gen_params(wave=5006.77*(1+z), FWHM=galaxy_info["FWHM"], beta=galaxy_info["beta"], LSF=galaxy_info["LSF"], vary_PSF=False, em_dict=emission_dict)    
run_minimiser(PNe_multi_params)

############################ The Great Filter ########################################
PNe_df.loc[PNe_df["A/rN"]<3.0, "ID"] = "-"
# reduced Chi sqr cut
upper_chi = chi2.ppf(0.9973, ((n_pixels*n_pixels)*len(wavelength))-fit_nvary) # 3 sigma = 0.9973
PNe_df.loc[PNe_df["Chi2"]>=upper_chi, "ID"] = "-" 

#### Fit for PSF via N highest A/rN PNe
if fit_PSF == True:
#     if len(PNe_df.where(PNe_df["A/rN"]>10))>0:  # look to see if any A/rN is above 10
        # If so, then use between 2 and 5 of the sources to get PSF
    # else:
    #   Just use df.nlargest(5, "A/rN")
    sel_PNe = PNe_df.loc[PNe_df["ID"]=="PN"].nlargest(5, "A/rN").index.values
    print(sel_PNe)

    selected_PNe = PNe_spectra[sel_PNe]
    selected_PNe_err = obj_error_cube[sel_PNe]

    # Set up PSF params
    PSF_params = Parameters()

    def model_params(p, n, amp, mean):
        PSF_params.add("moffat_amp_{:03d}".format(n), value=amp, min=0.01)
        PSF_params.add("x_{:03d}".format(n), value=(n_pixels/2.), min=(n_pixels/2.) -4, max=(n_pixels/2.) +4)
        PSF_params.add("y_{:03d}".format(n), value=(n_pixels/2.), min=(n_pixels/2.) -4, max=(n_pixels/2.) +4)
        PSF_params.add("wave_{:03d}".format(n), value=mean, min=mean-20., max=mean+20.)
        PSF_params.add("gauss_bkg_{:03d}".format(n),  value=0.001, vary=True)
        PSF_params.add("gauss_grad_{:03d}".format(n), value=0.001, vary=True)


    for i in np.arange(0,len(sel_PNe)):
            model_params(p=PSF_params, n=i, amp=200.0, mean=5006.77*(1+z))    

    PSF_params.add('FWHM', value=4.0, min=0.01, vary=True)
    PSF_params.add("beta", value=2.5, min=0.01, vary=True) 
    PSF_params.add("LSF",  value=2.5, min=0.01, vary=True)

    # Run minimiser to get PSF values
    PSF_results = minimize(PSF_residuals_3D, PSF_params, args=(wavelength, x_fit, y_fit, selected_PNe, selected_PNe_err, z), nan_policy="propagate")

    # Print out results from PSF fit
    print("FWHM: ", round(PSF_results.params["FWHM"].value, 4), "+/-", round(PSF_results.params["FWHM"].stderr, 4), "(", (PSF_results.params["FWHM"].stderr / PSF_results.params["FWHM"].value)*100, "%)")
    print("Beta: ", round(PSF_results.params["beta"].value, 4), "+/-", round(PSF_results.params["beta"].stderr, 4), "(", (PSF_results.params["beta"].stderr / PSF_results.params["beta"].value)*100, "%)")
    print("LSF: " , round(PSF_results.params["LSF"].value , 4), "+/-", round(PSF_results.params["LSF"].stderr , 4), "(", (PSF_results.params["LSF"].stderr / PSF_results.params["LSF"].value)*100, "%)")


    #### Re-fit PNe with fitted PSF values - if fit_PSF == True
    gen_params(wave=5006.77*(1+z), FWHM=PSF_results.params["FWHM"].value, beta=PSF_results.params["beta"].value,
               LSF=PSF_results.params["LSF"].value, vary_PSF=False, em_dict=emission_dict)

    run_minimiser(PNe_multi_params)

    #### Filter again via A/rN and chi square values
    PNe_df.loc[PNe_df["A/rN"] < 3.0, "ID"] = "-"
    PNe_df.loc[PNe_df["Chi2"] >= upper_chi, "ID"] = "-"


#### run impostor check, if not already done

# Prepare files for the impostor checks
prep_impostor_files(galaxy_name, DIR_dict, galaxy_info, PNe_spectra, model_spectra_list, wavelength, n_PNe, n_pixels)

# maybe run a bash/shell script, as we would need to change environment etc.

#### save impostor and interloper object ID's

# run the read_GIST_PNe.py file and store the output ID's

#### Filter with results from impostor check

# list of objects that are chosen to be filtered out (bad fits, objviously not PN, over luminous, etc.)
over_lum_filter = galaxy_info["over_lum"]

# Supernova remnants, HII regions and unknown impostor lists
SNR_filter, HII_filter, unknown_imp_filter = galaxy_info["impostor_filter"]

# Interloping objects list
interloper_filter  = galaxy_info["interloper_filter"]

## Apply filter

PNe_df.loc[PNe_df["PNe number"].isin(over_lum_filter), "ID"] = "OvLu"
PNe_df.loc[PNe_df["PNe number"].isin(SNR_filter), "ID"] = "SNR" 
PNe_df.loc[PNe_df["PNe number"].isin(HII_filter), "ID"] = "HII"
PNe_df.loc[PNe_df["PNe number"].isin(unknown_imp_filter), "ID"] = "imp" 
PNe_df.loc[PNe_df["PNe number"].isin(interloper_filter), "ID"] = "interl"

#### calc errors and determine Distance estimate from brightest m_5007 PNe
##### Error estimation #####
def Moffat_err(Moff_A, FWHM, beta, x_0, y_0):
    alpha = FWHM / (2. * umath.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2 + (y_fit - y_0)**2) / alpha**2
    F_OIII_xy_dist = Moff_A * (2 * ((beta -1)/(alpha**2)))*(1 + rr_gg)**(-beta)
    
    return np.sum(F_OIII_xy_dist*1e-20)

flux_plus_minus = np.ones((n_PNe,2))
mag_plus_minus  = np.ones((n_PNe,2))
                         
for i,p in enumerate(tqdm(range(n_PNe))):
    Moff_A_dist = N(moff_A[p][0], moff_A_err[p][0])
    FWHM_dist   = N(galaxy_info["FWHM"], galaxy_info["FWHM_err"])
    beta_dist   = N(galaxy_info["beta"], galaxy_info["beta_err"])
    x_0_dist    = N(list_of_x[p], x_0_err[p][0])
    y_0_dist    = N(list_of_y[p], y_0_err[p][0])
    
    flux_array = [Moffat_err(Moff_A_dist._mcpts[i], FWHM_dist._mcpts[i], beta_dist._mcpts[i], x_0_dist._mcpts[i], y_0_dist._mcpts[i]) for i in range(len(FWHM_dist._mcpts))]

    flux_plus_minus[i, 0] = np.nanpercentile(flux_array, 84) - np.nanpercentile(flux_array, 50)
    flux_plus_minus[i, 1] = np.nanpercentile(flux_array, 50) - np.nanpercentile(flux_array, 16)

    # Convert fluxes to magnitudes, then find 1 sigma values from median (84th - 50th) & (50th - 16th)
    mag_array = -2.5*np.log10(flux_array)-13.74
    mag_plus_minus[i, 0] = np.nanpercentile(mag_array, 84) - np.nanpercentile(mag_array, 50)
    mag_plus_minus[i, 1] = np.nanpercentile(mag_array, 50) - np.nanpercentile(mag_array, 16)



PNe_df["Flux error up"] = flux_plus_minus[:,0]
PNe_df["Flux error lo"] = flux_plus_minus[:,1]

PNe_df["mag error up"] = mag_plus_minus[:,0]
PNe_df["mag error lo"] = mag_plus_minus[:,1]

#### Distance estimation #######
p_n = int(PNe_df.loc[PNe_df["ID"]=="PN"].nsmallest(1, "m 5007").index.values)

m = PNe_df["m 5007"].iloc[p_n]
m_err_up = PNe_df["mag error up"].iloc[p_n]
m_err_lo = PNe_df["mag error lo"].iloc[p_n]
print("PNe: ", p_n)

M_star = -4.53
M_star_err = 0.08
D_diff_eq = 0.2 * np.log(10) * (10**(0.2*(m + 4.52 - 25)))

Dist_est = 10.**(((m - M_star) -25.) / 5.)
Dist_err_up = np.sqrt((D_diff_eq**2 * m_err_up**2) + ((-D_diff_eq)**2 * M_star_err**2))
Dist_err_lo = np.sqrt((D_diff_eq**2 * m_err_lo**2) + ((-D_diff_eq)**2 * M_star_err**2))

print("Distance Estimate from PNLF: ", f"{np.round(Dist_est,3)} (+ {np.round(Dist_err_up,3)}) (- {np.round(Dist_err_lo,3)}) Mpc")

dM =  5. * np.log10(Dist_est) + 25.
dM_diff_eq = 5/(np.log(10) * Dist_est)
dM_err_up = np.abs(dM_diff_eq)*Dist_err_up
dM_err_lo = np.abs(dM_diff_eq)*Dist_err_lo

print(f"dM = {np.round(dM, 3)} (+ {np.round(dM_err_up,3)}) (- {np.round(dM_err_lo, 3)})")

PNe_df["M 5007"] = PNe_df["m 5007"] - dM

#### save PNe_df and tables for documentation

# save the pandas data-frame for use in the impostor diagnostics.
PNe_df.to_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")

### Construct Table of filtered PNe, ready for paper
# Construct a Astropy table to save certain values for each galaxy.
y_idx = PNe_df.loc[PNe_df["ID"]!="-"].index.values
PNe_table = Table([list(PNe_df.loc[PNe_df["ID"]!="-"].index), PNe_df["Ra (J2000)"].loc[PNe_df["ID"]!="-"], PNe_df["Dec (J2000)"].loc[PNe_df["ID"]!="-"],
                   PNe_df["m 5007"].loc[PNe_df["ID"]!="-"].round(2),
                   PNe_df["A/rN"].loc[PNe_df["ID"]!="-"].round(1),],
                   names=("PNe number", "Ra", "Dec", "m 5007", "A/rN"))


# Save table in tab separated format.
ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results.txt", format="tab", overwrite=True) 
# Save latex table of data.
ascii.write(PNe_table, DIR_dict["EXPORT_DIR"]+"_fit_results_latex.txt", format="latex", overwrite=True) 

#### PNLF

# #####################################################
# ####################### PNLF ########################
# #####################################################


galaxy_image, rec_wave, rec_hdr = reconstructed_image(galaxy_name, loc)
galaxy_image = galaxy_image.reshape([y_data, x_data])

PNe_mag = PNe_df["M 5007"].loc[PNe_df["ID"]=="PN"].values
app_mag = PNe_df["m 5007"].loc[PNe_df["ID"]=="PN"].values

# Total PNLF
PNLF, PNLF_corr, completeness_ratio, Abs_M, app_m = completeness(galaxy_name, loc, DIR_dict, PNe_mag, PNe_multi_params, Dist_est, galaxy_image, 3.0,
                                                                 galaxy_info["gal_mask"], z, galaxy_info["star_mask"], c1=0.307)  # Estimating the completeness for the central pointing

step = abs(Abs_M[1]-Abs_M[0])
# Getting the normalisation - sum of correctied PNLF, times bin size
total_norm = np.sum(np.abs(PNLF_corr)) * step

# Scaling factor
scal = len(PNe_mag) / total_norm

# Constraining to -2.0 in magnitude
idx = np.where(Abs_M <= np.min(PNe_mag)+2.5)

# Plot the PNLF
plt.figure(figsize=(14,10))

binwidth = 0.2

hist = plt.hist(app_mag, bins=np.arange(min(app_mag), max(app_mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')

KS2_stat = KS2_test(dist_1=PNLF_corr[1:18:2]*scal*binwidth, dist_2=hist[0], conf_lim=0.1)
print(KS2_stat)

ymax = max(hist[0])

plt.plot(app_m, PNLF*scal*binwidth, '-', color='blue', marker="o", label="PNLF")
plt.plot(app_m, PNLF_corr*scal*binwidth,'-.', color='blue', label="Incompleteness corrected PNLF")
plt.xlabel(r'$m_{5007}$', fontsize=30)
plt.ylabel(r'$N_{PNe}$', fontsize=30)
#plt.yticks(np.arange(0,ymax+4, 5))
plt.plot(0,0, alpha=0.0, label=f"KS2 test = {round(KS2_stat[0],3)}")
plt.plot(0,0, alpha=0.0, label=f"pvalue   = {round(KS2_stat[1],3)}")
plt.xlim(-5.0+dM,-1.5+dM)
plt.ylim(0,ymax+(2*ymax))
# plt.xlim(26.0,30.0); plt.ylim(0,45);

plt.tick_params(labelsize = 25)

#plt.axvline(PNe_df["m 5007"].loc[PNe_df["ID"]=="PN"].values.min() - 31.63)
plt.legend(loc=2, fontsize=20)
plt.savefig(DIR_dict["PLOT_DIR"]+"_PNLF.pdf", bbox_inches='tight')
plt.savefig(DIR_dict["PLOT_DIR"]+"_PNLF.png", bbox_inches='tight')

# Calculate the number of PNe, via the PNLF
N_PNe = np.sum(PNLF[idx]*scal) * step

print("Number of PNe from PNLF: ", N_PNe, "+/-", (1/np.sqrt(len(PNe_df.loc[PNe_df["ID"]=="PN"])))*N_PNe)


#### L_bol

##### Integrated, bolometric Luminosity of galaxy FOV spectra #####
if calc_Lbol == True:
    L_bol = calc_Lbol_and_mag(DIR_dict, galaxy_info,
                              dM, dM_err=[dM_err_up, dM_err_lo])
    # raw_data_cube = DIR_dict["RAW_DATA"] # read in raw data cube
    
    # xe, ye, length, width, alpha = gal_mask_params
    
    # orig_hdulist = fits.open(raw_data_cube)
    # raw_data_cube = np.copy(orig_hdulist[1].data)
    # h1 = orig_hdulist[1].header
    # s = np.shape(orig_hdulist[1].data)
    # Y, X = np.mgrid[:s[1], :s[2]]
    # elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1    
    
    # # Now mask the stars
    # star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in galaxy_info["star_mask"]],0).astype(bool)
    
    # # Combine elip_mask and star)_mask_sum to make total_mask
    # total_mask = ((np.isnan(orig_hdulist[1].data[1,:,:])==False) & (elip_mask==False) & (star_mask_sum==False))
    # indx_mask = np.where(total_mask==True)
    
    # good_spectra = np.zeros((s[0], len(indx_mask[0])))
    
    # for i, (y, x)  in enumerate(zip(tqdm(indx_mask[0]), indx_mask[1])):
    #     good_spectra[:,i] = raw_data_cube[:,y,x]
    
    # print("Collapsing cube now....")    
        
    # gal_lin = np.nansum(good_spectra, 1)
            
    # print("Cube has been collapsed...")
    
    # L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=galaxy_info["velocity"], dist_mod=dM, dM_err=[dM_err_up, dM_err_lo])

#### alpha_2.5

#### save values to gal_df csv file
print("Number of PNe after A/rN cut: ", len(PNe_df["ID"].loc[PNe_df["ID"]=="PN"]))

print("Number of PNe after A/rN and Reduced chi-square cuts: ", len(PNe_df["ID"].loc[PNe_df["ID"]=="PN"]))


print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table.txt")
print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table_latex.txt")

print(galaxy_name)
n_p = len(PNe_df.loc[PNe_df["ID"]=="PN"])
print(f"N PNe used:      {n_p}")
print(f"PNLF N:          {N_PNe}")
print(f"Distance of:     {Dist_est} +/- {Dist_err_up}")
print(f"Distance Mod of: {dM} +/- {dM_err_up}")
if calc_Lbol == True:
    print(f"L_bol of:        {L_bol[0]}")
    print(f"L_bol error:     + {L_bol[1][0] - L_bol[0]}, - {L_bol[0] - L_bol[1][1]}")
    print(f"Rmag of :        {L_bol[7]}")
    print(f"Vmag of :        {L_bol[8]}")
   






######### This is the end of PNe analysis script
