import lmfit
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mcerp as mcerp
from tqdm import tqdm
from scipy.stats import chi2
from astropy.wcs import WCS, utils
from lmfit import Parameters


from functions.PNLF import PNLF_analysis, MC_PNLF_runner, calc_PNLF_interp_comp, scale_PNLF, ecdf, form_PNLF_CDF, calc_PNLF
from functions.MUSE_Models import PNe_residuals_3D, generate_3D_fit_params
from functions.PNe_functions import calc_chi2, plot_single_spec, dM_to_D
from functions.file_handling import paths, open_PNe, prep_impostor_files
from functions.PSF_funcs import run_PSF_analysis
from functions.completeness import prep_completness_data, calc_completeness

np.random.seed(42)

# import warnings
# uncomment this if you do not want to view "ignore" warnings. These do not alter the output of this script.
# warnings.filterwarnings("ignore")

# Read in arguments from command line
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True, 
                        help="The name of the galaxy to be analysed.")
my_parser.add_argument("--loc",    action="store", type=str, required=False, default="", 
                        help="The pointing location, e.g. center, halo or middle")
my_parser.add_argument("--fit_psf", action="store_true", default=False, 
                        help="A flag used to determine if a PSF fit should be run.")
my_parser.add_argument("--save_gist", action="store_true", default=False, 
                        help="This Flag indicates that you want to save the files used for impostor detection, ready for GIST.")

args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy   # galaxy name, format of FCC000
loc = args.loc              # MUSE pointing loc: center, middle, halo
fit_PSF = args.fit_psf
save_gist = args.save_gist


DIR_dict = paths(galaxy_name, loc)

# Load in the residual data, in list form
PNe_spectra, hdr, wavelength, obj_err, res_err, residual_shape, x_data, y_data, galaxy_info = open_PNe(galaxy_name, loc, DIR_dict)

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

# Calculate redshift of galaxy from galaxy's systemic velocity
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


# create Pandas dataframe for storage of values from the 3D fitter.
PNe_df = pd.DataFrame(columns=("PNe number", "Ra (J2000)", "Dec (J2000)", "m 5007", "m 5007 error", "M 5007", "[OIII] Flux", "M 5007 error", "A/rN", "PNe_LOS_V", "redchi", "ID"))
PNe_df["PNe number"] = np.arange(0, n_PNe)
PNe_df["ID"] = "PN" # all start as PN


wcs_obj = WCS(hdr, naxis=2)

for i in np.arange(0, n_PNe):
    Ra_Dec = utils.pixel_to_skycoord(x_PNe[i],y_PNe[i], wcs_obj).to_string("hmsdms", precision=2).split()
    PNe_df.loc[i,"Ra (J2000)"] = Ra_Dec[0]
    PNe_df.loc[i,"Dec (J2000)"] = Ra_Dec[1]



##################################################
# This is the start of the setup for the 3D fitter.
# Initialise the paramters for 3D fitting.
PNe_multi_params = Parameters()

# extract dictionary of emissions from Galaxy_info.yaml file.
emission_dict = galaxy_info["emissions"]

# Function to generate the parameters for the 3D model and fitter. Built to be able to handle a primary emission ([OIII] here).
# Buil to fit for other emissions lines, as many as are resent in the emission dictionary.


# storage setup
total_Flux = np.zeros((n_PNe, len(emission_dict)))
A_2D_list = np.zeros((n_PNe, len(emission_dict)))
F_xy_list = np.zeros((n_PNe, len(emission_dict), len(PNe_spectra[0])))
moff_A = np.zeros((n_PNe,len(emission_dict)))
model_spectra_list = np.zeros((n_PNe, n_pixels*n_pixels, len(wavelength)))
mean_wave_list = np.zeros((n_PNe,len(emission_dict)))
list_of_fit_residuals = np.zeros((n_PNe, n_pixels*n_pixels, len(wavelength)))
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
        # Prepare and run the minimisation analysis from LMfit, applying the 3D model to the PNe
        PNe_minimizer       = lmfit.Minimizer(PNe_residuals_3D, parameters, fcn_args=(wavelength, x_fit, y_fit, PNe_spectra[PNe_num], res_err[PNe_num], emission_dict, useful_stuff), nan_policy="propagate")
        multi_fit_results   = PNe_minimizer.minimize()
        # Store the results of the minimize() call.
        total_Flux[PNe_num] = np.sum(useful_stuff[1][1],1) * 1e-20
        list_of_fit_residuals[PNe_num] = useful_stuff[0]
        A_2D_list[PNe_num]  = useful_stuff[1][0]
        F_xy_list[PNe_num]  = useful_stuff[1][1]
        model_spectra_list[PNe_num] = useful_stuff[1][3]
        moff_A[PNe_num]     = [multi_fit_results.params["Amp_2D_{}".format(em)].value for em in emission_dict]
        mean_wave_list[PNe_num] = [multi_fit_results.params["wave_{}".format(em)].value for em in emission_dict]
        list_of_x[PNe_num]  = multi_fit_results.params["x_0"].value
        list_of_y[PNe_num]  = multi_fit_results.params["y_0"].value
        Gauss_bkg[PNe_num]  = multi_fit_results.params["Gauss_bkg"].value
        Gauss_grad[PNe_num] = multi_fit_results.params["Gauss_grad"].value
        # Store the errors from the minimisation run
        moff_A_err[PNe_num]     = [multi_fit_results.params["Amp_2D_{}".format(em)].stderr for em in emission_dict]
        mean_wave_err[PNe_num]  = [multi_fit_results.params["wave_{}".format(em)].stderr for em in emission_dict]
        x_0_err[PNe_num]        = multi_fit_results.params["x_0"].stderr
        y_0_err[PNe_num]        = multi_fit_results.params["y_0"].stderr
        Gauss_bkg_err[PNe_num]  = multi_fit_results.params["Gauss_bkg"].stderr
        Gauss_grad_err[PNe_num] = multi_fit_results.params["Gauss_grad"].stderr

    # Signal to noise calculations, taking the std of the residual array (data-model), to form the Amplitude to residual noise value: A/rN
    list_of_rN = [np.std(res) for res in list_of_fit_residuals]
    PNe_df["A/rN"] = A_2D_list[:,0] / list_of_rN # Using OIII amplitude

    # chi-square, and reduced chi-square calculations
    Chi_sqr, redchi = calc_chi2(n_PNe, n_pixels, multi_fit_results.nvarys, PNe_spectra, wavelength, \
                                F_xy_list, mean_wave_list, galaxy_info, Gauss_bkg, Gauss_grad)

    PNe_df['Chi2']   = Chi_sqr
    PNe_df["redchi"] = redchi

    # velocity
    PNe_df["PNe_LOS_V"] = (c * (mean_wave_list[:,0] - 5006.77) / 5006.77) / 1000.

    PNe_df["fitted_mean_wave"] = mean_wave_list[:,0]

    PNe_df["[OIII] Flux"] = total_Flux[:,0] #store total OIII 5007 line flux

    PNe_df["m 5007"] = -2.5 * np.log10(PNe_df["[OIII] Flux"].values) - 13.74 + (-2.5*np.log10(galaxy_info["F_corr"]))


    return multi_fit_results.nvarys

# generate the parameters to be used for the PNe 3D modelling function
PNe_multi_params = generate_3D_fit_params(wave=5006.77*(1+z), FWHM=galaxy_info["FWHM"], beta=galaxy_info["beta"], \
                                            LSF=galaxy_info["LSF"], vary_PSF=False, em_dict=emission_dict, z=z)

# Now we run the function "run_minimiser", which fits each object minicube with the 3D model. The function itself returns the number of parameters varied in the 3D fitting process.
fit_nvary = run_minimiser(PNe_multi_params)

############################ The Great Filter #############################
PNe_df.loc[PNe_df["A/rN"]<3.0, "ID"] = "-"
# reduced Chi-square cut
upper_chi = chi2.ppf(0.9973, ((n_pixels*n_pixels)*len(wavelength))-fit_nvary) # 3 sigma = 0.9973
PNe_df.loc[PNe_df["Chi2"]>=upper_chi, "ID"] = "-"

# list of objects that are chosen to be filtered out (bad fits, objviously not PN, over luminous, etc.)
over_lum_filter = galaxy_info["over_lum"]

# Supernova remnants, HII regions and unknown impostor lists
SNR_filter, HII_filter, unknown_imp_filter = galaxy_info["impostor_filter"]

# Interloping objects list
interloper_filter  = galaxy_info["interloper_filter"]


## Apply filters and apply ID labels appropriate to the filter.
PNe_df.loc[PNe_df["PNe number"].isin(over_lum_filter), "ID"] = "OvLu" # Over-luminous sources
PNe_df.loc[PNe_df["PNe number"].isin(SNR_filter), "ID"] = "SNR" # Supernova remnants
PNe_df.loc[PNe_df["PNe number"].isin(HII_filter), "ID"] = "HII" # Compact HII regions
PNe_df.loc[PNe_df["PNe number"].isin(unknown_imp_filter), "ID"] = "imp" # Undecided Impostors
PNe_df.loc[PNe_df["PNe number"].isin(interloper_filter), "ID"] = "interl" # interlopers.

# if the observational region is "halo" or "middle", then filter out sources that have already been catalogued.
# Label such overlapping / re-appearing sources via the "CrssMtch" filter ID
if loc in ["halo", "middle"]:
    crossmatch_filter = galaxy_info["crossmatch_filter"]
    PNe_df.loc[PNe_df["PNe number"].isin(crossmatch_filter), "ID"] = "CrssMtch"



#### Fit for PSF via N highest A/rN PNe
if fit_PSF == True:
    print("\n######################################################################")
    print("##################### Fitting PNe for the PSF ########################")
    print("######################################################################\n")

    sel_PNe = PNe_df.loc[(PNe_df["ID"] == "PN")].nlargest(3, "A/rN").index.values#[1:]

    PSF_results = run_PSF_analysis(sel_PNe, PNe_spectra, obj_err, wavelength, x_fit, y_fit, z, eval_Conf_int=False)

    generate_3D_fit_params(wave=5006.77*(1+z), FWHM=PSF_results.params["FWHM"].value, beta=PSF_results.params["beta"].value,
               LSF=PSF_results.params["LSF"].value, vary_PSF=False, em_dict=emission_dict, z=z)

    fit_nvarys = run_minimiser(PNe_multi_params)

    #### Filter again via A/rN and chi square values
    PNe_df.loc[PNe_df["A/rN"] < 3.0, "ID"] = "-"
    PNe_df.loc[PNe_df["Chi2"] >= upper_chi, "ID"] = "-"


#### run impostor check, if not already done ####
# Prepare files for the impostor checks
if save_gist == True:
    print("\n#####################################################################################")
    print("##################### Saving files for contamination testing ########################")
    print("#####################################################################################\n")
    # Filter through the sources for impostor checks by choosing only those sources that have passed initial fitting cuts (A/rN >3 and chi)
    filter_ind = PNe_df.loc[PNe_df["ID"]!="-"].index

    prep_impostor_files(galaxy_name, DIR_dict, PNe_spectra[filter_ind], model_spectra_list[filter_ind],
                        wavelength, n_pixels, x_PNe[filter_ind], y_PNe[filter_ind])
    
    print("Sources have been saved for Impostor verification checks.")


#### calc errors and determine Distance estimate from brightest m_5007 PNe
##### Error estimation #####
# print("\n################################################################")
# print("##################### Calculating errors #######################")
# print("################################################################\n")

def Moffat_err(Moff_A, FWHM, beta, x_0, y_0):
    alpha = FWHM / (2. * mcerp.umath.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2 + (y_fit - y_0)**2) / alpha**2
    F_OIII_xy_dist = Moff_A * (2 * ((beta -1)/(alpha**2)))*(1 + rr_gg)**(-beta)

    return np.sum(F_OIII_xy_dist*1e-20)

flux_err_plus_minus = np.ones((n_PNe,2))
mag_err_plus_minus  = np.ones((n_PNe,2))

# for PN in PNe_df, use the mcerp package to form model distributions, from the best-fit values, calculating the upper and lower bounds uncertainties
# for the flux and magnitude of each PNe.

for p in tqdm(PNe_df.loc[PNe_df["ID"]=="PN"].index):
    Moff_A_dist = mcerp.N(moff_A[p][0], np.abs(moff_A_err[p][0]))
    FWHM_dist   = mcerp.N(galaxy_info["FWHM"], galaxy_info["FWHM_err"])
    beta_dist   = mcerp.N(galaxy_info["beta"], galaxy_info["beta_err"])
    x_0_dist    = mcerp.N(list_of_x[p], x_0_err[p][0])
    y_0_dist    = mcerp.N(list_of_y[p], y_0_err[p][0])

    flux_array = [Moffat_err(Moff_A_dist._mcpts[i], FWHM_dist._mcpts[i], beta_dist._mcpts[i], x_0_dist._mcpts[i], y_0_dist._mcpts[i]) 
                    for i in range(len(FWHM_dist._mcpts))]

    flux_err_plus_minus[p, 0] = np.nanpercentile(flux_array, 84) - np.nanpercentile(flux_array, 50)
    flux_err_plus_minus[p, 1] = np.nanpercentile(flux_array, 50) - np.nanpercentile(flux_array, 16)

    # Convert fluxes to magnitudes, then find 1 sigma values from median (84th - 50th) & (50th - 16th)
    mag_array = -2.5*np.log10(flux_array)-13.74
    mag_err_plus_minus[p, 0] = np.nanpercentile(mag_array, 84) - np.nanpercentile(mag_array, 50)
    mag_err_plus_minus[p, 1] = np.nanpercentile(mag_array, 50) - np.nanpercentile(mag_array, 16)


# Store flux errors in the PNe_df dataframe
PNe_df["Flux error up"] = flux_err_plus_minus[:,0]
PNe_df["Flux error lo"] = flux_err_plus_minus[:,1]

# Store magnitude errors in the PNe_df dataframe
PNe_df["mag error up"] = mag_err_plus_minus[:,0]
PNe_df["mag error lo"] = mag_err_plus_minus[:,1]



# print("\n##########################")
# print("########## PNLF ##########")
# print("##########################\n")

# read in the observed completeness ratio profile for the given galaxy observation.
try:
    obs_comp = np.load(DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy")
except:
    print("\nThere appears to be no completeness profile saved for this galaxy.")
    print(f"Calculating the completeness profile of {galaxy_name} {loc}\n")
    step = 0.001
    m_5007 = np.arange(26, 31, step)
    image, Noise_map = prep_completness_data(galaxy_name, loc, DIR_dict, galaxy_info)

    completeness_ratio = calc_completeness(image, Noise_map, m_5007, galaxy_info, 3.0, 9, )

    np.save(DIR_dict["EXPORT_DIR"]+"_completeness_ratio", completeness_ratio)
    print(f"The completeness profile has been calculated and will be stored for later use\n")
    obs_comp = np.load(DIR_dict["EXPORT_DIR"]+"_completeness_ratio.npy")

# Retrieve key information from the PNe_df dataframe: [OIII] magntiudes and errors, only for sources that have the "ID" of "PN".
gal_m_5007 = PNe_df["m 5007"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_up = PNe_df["mag error up"].loc[PNe_df["ID"].isin(["PN"])].values
gal_m_5007_err_lo = PNe_df["mag error lo"].loc[PNe_df["ID"].isin(["PN"])].values

# prepare key values that are used to form the PNLF.
step = 0.001
M_star = -4.53
M_5007 = np.arange(M_star, 0.53, step)
m_5007 = np.arange(26, 31, step)

# dM_guess is used as an initial guess for the purposes of fitting.
dM_guess = 31.5

# Change the boolean values between True and False to alter which parameters are varied during the initial PNLF fit.
vary_dict = {"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}
PNLF_results = PNLF_analysis(galaxy_name, loc, gal_m_5007, obs_comp, M_5007, m_5007, dM_in=dM_guess, c2_in=0.307, vary_dict=vary_dict, comp_lim=False)
best_fit_dM = PNLF_results.params["dM"].value
best_fit_c2 = PNLF_results.params["c2"].value

# Uncertainty evaluation
# Change the boolean values between True and False to alter which parameters are varied during the MC_PNLF_runner function.
vary_dict_MC = {"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}
MC_dM_distr, MC_c2_distr, MC_PNLF_results = MC_PNLF_runner(galaxy_name, loc, gal_m_5007, gal_m_5007_err_up, gal_m_5007_err_lo, 
                                                        obs_comp, c2_in=0.307, n_runs=3000, vary_dict=vary_dict_MC, comp_lim=False)

# filter out values in the MC bootstrapper results that are close to, or evaluate to, the limits of the parameter bounds.
MC_filter = (MC_dM_distr > MC_PNLF_results.params["dM"].min+0.1) & (MC_dM_distr < MC_PNLF_results.params["dM"].max-0.1) & \
     (MC_c2_distr > MC_PNLF_results.params["c2"].min+0.1) & (MC_c2_distr < MC_PNLF_results.params["c2"].max-0.1)


# calculate the 16th, 50th and 84th percentile of the dM distribution returned from the MC bootstrapper.
MC_dM_16, MC_dM_50, MC_dM_84 = np.nanpercentile(MC_dM_distr[MC_filter], [16, 50, 84], axis=0)


dM_err_up = MC_dM_84-best_fit_dM 
dM_err_lo = best_fit_dM-MC_dM_16

PNLF_best_fit, PNLF_interp, PNLF_comp_corr = calc_PNLF_interp_comp(best_fit_dM, 0.307, obs_comp)

PNLF_err_16, PNLF_16_interp, PNLF_comp_corr_16 = calc_PNLF_interp_comp(best_fit_dM-dM_err_lo, 0.307, obs_comp)

PNLF_err_84, PNLF_84_interp, PNLF_comp_corr_84 = calc_PNLF_interp_comp(best_fit_dM+dM_err_up, 0.307, obs_comp)

bw = 0.2

## Plotting of binned PNe and PNLF and completeness corrected PNLF
plt.figure(figsize=(10,7))

# histogram plot of the observed PNe
plt.hist(gal_m_5007, bins=np.arange(min(gal_m_5007), max(gal_m_5007) + bw, bw), ec="black", alpha=0.8, zorder=1, label="PNe") # histogram of gal_m_5007

# Plot the interpolated, PNLF that is using the initial best fit dM value
plt.plot(m_5007, scale_PNLF(gal_m_5007, PNLF_interp, PNLF_comp_corr, bw, step), c="k", label="Best-fit Ciardullo PNLF")

# to show the 1 sigma uncertainty range, use the fillbetween, using scaled upper (16th) and lower (84th) percentile interpolated PNLFS
plt.fill_between(m_5007, scale_PNLF(gal_m_5007, PNLF_16_interp, PNLF_comp_corr, bw, step), scale_PNLF(gal_m_5007, PNLF_84_interp, PNLF_comp_corr, bw, step), \
                 alpha=0.4, color="b", zorder=2, label=r"C89 PNLF 1$\sigma$")

#Plot the completeness corrected, interpolated PNLF form, along with the associated uncertainty regions
plt.plot(m_5007, scale_PNLF(gal_m_5007, PNLF_comp_corr, PNLF_comp_corr, bw, step), c="k", ls="-.", label="Incompleteness-corrected C89 PNLF") 
plt.fill_between(m_5007, scale_PNLF(gal_m_5007, PNLF_comp_corr_16, PNLF_comp_corr, bw, step), scale_PNLF(gal_m_5007, PNLF_comp_corr_84, PNLF_comp_corr, bw, step), \
                 alpha=0.4, color="b",zorder=2) 


idx = np.where(M_5007 <= M_star+2.5)[0]
N_PNLF = np.sum(PNLF_best_fit[idx]* (len(gal_m_5007) / (np.sum(PNLF_comp_corr)*step))) * step


plt.xlim(26.0,30.0)
plt.ylim(0, np.max(scale_PNLF(gal_m_5007, PNLF_best_fit, PNLF_comp_corr, bw, step)[idx])*1.5)

plt.title(f"{galaxy_name}, dM={round(best_fit_dM,2)}$"+"^{+"+f"{round(dM_err_up,2)}"+"}"+f"_{ {-round(dM_err_lo,2)} }$")
plt.xlabel(r"$m_{5007}$", fontsize=15)
plt.ylabel(r"$N_{PNe} \ per \ bin$", fontsize=15)
plt.xlim(26.0,30.0)
plt.legend(loc="upper left", fontsize=12)
plt.savefig(DIR_dict["PLOT_DIR"]+"_fitted_PNLF.png", bbox_inches='tight')


PNe_df["M 5007"] = PNe_df["m 5007"] - best_fit_dM

PNe_df.to_csv(DIR_dict["EXPORT_DIR"]+"_PNe_df.csv")


#### plot 5 brightest PNe
five_bright = PNe_df.loc[PNe_df["ID"]=="PN"].nsmallest(5, "m 5007").index.values
for i, n in enumerate(five_bright):
    plot_single_spec(n, DIR_dict, wavelength, PNe_spectra, model_spectra_list, i)
    plt.close()

#### Plot the PNe eCDF vs the PNLF CDF, using best-fit values:
plt.figure(figsize=(10,8))
x, PNe_cdf = ecdf(gal_m_5007)
PNLF = calc_PNLF(m_star=M_star+best_fit_dM, mag=M_5007+best_fit_dM,)
PNLF_CDF = form_PNLF_CDF(PNLF, gal_m_5007, best_fit_dM, obs_comp, M_5007, m_5007)
plt.plot(x, PNe_cdf, c="k", label="PNe")
plt.plot(x, PNLF_CDF, c="b", label="PNLF")
plt.xlabel("m$_\mathrm{5007}$", fontsize=20)
plt.ylabel("CDF", fontsize=20)
plt.legend(loc="upper left", fontsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.tick_params(axis="y", labelsize=15)
plt.savefig(DIR_dict["PLOT_DIR"]+"_PNe_ECDF_vs_PNLF_CDF.png", bbox_inches="tight")


#### save values to gal_df csv file

print(f"{galaxy_name} {loc}")

print("Number of objects removed by A/rN and Reduced chi-square cuts: ", len(PNe_df["ID"].loc[PNe_df["ID"]=="-"]))


print("\n")
n_p = len(PNe_df.loc[PNe_df["ID"]=="PN"])
print(f"N PNe detected:      {n_p}")
print(f"PNLF N:              {round(N_PNLF,3)}")
print("\n")

dist = dM_to_D(best_fit_dM)
print("Distance in Mpc:", round(dist,3), " Mpc \n")

print(f"{galaxy_name} {loc} PNLF_dM={round(best_fit_dM, 3)} + {round(MC_dM_84-MC_dM_50, 3)} - {round(MC_dM_50-MC_dM_16, 3)}")
print(np.nanpercentile(MC_dM_distr[MC_filter], [50,16,84]))


PN_result_df = pd.DataFrame(data=[[n_p, N_PNLF, round(MC_dM_50, 3), round(MC_dM_84-best_fit_dM, 3), 
        round(best_fit_dM - MC_dM_16, 3), round(10.**((MC_dM_50 -25.) / 5.),3)]],
        columns=("PNe N", "PNLF N", "PNLF dM", "PNLF dM err up", "PNLF dM err lo", "PNLF Dist"))


PN_result_df.to_csv(DIR_dict["EXPORT_DIR"]+"_PN_result_df.csv")


########################################
# This is the end of PNe analysis script
########################################

