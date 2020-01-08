import sys
import yaml
import lmfit
import argparse
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from astropy.table import Table
from scipy.stats import norm, chi2
from astropy.io import ascii, fits
from astropy.wcs import WCS, utils, wcs
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters

from ppxf_gal_L import ppxf_L_tot
from PNLF import open_data, reconstructed_image, completeness
from MUSE_Models import PNe_residuals_3D, PNe_spectrum_extractor, PSF_residuals_3D, robust_sigma


# Load in yaml file to query galaxy properties
with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True)
my_parser.add_argument("--loc",    action="store", type=str, required=True)
args = my_parser.parse_args()

# Define galaxy name
galaxy_name = args.galaxy #input("Please type which Galaxy you want to analyse, use FCC000 format: ")

# Queries user for a galaxy name, in the form of FCC000, and taking the relevant info from the yaml file

galaxy_data = galaxy_info[galaxy_name]

RAW_DIR    = f"/local/tspriggs/Fornax_data_cubes/{galaxy_name}"
RAW_DAT    = f"/local/tspriggs/Fornax_data_cubes/{galaxy_name}{loc}.fits"
DATA_DIR   = f"galaxy_data/{galaxy_name}_data/"
EXPORT_DIR = f"exported_data/{galaxy_name}/"
PLOT_DIR   = f"Plots/{galaxy_name}/{galaxy_name}"

# Load in the residual data, in list form
hdulist = fits.open(f"{DATA_DIR}{galaxy_name}{loc}_residuals_list.fits")
hdr = hdulist[0].header
wavelength = np.exp(hdulist[1].data) # extract header from residual cube

x_data = res_hdr["XAXIS"]
y_data = res_hdr["YAXIS"]

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

# Read from the yaml
emission_dict = galaxy_data["emissions"]

gal_vel = galaxy_data["velocity"]
z = gal_vel*1e3 / c

gal_mask_params = galaxy_data["gal_mask"]
star_mask_params = galaxy_data["star_mask"]

# Construct the PNe FOV coordinate grid for use when fitting PNe.
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])


# load from saved files

# Read in list of x and y coordinates of detected sources for 3D fitting.
x_y_list = np.load(EXPORT_DIR+galaxy_name+"_PNe_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list]) # separate out from the list the list of x coordinates, as well as y coordinates.
y_PNe = np.array([y[1] for y in x_y_list])

# Retrieve the respective spectra for each PNe source, from the list of spectra data file, using a function to find the associated index locations of the spectra for a PNe.
PNe_spectra = np.array([PNe_spectrum_extractor(x, y, n_pixels, hdulist[0].data, x_data, wave=wavelength) for x,y in zip(x_PNe, y_PNe)])

# create Pandas dataframe for storage of values from the 3D fitter.
PNe_df = pd.DataFrame(columns=("PNe number", "Ra (J2000)", "Dec (J2000)", "V (km/s)", "m 5007", "m 5007 error", "M 5007", "[OIII] Flux", "M 5007 error", "A/rN", "redchi", "Filter"))
PNe_df["PNe number"] = np.arange(0,len(x_PNe))
PNe_df["Filter"] = "Y"

with fits.open(RAW_DIR+"center.fits") as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)

for i in np.arange(0, len(x_PNe)):
    Ra_Dec = utils.pixel_to_skycoord(x_PNe[i],y_PNe[i], wcs_obj).to_string("hmsdms", precision=2).split()
    PNe_df.loc[i,"Ra (J2000)"] = Ra_Dec[0]
    PNe_df.loc[i,"Dec (J2000)"] = Ra_Dec[1]

# Read in Objective Residual Cube .fits file.
obj_residual_cube = fits.open(EXPORT_DIR+galaxy_name+"_resids_obj.fits")

# Read in Data Residual Cube .fits file.
data_residual_cube = fits.open(EXPORT_DIR+galaxy_name+"_resids_data.fits")

# Function to extract the uncertainties and transform them into a standard deviation version for fitting purposes.
def uncertainty_cube_construct(data, x_P, y_P, n_pix):
    data[data == np.inf] = 0.01
    extract_data = np.array([PNe_spectrum_extractor(x, y, n_pix, data, x_data, wave=wavelength) for x,y in zip(x_P, y_P)])
    array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(wavelength)))
    for p in np.arange(0, len(x_P)):
        list_of_std = np.abs([robust_sigma(dat) for dat in extract_data[p]])            
        array_to_fill[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]

    return array_to_fill

# Run above function to get the error and obj_error cubes for fitting purposes (uncertainty).
error_cube = uncertainty_cube_construct(data_residual_cube[0].data, x_PNe, y_PNe, n_pixels)
obj_error_cube = uncertainty_cube_construct(obj_residual_cube[0].data, x_PNe, y_PNe, n_pixels)

##################################################
# This is the start of the setup for the 3D fitter.
# Initialise the paramters for 3D fitting.
PNe_multi_params = Parameters()

# extract dictionary of emissions from Galaxy_info.yaml file.
emission_dict = galaxy_data["emissions"]

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
total_Flux = np.zeros((len(x_PNe), len(emission_dict)))
A_2D_list = np.zeros((len(x_PNe), len(emission_dict)))
F_xy_list = np.zeros((len(x_PNe), len(emission_dict), len(PNe_spectra[0])))
moff_A = np.zeros((len(x_PNe),len(emission_dict)))
model_spectra_list = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength)))
mean_wave_list = np.zeros((len(x_PNe),len(emission_dict)))
residuals_list = np.zeros(len(x_PNe))
list_of_fit_residuals = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength)))
chi_2_r = np.zeros((len(x_PNe)))

# error lists
moff_A_err = np.zeros((len(x_PNe), len(emission_dict)))
x_0_err = np.zeros((len(x_PNe), len(emission_dict)))
y_0_err = np.zeros((len(x_PNe), len(emission_dict)))
mean_wave_err = np.zeros((len(x_PNe), len(emission_dict)))
Gauss_bkg_err = np.zeros((len(x_PNe), len(emission_dict)))
Gauss_grad_err = np.zeros((len(x_PNe), len(emission_dict)))

list_of_x = np.zeros(len(x_PNe))
list_of_y = np.zeros(len(x_PNe))
Gauss_bkg = np.zeros(len(x_PNe))
Gauss_grad = np.zeros(len(x_PNe))


# Define a function that contains all the steps needed for fitting, including the storage of important values, calculations and pandas assignment.
def run_minimiser(parameters):
    for PNe_num in tqdm(np.arange(0, len(x_PNe))):
        #progbar(int(PNe_num)+1, len(x_PNe), 40)
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
    gauss_list, redchi, Chi_sqr = [], [], []
    for p in range(len(x_PNe)):
        PNe_n = np.copy(PNe_spectra[p])
        flux_1D = np.copy(F_xy_list[p][0])
        A_n = ((flux_1D) / (np.sqrt(2*np.pi) * (galaxy_data["LSF"]// 2.35482)))
    
        def gaussian(x, amplitude, mean, FWHM, bkg, grad):
            stddev = FWHM/ 2.35482
            return ((bkg + grad*x) + np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.)) +
                    (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / (stddev**2.)))
    
        list_of_gauss = [gaussian(wavelength, A, mean_wave_list[p][0], galaxy_data["LSF"], Gauss_bkg[p], Gauss_grad[p]) for A in A_n]
        for kk in range(len(PNe_n)):
            temp = np.copy(list_of_gauss[kk])
            idx  = np.where(PNe_n[kk] == 0.0)[0]
            temp[idx] = 0.0
            PNe_n[kk,idx] = 1.0
            list_of_gauss[kk] = np.copy(temp)
        rN   = robust_sigma(PNe_n - list_of_gauss)
        res  = PNe_n - list_of_gauss
        Chi2 = np.sum((res**2)/(rN**2))
        s    = np.shape(PNe_n)
        redchi.append(Chi2/ ((len(wavelength) * n_pixels**2) - multi_fit_results.nvarys))
        gauss_list.append(list_of_gauss)
        Chi_sqr.append(Chi2)
    
    PNe_df['Chi2']   = Chi_sqr
    PNe_df["redchi"] = redchi
    
    # velocity
    de_z_means = np.array(mean_wave_list[:,0] / (1 + z)) # de redshift OIII wavelength position
    
    PNe_df["V (km/s)"] = (c * (de_z_means - 5006.77) / 5006.77) / 1000.    
        
    PNe_df["[OIII] Flux"] = total_Flux[:,0] #store total OIII 5007 line flux
        
    if "hb" in emission_dict:
        PNe_df["[OIII]/Hb"] = PNe_df["[OIII] Flux"] / total_Flux[:,2] # store [OIII]/Hb ratio

    if "ha" in emission_dict:
        PNe_df["Ha Flux"] = total_Flux[:, 1]
    
    PNe_df["m 5007"] = -2.5 * np.log10(PNe_df["[OIII] Flux"].values) - 13.74
    
    

gen_params(wave=5006.77*(1+z), FWHM=galaxy_data["FWHM"], beta=galaxy_data["beta"], LSF=galaxy_data["LSF"], vary_PSF=False, em_dict=emission_dict)    
run_minimiser(PNe_multi_params)

############################ The Great Filter ########################################
PNe_df["Filter"] = "Y"
PNe_df.loc[PNe_df["A/rN"]<3.0, "Filter"] = "N"
# reduced Chi sqr cut
upper_chi = chi2.ppf(0.9973, ((n_pixels**2)*len(wavelength))-6) # 3 sigma = 0.9973
PNe_df.loc[PNe_df["Chi2"]>=upper_chi, "Filter"] = "N" 

## New filter read in area

# read in 3 different filter lists:
# my_filter = []
# impostor_filter = [[],[],[]]
# interloper_filter = []


# list of objects that are chosen to be filtered out (bad fits, objviously not PN, over luminous, etc.)
my_filter = galaxy_data["my_filter"]

# Supernova remnants, HII regions and unknown impostor lists
SNR_filter, HII_filter, unknown_imp_filter = galaxy_data["impostor_filter"]

# Interloping objects list
interloper_filter  = galaxy_data["interloper_filter"]

## Apply filter

[PNe_df.loc[PNe_df["PNe number"] == n, "Filter"] = "N" for n in my_filter]
[PNe_df.loc[PNe_df["PNe number"] == n, "Filter"] = "N" for n in SNR_filter]
[PNe_df.loc[PNe_df["PNe number"] == n, "Filter"] = "N" for n in HII_filter]
[PNe_df.loc[PNe_df["PNe number"] == n, "Filter"] = "N" for n in unknown_imp_filter]
[PNe_df.loc[PNe_df["PNe number"] == n, "Filter"] = "N" for n in interloper_filter]

##

## Current exclusion list - re-write to use yaml file
# ## FCC167
# if (galaxy_name == "FCC167") & (loc=="center"):
#     PNe_df.loc[PNe_df["PNe number"]==32, "Filter"] = "N" # Over luminous PNe 
#     PNe_df.loc[PNe_df["PNe number"]==10, "Filter"] = "N"  # Double reading from source
# ## FCC219
# elif (galaxy_name == "FCC219") & (loc=="center"):
#      PNe_df.loc[PNe_df["PNe number"]==0, "Filter"] = "N"
# elif galaxy_name == "FCC193":
#     PNe_df.loc[PNe_df["PNe number"]==143, "Filter"] = "N" 
#     PNe_df.loc[PNe_df["PNe number"]==141, "Filter"] = "N" 
#     PNe_df.loc[PNe_df["PNe number"]==84, "Filter"] = "N"
#     PNe_df.loc[PNe_df["PNe number"]==77, "Filter"] = "Y" 
#     PNe_df.loc[PNe_df["PNe number"]==94, "Filter"] = "Y" 
# #elif galaxy_name == "FCC147":
#     #PNe_df.loc[PNe_df["PNe number"]==41, "Filter"] = "N"
# # elif galaxy_name == "FCC249":
# #     PNe_df.loc[PNe_df["PNe number"]==2, "Filter"] = "N"
# elif galaxy_name == "FCC276":
#     PNe_df.loc[PNe_df["PNe number"]==20, "Filter"] = "N" # Overly bright object, sets D=15Mpc, could be overlap/super-position of two.
#     PNe_df.loc[PNe_df["PNe number"]==40, "Filter"] = "Y"
#     PNe_df.loc[PNe_df["PNe number"]==79, "Filter"] = "Y"
#     PNe_df.loc[PNe_df["PNe number"]==85, "Filter"] = "Y"
# # elif galaxy_name == "FCC184":
# #     PNe_df.loc[PNe_df["PNe number"]==15, "Filter"] = "N"
# #     PNe_df.loc[PNe_df["PNe number"]==35, "Filter"] = "N"
# elif galaxy_name == "FCC301":
#     PNe_df.loc[PNe_df["PNe number"]==14, "Filter"] = "N"
#     PNe_df.loc[PNe_df["PNe number"]==16, "Filter"] = "N"
# elif galaxy_name == "FCC255":
#     PNe_df.loc[PNe_df["PNe number"]==32, "Filter"] = "N"


## End of exclusion list
    
    
    
##### Error estimation #####
def Moffat_err(Moff_A, FWHM, beta, x_0, y_0):
    alpha = FWHM / (2. * umath.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2 + (y_fit - y_0)**2) / alpha**2
    F_OIII_xy_dist = Moff_A * (2 * ((beta -1)/(alpha**2)))*(1 + rr_gg)**(-beta)
    
    return np.sum(F_OIII_xy_dist*1e-20)

flux_plus_minus = np.ones((len(x_PNe),2))
mag_plus_minus  = np.ones((len(x_PNe),2))
                         
for i,p in enumerate(tqdm(range(len(x_PNe)))):
    Moff_A_dist = N(moff_A[p][0], moff_A_err[p][0])
    FWHM_dist   = N(galaxy_data["FWHM"], galaxy_data["FWHM_err"])
    beta_dist   = N(galaxy_data["beta"], galaxy_data["beta_err"])
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
# 
PNe_df["mag error up"] = mag_plus_minus[:,0]
PNe_df["mag error lo"] = mag_plus_minus[:,1]

#### Distance estimation #######
p_n = int(PNe_df.loc[PNe_df["Filter"]=="Y"].nsmallest(1, "m 5007").index.values)

m = PNe_df["m 5007"].iloc[p_n]
m_err_up = PNe_df["mag error up"].iloc[p_n]
m_err_lo = PNe_df["mag error lo"].iloc[p_n]
print("PNe: ", p_n)

M_star = -4.53
M_star_err = 0.08
D_diff_eq = 0.2 * np.log(10) * (10**(0.2*(m + 4.52 - 25)))

Dist_est = 10.**(((m + 4.52) -25.) / 5.)
Dist_err_up = np.sqrt((D_diff_eq**2 * m_err_up**2) + ((-D_diff_eq)**2 * M_star_err**2))
Dist_err_lo = np.sqrt((D_diff_eq**2 * m_err_lo**2) + ((-D_diff_eq)**2 * M_star_err**2))

print("Distance Estimate from PNLF: ", f"{np.round(Dist_est,3)} (+ {np.round(Dist_err_up,3)}) (- {np.round(Dist_err_lo,3)}) Mpc")

dM =  5. * np.log10(Dist_est) + 25.
dM_diff_eq = 5/(np.log(10) * Dist_est)
dM_err_up = np.abs(dM_diff_eq)*Dist_err_up
dM_err_lo = np.abs(dM_diff_eq)*Dist_err_lo

print(f"dM = {np.round(dM, 3)} (+ {np.round(dM_err_up,3)}) (- {np.round(dM_err_lo, 3)})")

PNe_df["M 5007"] = PNe_df["m 5007"] - dM

# save the pandas data-frame for use in the impostor diagnostics.
PNe_df.to_csv(f"exported_data/{galaxy_name}/{galaxy_name}_PNe_df.csv")

### Construct Table of filtered PNe, ready for paper
# Construct a Astropy table to save certain values for each galaxy.
y_idx = PNe_df.loc[PNe_df["Filter"]=="Y"].index.values
PNe_table = Table([list(PNe_df.loc[PNe_df["Filter"]=="Y"].index), PNe_df["Ra (J2000)"].loc[PNe_df["Filter"]=="Y"], PNe_df["Dec (J2000)"].loc[PNe_df["Filter"]=="Y"],
                   PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].round(2),
                   PNe_df["A/rN"].loc[PNe_df["Filter"]=="Y"].round(1),],
                   names=("PNe number", "Ra", "Dec", "m 5007", "A/rN"))


# Save table in tab separated format.
ascii.write(PNe_table, f"exported_data/{galaxy_name}/{galaxy_name}_fit_results.txt", format="tab", overwrite=True) 
# Save latex table of data.
ascii.write(PNe_table, f"exported_data/{galaxy_name}/{galaxy_name}_fit_results_latex.txt", format="latex", overwrite=True) 

###### Plot the FOV with PNe circled
A_rN_plot = np.load(EXPORT_DIR+galaxy_name+"_A_rN_cen.npy")
A_rN_plot_shape = A_rN_plot.reshape(y_data, x_data)

with fits.open(RAW_DAT) as hdu_wcs:
    hdr_wcs = hdu_wcs[1].header
    wcs_obj = WCS(hdr_wcs, naxis=2)

plt.figure(figsize=(15,15))
plt.axes(projection=wcs_obj)
plt.imshow(A_rN_plot_shape, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8)
ax = plt.gca()
RA = ax.coords[0]
DEC = ax.coords[1]


cb=plt.colorbar(fraction=0.0455, pad=0.04)
cb.set_label("A/rN",fontsize=30)
cb.ax.tick_params(labelsize=22)


plt.xlabel("RA (J2000)", fontsize=30)
plt.ylabel("DEC (J2000)", fontsize=30)

plt.tick_params(labelsize = 22)

Y, X = np.mgrid[:y_data, :x_data]
xe, ye, length, width, alpha = gal_mask_params

if (galaxy_name=="FCC219") & (loc=="center"):
    plt.ylim(0,440)
    plt.xlim(0,440);
if (galaxy_name=="FCC219") & (loc=="halo"):
    plt.ylim(350,)
#     plt.xlim(440,);
elif galaxy_name=="FCC193":
    plt.ylim(250,)
    plt.xlim(0,350)
elif galaxy_name=="FCC161":
    plt.xlim(0,450)
elif galaxy_name=="FCC147":
    plt.xlim(230,)
    plt.ylim(0,320)
elif galaxy_name=="FCC083":
    plt.xlim(0,370)
    plt.ylim(0,370)
elif galaxy_name=="FCC310":
    plt.xlim(0,410)
    plt.ylim(100,)
elif galaxy_name=="FCC276":
    plt.xlim(310,)
elif galaxy_name=="FCC184":
    plt.xlim(0,450)
    plt.ylim(0,450)

elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=False, color="grey", ls="--")
ax.add_artist(elip_gal)

for star in star_mask_params:
    ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=False, color="grey", ls="--"))


for i, item in enumerate(x_y_list):
    if PNe_df.iloc[i].Filter == "Y":
        ax = plt.gca()
        circ = plt.Circle((item[0],item[1]),5, color="black",lw=1.2, fill=False, alpha=0.8)
        ax.add_artist(circ)
    elif PNe_df.iloc[i].Filter == "N":
        ax = plt.gca()
        circ = plt.Circle((item[0],item[1]),4, color="red",lw=1., fill=False, alpha=0.8)
    ax.add_artist(circ)
#     if item[0]<240.:
#        ax.annotate(i, (item[0]+6, item[1]-2), color="black", size=15)
#     else:
#        ax.annotate(i, (item[0]+6, item[1]+1), color="black", size=15)

# plt.arrow(400,380, 0,30, head_width=5, width=0.5, color="k")
# plt.annotate("N", xy=(395, 420), fontsize=25)
# plt.arrow(400,380, -20,0, head_width=5, width=0.5, color="k")
# plt.annotate("E", xy=(360, 375), fontsize=25)

plt.savefig(PLOT_DIR+"_A_rN_circled.png", bbox_inches='tight')
plt.savefig(PLOT_DIR+"_A_rN_circled.pdf", bbox_inches='tight')

# #####################################################
# ####################### PNLF ########################
# #####################################################


galaxy_image, wave = reconstructed_image(galaxy_name, loc)
galaxy_image = galaxy_image.reshape([y_data, x_data])

PNe_mag = PNe_df["M 5007"].loc[PNe_df["Filter"]=="Y"].values
app_mag = PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].values

# Total PNLF
PNLF, PNLF_corr, completeness_ratio, Abs_M, app_m = completeness(galaxy_name, loc, PNe_mag, PNe_multi_params, Dist_est, galaxy_image, peak=3.0,
                                      gal_mask_params=gal_mask_params, star_mask_params=star_mask_params, c1=0.307, z=z ) # Estimating the completeness for the central pointing

step = abs(Abs_M[1]-Abs_M[0])
# Getting the normalisation - sum of correctied PNLF, times bin size
total_norm = np.sum(np.abs(PNLF_corr)) * step

# Scaling factor
scal = len(PNe_mag) / total_norm

# Constraining to -2.5 in magnitude
idx = np.where(Abs_M <= np.min(PNe_mag)+2.5)

plt.figure(figsize=(14,10))

binwidth = 0.2

# hist = plt.hist(PNe_mag, bins=np.arange(min(PNe_mag), max(PNe_mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')
hist = plt.hist(app_mag, bins=np.arange(min(app_mag), max(app_mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')

KS2_stat = KS2_test(dist_1=PNLF_corr[1:18:2]*scal*binwidth, dist_2=hist[0], conf_lim=0.1)
print(KS2_stat)

ymax = max(hist[0])

plt.plot(app_m, PNLF*scal*binwidth, '-', color='blue', marker="o", label="PNLF")
plt.plot(app_m, PNLF_corr*scal*binwidth,'-.', color='blue', label="Completeness corrected PNLF")
# plt.plot(Abs_M, completeness_ratio*200*binwidth, "--", color="k", label="completeness")
plt.xlabel(r'$m_{5007}$', fontsize=30)
plt.ylabel(r'$N_{PNe}$', fontsize=30)
#plt.yticks(np.arange(0,ymax+4, 5))
plt.plot(0,0, alpha=0.0, label=f"KS2 test = {round(KS2_stat[0],3)}")
plt.plot(0,0, alpha=0.0, label=f"pvalue   = {round(KS2_stat[1],3)}")
plt.xlim(-5.0+dM,-1.5+dM); 
plt.ylim(0,ymax+(2*ymax));
# plt.xlim(26.0,30.0); plt.ylim(0,45);

plt.tick_params(labelsize = 25)

plt.legend(loc=2, fontsize=20)
# plt.savefig(PLOT_DIR+"_PNLF.pdf", bbox_inches='tight')
# plt.savefig(PLOT_DIR+"_PNLF.png", bbox_inches='tight')

N_PNe = np.sum(PNLF[idx]*scal) * step

print("Number of PNe from PNLF: ", N_PNe, "+/-", (1/np.sqrt(len(PNe_df.loc[PNe_df["Filter"]=="Y"])))*N_PNe)


##### Integrated, bolometric Luminosity of galaxy FOV spectra #####
raw_data_cube = RAW_DAT # read in raw data cube

xe, ye, length, width, alpha = gal_mask_params

orig_hdulist = fits.open(raw_data_cube)
raw_data_cube = np.copy(orig_hdulist[1].data)
h1 = orig_hdulist[1].header
s = np.shape(orig_hdulist[1].data)
Y, X = np.mgrid[:s[1], :s[2]]
elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1    

# Now mask the stars
star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in star_mask_params],0).astype(bool)

total_mask = ((np.isnan(orig_hdulist[1].data[1,:,:])==False) & (elip_mask==False) & (star_mask_sum==False))
indx_mask = np.where(total_mask==True)

good_spectra = np.zeros((s[0], len(indx_mask[0])))

for i, (y, x)  in enumerate(zip(tqdm(indx_mask[0]), indx_mask[1])):
    good_spectra[:,i] = raw_data_cube[:,y,x]

print("Collapsing cube now....")    
    
gal_lin = np.nansum(good_spectra, 1)
        
print("Cube has been collapsed...")

L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_vel, dist_mod=dM, dM_err=[dM_err_up, dM_err_lo])


plt.show()

print("Number of PNe after A/rN cut: ", len(PNe_df["Filter"].loc[PNe_df["Filter"]=="Y"]))

print("Number of PNe after A/rN and Reduced chi-square cuts: ", len(PNe_df["Filter"].loc[PNe_df["Filter"]=="Y"]))


print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table.txt")
print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table_latex.txt")

print(galaxy_name)
n_p = len(PNe_df.loc[PNe_df["Filter"]=="Y"])
print(f"N PNe used:      {n_p}")
print(f"PNLF N:          {N_PNe}")
print(f"L_bol of:        {L_bol[0]}")
print(f"L_bol error:     + {L_bol[1][0] - L_bol[0]}, - {L_bol[0] - L_bol[1][1]}")
print(f"Rmag of :        {L_bol[7]}")
print(f"Vmag of :        {L_bol[8]}")
print(f"Distance of:     {Dist_est} +/- {Dist_err_up}")
print(f"Distance Mod of: {dM} +/- {dM_err_up}")

# print("This is the end of PNe analysis script. Goodbye")
