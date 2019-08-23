import yaml
import lmfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import sys
from scipy.stats import norm, chi2
from scipy import stats
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from ppxf_gal_L import ppxf_L_tot
from matplotlib.patches import Rectangle, Ellipse, Circle
from PNLF import open_data, reconstructed_image, completeness
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
from MUSE_Models import PNe_residuals_3D, PNe_spectrum_extractor, PSF_residuals_3D, robust_sigma


# Load in yaml file to query galaxy properties
with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

# Queries user for a galaxy name, in the form of FCC000, and taking the relevant info from the yaml file
galaxy_name = sys.argv[1]#input("Please type which Galaxy you want to analyse, use FCC000 format: ")
galaxy_data = galaxy_info[galaxy_name]

RAW_DIR    = "/local/tspriggs/Fornax_data_cubes/"+galaxy_name
DATA_DIR   = "galaxy_data/"+galaxy_name+"_data/"
EXPORT_DIR = "exported_data/"+galaxy_name+"/"
PLOT_DIR   = "Plots/"+galaxy_name+"/"+galaxy_name

# Load in the residual data, in list form
hdulist = fits.open(DATA_DIR+galaxy_name+"_residuals_list.fits")
res_hdr = hdulist[0].header # extract header from residual cube

# Check to see if the wavelength is in the fits fileby checking length of fits file.
if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
    wavelength = np.exp(hdulist[1].data)
else:
    wavelength = np.load(DATA_DIR+"_wavelength.npy")

x_data = res_hdr["XAXIS"]
y_data = res_hdr["YAXIS"]

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

gal_vel = galaxy_data["velocity"] 
z = gal_vel*1e3 / c 
D = galaxy_data["Distance"] # Distance in Mpc - from Simbad / NED - read in from yaml file
gal_mask = galaxy_data["gal_mask"]
star_mask = galaxy_data["star_mask"]

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
    hdu_wcs = fits.open(RAW_DIR+"center.fits")
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
            PNe_multi_params.add("wave_{}".format(em), value=wave, min=wave-15., max=wave+15.)
        else:
            PNe_multi_params.add("wave_{}".format(em), expr=emission_dict[em][2].format(z))
    
    PNe_multi_params.add("x_0", value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    PNe_multi_params.add("y_0", value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    PNe_multi_params.add("LSF", value=LSF, vary=vary_LSF, min=LSF-1, max=LSF+1)
    PNe_multi_params.add("M_FWHM", value=FWHM, min=FWHM - FWHM_err, max=FWHM + FWHM_err, vary=vary_PSF)
    PNe_multi_params.add("beta", value=beta, min=beta - beta_err, max=beta + beta_err, vary=vary_PSF)   
    PNe_multi_params.add("Gauss_bkg",  value=0.1, vary=True)#1, min=-200, max=500)
    PNe_multi_params.add("Gauss_grad", value=0.0001, vary=True)#1, min=-2, max=2)
    
# storage setup
total_Flux = np.zeros((len(x_PNe),len(emission_dict)))
A_2D_list = np.zeros((len(x_PNe),len(emission_dict)))
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
    
    


gen_params(wave=5006.77*(1+z), FWHM=galaxy_data["FWHM"], beta=galaxy_data["beta"], LSF=galaxy_data["LSF"], em_dict=emission_dict)
    
run_minimiser(PNe_multi_params)


## The Great Filter #####
PNe_df["Filter"] = "Y"
PNe_df.loc[PNe_df["A/rN"]<3., "Filter"] = "N"
# reduced Chi sqr cut
upper_chi = chi2.ppf(0.9973, (9*9*len(wavelength))-6) # 3 sigma = 0.9973
PNe_df.loc[PNe_df["Chi2"]>=upper_chi, "Filter"] = "N" 




## Current exclusion list
if galaxy_name == "FCC167":
    PNe_df.loc[PNe_df["PNe number"]==29, "Filter"] = "N"
    PNe_df.loc[PNe_df["PNe number"]==15, "Filter"] = "N"
    PNe_df.loc[PNe_df["PNe number"]==8, "Filter"] = "N"
## FCC219
# elif galaxy_name == "FCC219":
#     PNe_df.loc[PNe_df["PNe number"]==11, "Filter"] = "N"
elif galaxy_name == "FCC193":
    PNe_df.loc[PNe_df["PNe number"]==141, "Filter"] = "N" 
    PNe_df.loc[PNe_df["PNe number"]==143, "Filter"] = "N" 
    PNe_df.loc[PNe_df["PNe number"]==84, "Filter"] = "N"
    PNe_df.loc[PNe_df["PNe number"]==94, "Filter"] = "Y" 
    PNe_df.loc[PNe_df["PNe number"]==77, "Filter"] = "Y" 
    
# elif galaxy_name == "FCC147":
#     PNe_df.loc[PNe_df["PNe number"]==41, "Filter"] = "N"
elif galaxy_name == "FCC249":
    PNe_df.loc[PNe_df["PNe number"]==2, "Filter"] = "N"
elif galaxy_name == "FCC276":
    PNe_df.loc[PNe_df["PNe number"]==20, "Filter"] = "N" # Overly bright object, sets D=15Mpc, could be overlap/super-position of two.
    PNe_df.loc[PNe_df["PNe number"]==40, "Filter"] = "Y"
    PNe_df.loc[PNe_df["PNe number"]==79, "Filter"] = "Y"
    PNe_df.loc[PNe_df["PNe number"]==85, "Filter"] = "Y"
elif galaxy_name == "FCC184":
    PNe_df.loc[PNe_df["PNe number"]==15, "Filter"] = "N"
    PNe_df.loc[PNe_df["PNe number"]==35, "Filter"] = "N"
elif galaxy_name == "FCC301":
    PNe_df.loc[PNe_df["PNe number"]==14, "Filter"] = "N"
    PNe_df.loc[PNe_df["PNe number"]==16, "Filter"] = "N"
elif galaxy_name == "FCC255":
    PNe_df.loc[PNe_df["PNe number"]==32, "Filter"] = "N"
    
## End of exclusion list
    
    
    
##### Error estimation #####
def Moffat_err(Moff_A, FWHM, beta, x_0, y_0):
    if beta <0.01:
        beta = 0.01
    gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((np.array(x_fit) - x_0)**2 + (np.array(y_fit) - y_0)**2) / gamma**2
    F_OIII_xy_dist = Moff_A * (1 + rr_gg)**(-beta)
    
    return np.sum(F_OIII_xy_dist*1e-20)
    
def error_sim(n_sim, n_PNe):
    mean_flux = np.ones(n_PNe)
    flux_err  = np.ones(n_PNe)
    mean_m5007 = np.ones(n_PNe)
    m5007_err = np.ones(n_PNe)
    for n in np.arange(0,len(x_PNe)):
        Moff_A_dist = np.random.normal(moff_A[n][0], moff_A_err[n][0],n_sim)
        FWHM_dist = np.random.normal(galaxy_data["FWHM"], galaxy_data["FWHM_err"], n_sim)
        beta_dist = np.abs(np.random.normal(galaxy_data["beta"], galaxy_data["beta_err"], n_sim))
        x_0_dist = np.random.normal(list_of_x[n], x_0_err[n][0], n_sim)
        y_0_dist = np.random.normal(list_of_y[n], y_0_err[n][0], n_sim)
        
        flux_array = [Moffat_err(Moff_A_dist[i], FWHM_dist[i], beta_dist[i], x_0_dist[i], y_0_dist[i]) for i in range(0,n_sim)]
#         flux_array = [Moffat_err(Moff_A_dist[i], galaxy_data["FWHM"], galaxy_data["beta"], x_0_dist[i], y_0_dist[i]) for i in range(0,n_sim)]

        mean_flux[n] = np.median(flux_array)
        flux_err[n] = mean_flux[n] - np.median([np.percentile(flux_array, 16), np.percentile(flux_array, 84)])
        #norm.fit(flux_array)
                
        mean_m5007[n] = np.median(-2.5*np.log10(flux_array)-13.72)
        m5007_err[n]  = mean_m5007[n] - np.median([np.percentile((-2.5*np.log10(flux_array)-13.72), 16),
                                   np.percentile((-2.5*np.log10(flux_array)-13.72), 84)])

    return mean_flux, flux_err, mean_m5007, m5007_err

mean_flux, PNe_df["Flux error"], mean_m5007, PNe_df["m 5007 error"] = error_sim(5000, len(x_PNe))

## Show F_err in percentage terms
PNe_df["F[OIII] err percent"] = (PNe_df["Flux error"] / PNe_df["[OIII] Flux"])*100

#### Distance estimation #######
brightest_PN = int(PNe_df.loc[PNe_df["Filter"]=="Y"].nsmallest(1, "m 5007").index.values)
flux = PNe_df["[OIII] Flux"].iloc[brightest_PN]
flux_err = PNe_df["Flux error"].iloc[brightest_PN]
m = PNe_df["m 5007"].iloc[brightest_PN]
m_err = PNe_df["m 5007 error"].iloc[brightest_PN]

M_star = -4.52 # Ciardullo cutoff mag (2102)
M_star_err = 0.08
D_diff_eq = 0.2 * np.log(10) * (10**(0.2*(m + 4.52 - 25)))

# galaxy_df = pd.read_csv("exported_data/galaxy_dataframe.csv")

Dist_est = 10.**(((m + 4.52) -25.) / 5.)
# Dist_est = galaxy_df.loc[galaxy_df["Galaxy"]==galaxy_name, "lit D"].values[0] #Use lit values
Dist_err = np.sqrt((D_diff_eq**2 * m_err**2) + ((-D_diff_eq)**2 * M_star_err**2))

dM         =  5. * np.log10(Dist_est) + 25.
dM_diff_eq = 5 / (np.log(10) * Dist_est)
dM_err     = np.abs(dM_diff_eq)*Dist_err


PNe_df["M 5007"] = PNe_df["m 5007"] - dM


### Construct Table of filtered PNe, ready for paper
# Construct a Astropy table to save certain values for each galaxy.
y_idx = PNe_df.loc[PNe_df["Filter"]=="Y"].index.values
PNe_table = Table([np.arange(0,len(y_idx)), PNe_df["Ra (J2000)"].loc[PNe_df["Filter"]=="Y"], PNe_df["Dec (J2000)"].loc[PNe_df["Filter"]=="Y"],
                   PNe_df["[OIII] Flux"].loc[PNe_df["Filter"]=="Y"].round(20),
                   PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].round(2),
                   PNe_df["A/rN"].loc[PNe_df["Filter"]=="Y"].round(1),],
                   names=("PNe number", "Ra", "Dec", "[OIII] Flux", "m 5007", "A/rN"))

# Save table in tab separated format.
ascii.write(PNe_table, f"exported_data/{galaxy_name}/{galaxy_name}_fit_results.txt", format="tab", overwrite=True) 
# Save latex table of data.
ascii.write(PNe_table, f"exported_data/{galaxy_name}/{galaxy_name}_fit_results_latex.txt", format="latex", overwrite=True) 


###### Plotting
A_rN_plot = np.load(EXPORT_DIR+galaxy_name+"_A_rN_cen.npy")
A_rN_plot_shape = A_rN_plot.reshape(y_data, x_data)

plt.figure(figsize=(20,20))
plt.imshow(A_rN_plot_shape, origin="lower", cmap="CMRmap_r",  vmin=1.5, vmax=8)
ax = plt.gca()
cb=plt.colorbar(fraction=0.0455, pad=0.04)
cb.set_label("A/rN",fontsize=30)
cb.ax.tick_params(labelsize=22)
cb.ax.invert_yaxis()

plt.xlabel("x (pixels)", fontsize=30)
plt.ylabel("y (pixels)", fontsize=30)

plt.tick_params(labelsize = 22)

Y, X = np.mgrid[:y_data, :x_data]
xe, ye, length, width, alpha = gal_mask

if galaxy_name=="FCC219":
    plt.ylim(0,440)
    plt.xlim(0,440)
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

for star in star_mask:
    ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=False, color="grey", ls="--"))



for i, item in enumerate(x_y_list):
    if PNe_df.iloc[i].Filter == "Y":
        ax = plt.gca()
        circ = plt.Circle((item[0],item[1]),5, color="black",lw=1.2, fill=False, alpha=0.8)
        ax.add_artist(circ)
    elif PNe_df.iloc[i].Filter == "N":
        ax = plt.gca()
        circ = plt.Circle((item[0],item[1]),4, color="grey",lw=1.5, fill=False, alpha=0.8)
    ax.add_artist(circ)
    #if item[0]<240.:
    #    ax.annotate(i, (item[0]+4, item[1]-2), color="white", size=10)
    #else:
    #    ax.annotate(i, (item[0]+4, item[1]+1), color="white", size=10)

plt.savefig(PLOT_DIR+"_A_rN_circled.png", bbox_inches='tight')

    
# #####################################################
# ####################### PNLF ########################
# #####################################################


x_data_cen, y_data_cen, map_cen, aux = open_data(galaxy_name)

image, wave = reconstructed_image(galaxy_name)
image = image.reshape([y_data_cen,x_data_cen])

mag = PNe_df["M 5007"].loc[PNe_df["Filter"]=="Y"].values

# Total PNLF
PNLF, PNLF_corr, Abs_M = completeness(galaxy_name, mag, PNe_multi_params, Dist_est, image, peak=3.,
                                      gal_mask_params=gal_mask, star_mask_params=star_mask, c1=0.307, z=z )
# Getting the normalisation
total_norm = np.sum(PNLF_corr)*abs(Abs_M[1]-Abs_M[0])

# Scaling factor
scal = len(mag)/total_norm

# Constraining to -2.0 in magnitude
idx = np.where(Abs_M <= -2.0)
# Total number of PNe
tot_N_PNe = np.sum(PNLF_corr[idx]*scal)*abs(Abs_M[1]-Abs_M[0])

step = Abs_M[1]-Abs_M[0]
N_PNe = np.sum(PNLF[:25]*scal*step)

plt.figure(figsize=(18,16))

binwidth = 0.2
hist = plt.hist(mag, bins=np.arange(min(mag), max(mag) + binwidth, binwidth), edgecolor="black", linewidth=0.8, alpha=0.5, color='blue')

ymax = max(hist[0])

plt.plot(Abs_M, PNLF*scal*binwidth, '-', color='blue', marker="o", label="PNLF")
plt.plot(Abs_M, PNLF_corr*scal*binwidth,'-.', color='blue', label="Completeness corrected PNLF")

plt.xlabel('$M_{5007}$', fontsize=26)
plt.ylabel('N PNe', fontsize=26)
#plt.yticks(np.arange(0,ymax+4, 5))

plt.xlim(-5.0,-2.0);
plt.ylim(0,ymax+10);
plt.tick_params(labelsize = 22)

# #plt.axvline(PNe_df["m 5007"].loc[PNe_df["Filter"]=="Y"].values.min() - 31.63)
plt.legend(loc=2, fontsize=20)
plt.savefig(PLOT_DIR+"_PNLF.png", bbox_inches='tight')


##### Integrated, bolometric Luminosity of galaxy FOV spectra #####
plt.figure()
raw_data_cube = RAW_DIR+"center.fits" # read in raw data cube

xe, ye, length, width, alpha = gal_mask

orig_hdulist = fits.open(raw_data_cube)
raw_data_cube = np.copy(orig_hdulist[1].data)
h1 = orig_hdulist[1].header
s = np.shape(orig_hdulist[1].data)
Y, X = np.mgrid[:s[1], :s[2]]
elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1    

# Now mask the stars
star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in star_mask],0).astype(bool)
    
    
total_mask = ((np.isnan(orig_hdulist[1].data[1,:,:])==False) & (elip_mask==False) & (star_mask_sum==False))
indx_mask = np.where(total_mask==True)

good_spectra = np.zeros((s[0], len(indx_mask[0])))

for i, (y, x)  in enumerate(zip(tqdm(indx_mask[0]), indx_mask[1])):
    good_spectra[:,i] = raw_data_cube[:,y,x]

print("Collapsing cube now....")    
    
gal_lin = np.nansum(good_spectra, 1)
        
print("Cube has been collapsed...")


L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_vel, dist_mod=dM)

# ##### Alpha value calculation #####
alpha_2_5 = N_PNe/L_bol
log_alpha = np.log10(alpha_2_5)

plt.show()
print("Number of PNe after A/rN cut: ", len(PNe_df["Filter"].loc[PNe_df["Filter"]=="Y"]))

print("Number of PNe after A/rN and Reduced chi-square cuts: ", len(PNe_df["Filter"].loc[PNe_df["Filter"]=="Y"]))


print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table.txt")
print(f"File saved: exported_data/{galaxy_name}/{galaxy_name}_table_latex.txt")

n_p = len(PNe_df.loc[PNe_df["Filter"]=="Y"])
print(f"N PNe used:      {n_p}")
print(f"PNLF N:          {N_PNe}")
print(f"L_bol of:        {L_bol}")
print(f"Distance of:     {Dist_est} +/- {Dist_err}")
print(f"Distance Mod of: {dM} +/- {dM_err}")

# print("This is the end of PNe analysis script. Goodbye")
