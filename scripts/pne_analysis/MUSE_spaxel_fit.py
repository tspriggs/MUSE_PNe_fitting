import sep
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from astropy.io import fits
from photutils import CircularAperture

from matplotlib.patches import Ellipse, Circle
from lmfit import minimize, Parameters

# Load in local functions
from functions.MUSE_Models import spaxel_by_spaxel
from functions.PNe_functions import PNe_minicube_extractor, uncertainty_cube_construct, generate_mask
from functions.file_handling import paths, open_data


# Queries sys arguments for galaxy name
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--galaxy', action='store', type=str, required=True, 
                        help="The name of the galaxy to be analysed.")
my_parser.add_argument("--loc", action="store", type=str, required=False, default="", 
                        help="The pointing location, e.g. center, halo or middle")
my_parser.add_argument("--fit", action="store_true", default=False, 
                        help="This flag is needed when you want to run the spaxel-by-spaxel fit to the residual MUSE cube.")
my_parser.add_argument("--sep", action="store_true", default=False, 
                        help="The sep flag makes the script save the output PNe locations, as detected using the SEP package.")

args = my_parser.parse_args()

galaxy_name = args.galaxy
loc = args.loc
fit_spaxel = args.fit
save_sep = args.sep

DIR_dict = paths(galaxy_name, loc)

# To be used when working with residual cubes
res_cube, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy_name, loc, DIR_dict)


# reshape the residual cube into a residual list, so as to work with current code
res_data_list = res_cube.reshape(len(wavelength),x_data*y_data)
res_data_list = np.swapaxes(res_data_list, 1, 0)
n_spax = np.shape(res_data_list)[0]


# Indexes where there is spectral data to fit. We check where there is data that doesn't start with 0.0 (spectral data should never be 0.0).
non_zero_index = np.squeeze(np.where(res_cube[1,:,:] != 0.)) # use with residual cube


# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

z = galaxy_info["velocity"]*1e3 / c    

gal_mask = galaxy_info["gal_mask"]

# Construct the PNe FOV coordinate grid for use when fitting PNe.
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])


################################################################################
#################################### Fit 1D ####################################
################################################################################

# Run Spaxel by Spaxel fit of the spectra within the .fits file.
# Check if fit_1D parameter, within the Galaxy_info.yaml file is set to Y (yes to fit), or N (no to fit - has been fitted before).
print("Spaxel by Spaxel fit underway...")

# Run Spaxel by Spaxel fitter
print("Fitting Spaxel by Spaxel for [OIII] doublet.")

# list_of_std = np.abs(np.std(res_cube, 0)
list_of_std = np.abs([np.nanstd(dat) for dat in res_data_list])
input_errors = [np.repeat(item, len(wavelength)) for item in list_of_std] # Intially use the standard deviation of each spectra as the uncertainty for the spaxel fitter.

# Setup numpy arrays for storage of best fit values.
g_bkg  = np.zeros(n_spax)
g_grad = np.zeros(n_spax)
g_FWHM = np.zeros(n_spax)
gauss_A = np.zeros(n_spax)
list_of_rN = np.zeros(n_spax)
list_of_mean = np.zeros(n_spax)
obj_residuals = np.zeros((n_spax, len(wavelength)))
data_residuals = np.zeros((n_spax, len(wavelength)))
list_of_models = np.zeros((n_spax, len(wavelength)))

# setup LMfit paramterts
expect_em_line_pos = 5006.77*(1+z)

spaxel_params = Parameters()
spaxel_params.add("Amp", value=150., min=0.001)
spaxel_params.add("wave", value=expect_em_line_pos, min=expect_em_line_pos-30, max=expect_em_line_pos+30) #Starting position calculated from redshift value of galaxy.
spaxel_params.add("FWHM", value=2.8, vary=False)#galaxy_info["LSF"], vary=False) # Line Spread Function
spaxel_params.add("Gauss_bkg", value=0.01)
spaxel_params.add("Gauss_grad", value=0.0001)

# Loop through spectra from list format of data.
if fit_spaxel == True:
    for y,x in tqdm(zip(non_zero_index[0], non_zero_index[1]), total=len(non_zero_index[0])):
        get_data_residuals = []
        fit_results = minimize(spaxel_by_spaxel, spaxel_params, args=(wavelength, res_cube[:,y,x], np.repeat(np.nanstd(res_cube[:,y,x],0), len(wavelength)), z), nan_policy="propagate")
        pos = y*x_data+x
        gauss_A[pos] = fit_results.params["Amp"].value
        obj_residuals[pos] = fit_results.residual
        data_residuals[pos] = fit_results.residual * np.repeat(np.nanstd(res_cube[:,y,x],0), len(wavelength))
        g_bkg[pos]  = fit_results.params["Gauss_bkg"].value
        g_grad[pos] = fit_results.params["Gauss_grad"].value
        
    list_of_rN = np.array([np.nanstd(d_r) for d_r in data_residuals])
    
    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

    # Save A/rN, Gauss A, Guass F and rN arrays as npy files. Change to .fits soon maybe
    np.save(DIR_dict["EXPORT_DIR"]+"_gauss_A", gauss_A)
    np.save(DIR_dict["EXPORT_DIR"]+"_gauss_F", gauss_F)
    np.save(DIR_dict["EXPORT_DIR"]+"_A_rN", A_rN)

    print("Cube fitted, data saved.")

else:
    print(f"Did not fit spaxel by spaxel for {galaxy_name} {loc}.")
    # load up gauss_A, gauss_F and A_rN
    gauss_A = np.load(DIR_dict["EXPORT_DIR"]+"_gauss_A.npy")
    gauss_F = np.load(DIR_dict["EXPORT_DIR"]+"_gauss_F.npy")
    A_rN    = np.load(DIR_dict["EXPORT_DIR"]+"_A_rN.npy")


# Construct A/rN, A_5007 and F_5007 plots, and save in Plots/Galaxy_name/
# Plot A/rN
plt.figure(figsize=(20,20))
plt.imshow(A_rN.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=1, vmax=8)
plt.title("A/rN")
plt.colorbar()
plt.savefig(DIR_dict["PLOT_DIR"]+"_A_rN_map.png")

# Plot A_5007
plt.figure(figsize=(20,20))
plt.imshow(gauss_A.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.title("Amplitude")
plt.colorbar()
plt.savefig(DIR_dict["PLOT_DIR"]+"_A_5007_map.png")

# Plot F_5007
plt.figure(figsize=(20,20))
plt.imshow(gauss_F.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
plt.title("Flux")
plt.colorbar()
plt.savefig(DIR_dict["PLOT_DIR"]+"_F_5007_map.png")

print(f"Plots saved in Plots/{galaxy_name}")



############################################################################
################################# run SEP  #################################
############################################################################

A_rN_img = A_rN.reshape(y_data, x_data)

# Where element is equal to element [0,0], set equal to 0.0, essentially making out of bound areas equal to 0.0
A_rN_img[A_rN_img == A_rN_img[0,0]] = 0.0

plt.figure(figsize=(20,20))

# analyse background noise using sep.background
bkg = sep.Background(A_rN_img)

bkg_image = bkg.rms()

Y, X = np.mgrid[:y_data, :x_data]

# set up the mask parameters, as taken from the yaml file. default is [0,0,0,0,0]
if loc == "middle" or loc == "halo":
    xe, ye, length, width, alpha = [0,0,0,0,0]
else:
    xe, ye, length, width, alpha = galaxy_info["gal_mask"]

elip_mask_gal = generate_mask(img_shape=[y_data, x_data], mask_params=galaxy_info["gal_mask"], mask_shape="ellipse")


# mask out any known and selected stars
star_mask = np.sum([generate_mask(img_shape=[y_data, x_data], mask_params=star_mask, mask_shape="circle") for star_mask in galaxy_info["star_mask"]], 0).astype(bool)

###################
##      SEP      ##
###################
# Use sep.extract to get the locations of sources
objects = sep.extract(A_rN_img, thresh=2.5, clean=True, mask=elip_mask_gal+star_mask, deblend_cont=0.001,)
peak_filter = np.where(objects["peak"] < 30)

x_sep = objects["x"][peak_filter]
y_sep = objects["y"][peak_filter]

positions = [(x,y) for x,y in zip(x_sep, y_sep)]
apertures = CircularAperture(positions, r=4)
plt.figure(figsize=(16,16))
plt.imshow(A_rN_img, origin="lower", cmap="CMRmap", vmin=1, vmax=8.)
apertures.plot(color="green")

# Add on the eliptical and star masks
ax = plt.gca()
elip_gal = Ellipse((xe, ye), width, length, angle=alpha*(180/np.pi), fill=False, color="white")
ax.add_artist(elip_gal)
for star in galaxy_info["star_mask"]:
    ax.add_artist(Circle((star[0], star[1]), radius=star[2], fill=False, color="grey", ls="--"))


x_y_list = np.array([[x,y] for x,y in zip(x_sep, y_sep)])
x_PNe = np.array([x[0] for x in x_y_list])
y_PNe = np.array([y[1] for y in x_y_list])


for i, item in enumerate(x_y_list):
    ax.annotate(i, (item[0]+6, item[1]-2), color="white", size=10)

        
plt.savefig(DIR_dict["PLOT_DIR"]+"_circled_sources.png", bbox_inches='tight')

sep_peak = objects["peak"][peak_filter]
# store list of objects, and print number of detected objects



print(f"Number of detected [OIII] sources: {len(x_y_list)}")

if save_sep == True:
    np.save(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list", x_y_list)
    np.save(DIR_dict["EXPORT_DIR"]+"_PNE_SEP_peak", sep_peak)
    
######
# save a fits file of the PNe, with objective and error lists
# including header for wcs info

# Retrieve the respective spectra for each PNe source, from the list of spectra data file, using a function to find the associated index locations of the spectra for a PNe.
PNe_spectra = np.array([PNe_minicube_extractor(x, y, n_pixels, res_cube, wavelength) for x,y in zip(x_PNe, y_PNe)])

obj_error_cube = uncertainty_cube_construct(obj_residuals, x_PNe, n_pixels, PNe_spectra, wavelength)

res_error_cube = uncertainty_cube_construct(data_residuals, x_PNe, n_pixels, PNe_spectra, wavelength)

primary_hdu = fits.PrimaryHDU()

res_hdr.set("YAXIS", value=x_data)
res_hdr.set("XAXIS", value=y_data)

# Use once residual cubes are introduced
PNe_hdu = fits.ImageHDU(data=PNe_spectra, header=res_hdr, name="PNe_spectra")

wave_hdu = fits.ImageHDU(data=wavelength, name="wavelength", )
objective_hdu = fits.ImageHDU(data=obj_error_cube, name="obj_err",)
res_error_hdu = fits.ImageHDU(data=res_error_cube, name="res_err",)


# Save fits file
print("PNe minicubes saved to "+DIR_dict["DATA_DIR"])
PNe_HDUList = fits.HDUList([primary_hdu, PNe_hdu, wave_hdu, objective_hdu, res_error_hdu])
PNe_HDUList.writeto(DIR_dict["DATA_DIR"]+"_PNe_spectra.fits", overwrite=True)
