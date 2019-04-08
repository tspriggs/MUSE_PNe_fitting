import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.wcs import WCS, utils, wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from matplotlib.patches import Rectangle, Ellipse, Circle
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
from MUSE_Models import PNe_residuals_3D, PNe_spectrum_extractor, PSF_residuals_3D, data_cube_y_x, robust_sigma

# Load in yaml file to query galaxy properties
with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)

# Queries user for a galaxy name, in the form of FCC000, and taking the relevant info from the yaml file
choose_galaxy = input("Please type which Galaxy you want to analyse, use FCC000 format: ")
galaxy_data = galaxy_info[choose_galaxy]

# Load in the residual data, in list form
hdulist = fits.open(galaxy_data["residual cube"]) # Path to data
res_hdr = hdulist[0].header # extract header from residual cube

# Check to see if the wavelength is in the fits fileby checking length of fits file.
if len(hdulist) == 2: # check to see if HDU data has 2 units (data, wavelength)
    wavelength = np.exp(hdulist[1].data)
    np.save(galaxy_data["wavelength"], wavelength)
else:
    wavelength = np.load(galaxy_data["wavelength"])


# Use the length of the data to return the size of the y and x dimensions of the spatial extent.
if choose_galaxy == "FCC219":
    x_data, y_data, n_data = data_cube_y_x(len(hdulist[0].data))
else:
    y_data, x_data, n_data = data_cube_y_x(len(hdulist[0].data))


# Indexes where there is spectral data to fit. We check where there is data that doesn't start with 0.0 (spectral data should never be 0.0).
non_zero_index = np.squeeze(np.where(hdulist[0].data[:,0] != 0.))

# Constants
n_pixels = 9 # number of pixels to be considered for FOV x and y range
c = 299792458.0 # speed of light

z = galaxy_data["z"] # Redshift - taken from simbad / NED - read in from yaml file
D = galaxy_data["Distance"] # Distance in Mpc - from Simbad / NED - read in from yaml file

# Construct the PNe FOV coordinate grid for use when fitting PNe.
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

# Progress bar

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

# Defines spaxel by spaxel fitting model
def spaxel_by_spaxel(params, x, data, error, spec_num):
    """
    Using a Gaussian double peaked model, fit the [OIII] lines at 4959 and 5007 Angstrom, found within Stellar continuum subtracted spectra, from MUSE.
    Inputs:
        Params - Using the LMfit python package, contruct the parameters needed and read them in:
                Amplitude of [OIII] at 5007 A.
                mean wavelength position of [OIII] 5007 A peak.
                FWHM of Gaussian profiles.
                Gaussian backrgound level of residuals.
                Gaussian gradient of background residuals.
        x - Wavelength array
        data - read in sprectrum by spectrum of data via list form.
        error - associated errors for each spectrum.
        spec_num - from enumerate, just the index number of spectrum, for storing value sin np array.

    Returns -  (Data - model) / error   for chi square minimiser.
    """
    Amp = params["Amp"]
    wave = params["wave"]
    FWHM = params["FWHM"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]

    Gauss_std = FWHM / 2.35482 # FWHM to Standard Deviation calculation.

    model = ((Gauss_bkg + Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - wave)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (x - (wave - 47.9399*(1+z)))** 2 / Gauss_std**2.))

    # Saves both the Residual noise level of the fit, alongside the 'data residual' (data-model) array from the fit.
    list_of_rN[spec_num] = robust_sigma(data - model)
    data_residuals[spec_num] = data - model

    return (data - model) / error

################################################################################
#################################### Fit 1D ####################################
################################################################################

# Run Spaxel by Spaxel fit of the spectra within the .fits file.
# Check if fit_1D parameter, within the Galaxy_info.yaml file is set to Y (yes to fit), or N (no to fit - has been fitted before).
fit_1D = input("Spaxel by Spaxel fit for the [OIII] doublet? (y/n) ")
if fit_1D == "y":
    # Run Spaxel by Spaxel fitter
    print("Fitting Spaxel by Spaxel for [OIII] doublet.")

    list_of_std = np.abs([robust_sigma(dat) for dat in hdulist[0].data])
    input_errors = [np.repeat(item, len(wavelength)) for item in list_of_std] # Intially use the standard deviation of each spectra as the uncertainty for the spaxel fitter.

    # Setup numpy arrays for storage of best fit values.
    gauss_A = np.zeros(len(hdulist[0].data))
    list_of_rN = np.zeros(len(hdulist[0].data))
    data_residuals = np.zeros((len(hdulist[0].data),len(wavelength)))
    obj_residuals = np.zeros((len(hdulist[0].data),len(wavelength)))

    # setup LMfit paramterts
    spaxel_params = Parameters()
    spaxel_params.add("Amp",value=150., min=0.001)
    spaxel_params.add("wave", value=5007.0*(1+z)-3, min=5007.0*(1+z)-40, max=5007.0*(1+z)+40) #Starting position calculated from redshift value of galaxy.
    spaxel_params.add("FWHM", value=galaxy_data["LSF"], vary=False) # Line Spread Function
    spaxel_params.add("Gauss_bkg", value=0.001)
    spaxel_params.add("Gauss_grad", value=0.0001)

    # Loop through spectra from list format of data.
    for j,i in enumerate(non_zero_index):
        progbar(j, len(non_zero_index), 40)
        fit_results = minimize(spaxel_by_spaxel, spaxel_params, args=(wavelength, hdulist[0].data[i], input_errors[i], i), nan_policy="propagate")
        gauss_A[i] = fit_results.params["Amp"].value
        obj_residuals[i] = fit_results.residual

    A_rN = np.array([A / rN for A,rN in zip(gauss_A, list_of_rN)])
    Gauss_F = np.array(gauss_A) * np.sqrt(2*np.pi) * 1.19

    # Save A/rN, Gauss A, Guass F and rN arrays as npy files.
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/A_rN_cen", A_rN)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_A_cen", gauss_A)
    #np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_A_err_cen", A_err)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/gauss_F_cen", Gauss_F)
    np.save("exported_data/"+ galaxy_data["Galaxy name"] +"/rN", list_of_rN)

    # save the data and obj res in fits file format to us memmapping.
    hdu_data_res = fits.PrimaryHDU(data_residuals)
    hdu_obj_res = fits.PrimaryHDU(obj_residuals)
    hdu_data_res.writeto("exported_data/"+ galaxy_data["Galaxy name"] +"/resids_data.fits", overwrite=True)
    hdu_obj_res.writeto("exported_data/"+ galaxy_data["Galaxy name"] +"/resids_obj.fits", overwrite=True)

    print("Cube fitted, data saved.")

    # Construct A/rN, A_5007 and F_5007 plots, and save in Plots/Galaxy_name/
    # Plot A/rN
    plt.figure(figsize=(20,20))
    plt.imshow(A_rN.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=1, vmax=8)
    plt.colorbar()
    plt.savefig("Plots/"+ galaxy_data["Galaxy name"]+"/A_rN_map.png")

    # Plot A_5007
    plt.figure(figsize=(20,20))
    plt.imshow(gauss_A.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
    plt.colorbar()
    plt.savefig("Plots/"+ galaxy_data["Galaxy name"]+"/A_5007_map.png")

    # Plot F_5007
    plt.figure(figsize=(20,20))
    plt.imshow(Gauss_F.reshape(y_data, x_data), origin="lower", cmap="CMRmap", vmin=10, vmax=100)
    plt.colorbar()
    plt.savefig("Plots/"+ galaxy_data["Galaxy name"]+"/F_5007_map.png")

    print("Plots saved in Plots/"+galaxy_data["Galaxy name"])


# If spaxel-by-spaxel fit has already been done, fit_1D is n, proceed to 3D fit.
elif fit_1D == "n":
    print("3D fitter it is.......")

################################################################################
#################################### Fit 3D ####################################
################################################################################

# Check is user wants to run the rest of the script, i.e. 3D model and PSF analysis
fit_3D = input("Do you want to fit the detected sources in 3D + PSF analysis? (y/n)")
if fit_3D == "y":

    print("Starting PNe analysis.....")
    # load from saved files

    # Read in list of x and y coordinates of detected sources for 3D fitting.
    x_y_list = np.load("exported_data/"+ galaxy_data["Galaxy name"] +"/PNe_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list]) # separate out from the list the list of x coordinates, as well as y coordinates.
    y_PNe = np.array([y[1] for y in x_y_list])

    # Retrieve the respective spectra for each PNe source, from the list of spectra data file, using a function to find the associated index locations of the spectra for a PNe.
    PNe_spectra = np.array([PNe_spectrum_extractor(x, y, n_pixels, hdulist[0].data, x_data, wave=wavelength) for x,y in zip(x_PNe, y_PNe)])

    # create Pandas dataframe for storage of values from the 3D fitter.
    PNe_df = pd.DataFrame(columns=("PNe number", "Ra (J2000)", "Dec (J2000)", "V (km/s)", "m 5007", "M 5007", "[OIII] Flux", "[OIII] Flux error","A/rN", "redchi", "Filter"))
    PNe_df["PNe number"] = np.arange(0,len(x_PNe))
    PNe_df["Filter"] = "Y"


    if choose_galaxy == "FCC167" or choose_galaxy == "FCC219":
        hdu_wcs = fits.open(choose_galaxy+"_data/"+choose_galaxy+"center.fits")
        hdr_wcs = hdu_wcs[1].header
        wcs_obj = WCS(hdr_wcs, naxis=2)

        for i in np.arange(0, len(x_PNe)):
            Ra_Dec = utils.pixel_to_skycoord(x_PNe[i],y_PNe[i], wcs_obj).to_string("hmsdms", precision=2).split()
            PNe_df.loc[i,"Ra (J2000)"] = Ra_Dec[0]
            PNe_df.loc[i,"Dec (J2000)"] = Ra_Dec[1]

    # Read in Objective Residual Cube .fits file.
    obj_residual_cube = fits.open("exported_data/"+ galaxy_data["Galaxy name"] +"/resids_obj.fits")

    # Read in Data Residual Cube .fits file.
    data_residual_cube = fits.open("exported_data/"+ galaxy_data["Galaxy name"] +"/resids_data.fits")

    # Function to extract the uncertainties and transform them into a standard deviation version for fitting purposes.
    def uncertainty_cube_construct(data, x_P, y_P, n_pix):
        data[data == np.inf] = 0.01
        extract_data = np.array([PNe_spectrum_extractor(x, y, n_pix, data, x_data, wave=wavelength) for x,y in zip(x_P, y_P)])
        array_to_fill = np.zeros((len(x_P), n_pix*n_pix, len(wavelength)))
        for p in np.arange(0, len(x_P)):
            list_of_std = np.abs(robust_sigma(dat) for dat in extract_data[p])
            array_to_fill[p] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0, len(list_of_std))]

        return array_to_fill

    # Run above function to get the error and obj_error cubes for fitting purposes (uncertainty).
    error_cube = uncertainty_cube_construct(data_residual_cube[0].data, x_PNe, y_PNe, n_pixels)
    obj_error_cube = uncertainty_cube_construct(obj_residual_cube[0].data, x_PNe, y_PNe, n_pixels)

    print("Files loaded.")


    # This is the start of the setup for the 3D fitter.
    # Initialise the paramters for 3D fitting.
    PNe_3D_params = Parameters()

    # extract dictionary of emissions from Galaxy_info.yaml file.
    emission_dict = galaxy_data["emissions"]

    # Function to generate the parameters for the 3D model and fitter. Built to be able to handle a primary emission ([OIII] here).
    # Buil to fit for other emissions lines, as many as are resent in the emission dictionary.
    def gen_params(wave=5007*(1+z), FWHM=4.0, beta=2.5, LSF=2.81, em_dict=None):
        # loop through emission dictionary to add different element parameters
        for em in em_dict:
            # Amplitude parameter for each emission
            PNe_3D_params.add('Amp_2D_{}'.format(em), value=emission_dict[em][0], min=0.001, max=1e5, expr=emission_dict[em][1])
            # Wavelength parameter for each emission
            if emission_dict[em][2] == None:
                PNe_3D_params.add("wave_{}".format(em), value=wave, min=wave-40., max=wave+40.)
            else:
                PNe_3D_params.add("wave_{}".format(em), expr=emission_dict[em][2].format(z))

        # Add the rest of the paramters for the 3D fitter here, including the PSF (Moffat FWHM (M_FWHM) and beta)
        PNe_3D_params.add('x_0', value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
        PNe_3D_params.add('y_0', value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
        PNe_3D_params.add("M_FWHM", value=FWHM, vary=False)
        PNe_3D_params.add("beta", value=beta, vary=False)
        PNe_3D_params.add("LSF", value=LSF, vary=False)
        PNe_3D_params.add("Gauss_bkg",  value=0.0001)
        PNe_3D_params.add("Gauss_grad", value=0.0001)

    # Setup Numpy arrays for storing values from the fitter
    total_flux = np.zeros((len(x_PNe), len(emission_dict)))                            # Total integrated flux of each emission, as measured for the PNe.
    A_2D_list = np.zeros((len(x_PNe), len(emission_dict)))                             # Amplitude from the Moffat function
    F_xy_list = np.zeros((len(x_PNe), len(emission_dict), len(PNe_spectra[0])))        # Array of flux arrays for each emission.
    emission_amp_list = np.zeros((len(x_PNe), len(emission_dict)))                     # List of each emission's amplitude
    model_spectra_list = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength)))    # Array of the model spectral fits
    mean_wave_list = np.zeros((len(x_PNe), len(emission_dict)))                        # List of the measured wavelength positions for each emission
    residuals_list = np.zeros(len(x_PNe))                                              # List of the residual noise level of each fit.
    list_of_fit_residuals = np.zeros((len(x_PNe), n_pixels*n_pixels, len(wavelength))) # List of arrays of best fit residuals (data-model)

    # Setup Numpy arrays for storing the errors from the fitter.
    moff_A_err = np.zeros((len(x_PNe), len(emission_dict)))
    x_0_err = np.zeros((len(x_PNe), len(emission_dict)))
    y_0_err = np.zeros((len(x_PNe), len(emission_dict)))
    mean_wave_err = np.zeros((len(x_PNe), len(emission_dict)))
    Gauss_bkg_err = np.zeros((len(x_PNe), len(emission_dict)))
    Gauss_grad_err = np.zeros((len(x_PNe), len(emission_dict)))

    # Setup Numpy arrays for storing the other best fit values from the 3D fitter.
    FWHM_list = np.zeros(len(x_PNe))
    list_of_x = np.zeros(len(x_PNe))
    list_of_y = np.zeros(len(x_PNe))
    Gauss_bkg = np.zeros(len(x_PNe))
    Gauss_grad = np.zeros(len(x_PNe))


    # Define a function that contains all the steps needed for fitting, including the storage of important values, calculations and pandas assignment.
    def run_minimiser(parameters):
        for PNe_num in np.arange(0, len(x_PNe)):
            progbar(int(PNe_num)+1, len(x_PNe), 40)
            useful_stuff = [] # Used to store other outputs from the 3D model function: maximum spectral amplitude, flux arrays for each emission, A_xy, model spectra
            #run minimizer fitting routine
            PNe_minimizer     = lmfit.Minimizer(PNe_residuals_3D,PNe_3D_params, fcn_args=(wavelength, x_fit, y_fit, PNe_spectra[PNe_num], error_cube[PNe_num], PNe_num, emission_dict, useful_stuff), nan_policy="propagate")
            multi_fit_results = PNe_minimizer.minimize()
            total_flux[PNe_num] = np.sum(useful_stuff[1][1],1) * 1e-20
            list_of_fit_residuals[PNe_num] = useful_stuff[0]
            A_2D_list[PNe_num] = useful_stuff[1][0]
            F_xy_list[PNe_num] = useful_stuff[1][1]
            model_spectra_list[PNe_num] = useful_stuff[1][3]
            emission_amp_list[PNe_num] = [PNe_3D_results.params["Amp_2D_{}".format(em)] for em in emission_dict]
            mean_wave_list[PNe_num] = [PNe_3D_results.params["wave_{}".format(em)] for em in emission_dict]
            list_of_x[PNe_num] = PNe_3D_results.params["x_0"]
            list_of_y[PNe_num] = PNe_3D_results.params["y_0"]
            Gauss_bkg[PNe_num] = PNe_3D_results.params["Gauss_bkg"]
            Gauss_grad[PNe_num] = PNe_3D_results.params["Gauss_grad"]
            #save errors
            moff_A_err[PNe_num] = [PNe_3D_results.params["Amp_2D_{}".format(em)] for em in emission_dict]
            mean_wave_err[PNe_num] = [PNe_3D_results.params["wave_{}".format(em)] for em in emission_dict]
            x_0_err[PNe_num] = PNe_3D_results.params["x_0"].stderr
            y_0_err[PNe_num] = PNe_3D_results.params["y_0"].stderr
            Gauss_bkg_err[PNe_num] = PNe_3D_results.params["Gauss_bkg"].stderr
            Gauss_grad_err[PNe_num] = PNe_3D_results.params["Gauss_grad"].stderr

        # Amplitude / residul Noise calculation
        list_of_rN = np.array([robust_sigma(PNe_res) for PNe_res in list_of_fit_residuals])
        PNe_df["A/rN"] = A_2D_list[:,0] / list_of_rN # Using OIII amplitude

        # chi square analysis
        gauss_list, redchi, Chi_sqr = [], [], []
        for p in range(len(x_PNe)):
            PNe_n = np.copy(PNe_spectra[p])
            flux_1D = np.copy(F_xy_list[p][0])
            A_n = ((flux_1D) / (np.sqrt(2*np.pi) * 1.19))

            def gaussian(x, amplitude, mean, stddev, bkg, grad):
                return ((bkg + grad*x) + np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 /    (stddev**2.)) +
                (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399))** 2 /  (stddev**2.)))

                list_of_gauss = [gaussian(wavelength, A, mean_wave_list[p][0], 1.19,Gauss_bkg[p],  Gauss_grad[p]) for A in A_n]
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
            redchi.append(Chi2/(len(wavelength)*n_pixels**2 - PNe_minimizer.nfree))
            gauss_list.append(list_of_gauss)
            Chi_sqr.append(Chi2)

        PNe_df['Chi2']   = Chi_sqr
        PNe_df["redchi"] = redchi

        # de-redshift the fitted wavelengths to get the velocity
        de_z_means = mean_wave_list[:,0] / (1 + z)

        PNe_df["V (km/s)"] = (c * (de_z_means - 5006.77) / 5006.77) / 1000.

        PNe_df["[OIII] Flux"] = total_flux[:,0]                       #store total [OIII] 5007A emission line flux

        if "hb" in emission_dict:
            PNe_df["[OIII]/Hb"] = PNe_df["[OIII] Flux"] / total_flux[:,2] # store [OIII]/Hb ratio

        if "ha" in emission_dict:
            PNe_df["Ha Flux"] = total_Flux[:, 1]                          # store total Ha flux.


        # Calculate the apparent and Absolute Magnitudes for each PNe
        PNe_df["m 5007"] = -2.5 * np.log10(PNe_df["[OIII] Flux"].values) - 13.74       # Apparent Magnitude

        # Construct a Astropy table to save certain values for each galaxy.
        PNe_table = Table([np.arange(0,len(x_PNe)), PNe_df["Ra (J2000)"], PNe_df["Dec (J2000)"],
                           PNe_df["[OIII] Flux"].round(20),
                           PNe_df["m 5007"].round(2),
                           PNe_df["A/rN"].round(1),
                           PNe_df["[OIII]/Hb"].round(2),],
                           names=("PNe number", "Ra", "Dec", "[OIII] Flux", "m 5007", "A/rN", "[OIII]/Hb"))
        ascii.write(PNe_table, "exported_data/"+"{0}/{0}_table.txt".format(galaxy_data["Galaxy name"]), format="tab", overwrite=True) # Save table in tab separated format.
        ascii.write(PNe_table, "exported_data/"+"{0}/{0}_table_latex.txt".format(galaxy_data["Galaxy name"]), format="latex", overwrite=True) # Save latex table of data.
        print("exported_data/"+galaxy_data["Galaxy name"]+"/"+galaxy_data["Galaxy name"]+"_table.txt saved")
        print("exported_data/"+galaxy_data["Galaxy name"]+"/"+galaxy_data["Galaxy name"]+"_table_latex.txt saved")

    print("Running 3D fitter")
    gen_params(wave=5007*(1+z)-3, FWHM=galaxy_data["FWHM"], beta=galaxy_data["beta"], LSF=galaxy_data["LSF"], em_dict=emission_dict)
    run_minimiser(PNe_3D_params) # Run the 3D model fitter.

    ## The Great Filter - goes here
    PNe_df["Filter"] = "Y"
    PNe_df.loc[PNe_df["A/rN"]<3., "Filter"] = "N"
    # reduced Chi sqr cut
    redchi_med = np.median(PNe_df["redchi"].values)
    redchi_std = robust_sigma(PNe_df["redchi"].values)
    lower_redchi = redchi_med - 3*redchi_std
    upper_redchi = redchi_med + 3*redchi_std
    PNe_df.loc[(PNe_df["redchi"]>=upper_redchi) & (PNe_df["redchi"]<=lower_redchi), "Filter"] = "N"


    ## FCC167
    if gal_name == "FCC167":
        PNe_df.loc[PNe_df["PNe number"]==30, "Filter"] = "N" # Over luminous [OIII] source
        PNe_df.loc[PNe_df["PNe number"]==15, "Filter"] = "N" # SNR maybe, extended source with dual peaked [OIII] 5007

    ## Error estimation goes here....

    ## Distance estimation goes here...

    ## M_5007 calculation goes here....

    # Here we run the PSF analysis if needed.....
    # First ask which to attempt: brightest or pre-selected PNe
    # if brightest, then ask how many PNe to use from a list of the brightest in m_5007
    # if pre-selected, then ask for which PNe numbers to use.
    fit_for_PSF = input("Do you wish to fit for PSF? [y/n] ")
    if fit_for_PSF == "y":
        sel_PNe = PNe_df.loc[PNe_df["Filter"]=="Y"].nlargest(10, "A/rN").index.value

        print("Using the 10 brightest PNe in A/rN. Selected PNe are: " + sel_PNe)

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
                model_params(p=PSF_params, n=i, amp=200.0, wave=5007*(1+z))

        PSF_params.add('FWHM', value=4.0, min=0.01, max=12., vary=True)
        PSF_params.add("beta", value=4.0, min=0.01, max=12., vary=True)
        PSF_params.add("LSF",  value=3.0, min=0.01, max=12., vary=True)

        print("Fitting for PSF")
        PSF_results = minimize(PSF_residuals_3D, PSF_params, args=(wavelength, x_fit, y_fit, selected_PNe, selected_PNe_err, z), nan_policy="propagate")

        #determine PSF values and feed back into 3D fitter

        fitted_FWHM = PSF_results.params["FWHM"].value
        fitted_FWHM_err = PSF_results.params["FWHM"].stderr
        fitted_beta = PSF_results.params["beta"].value
        fitted_beta_err = PSF_results.params["beta"].stderr
        fitted_LSF  = PSF_results.params["LSF"].value
        fitted_LSF_err  = PSF_results.params["LSF"].stderr

        print("Fitted FWHM and error: {0} +/- {1}".format(fitted_FWHM, fitted_FWHM_err)) # print FWHM
        print("Fitted beta and error: {0} +/- {1}".format(fitted_beta, fitted_beta_err)) # print beta
        print("Fitted LSF and error:  {0} +/- {1}".format(fitted_LSF, fitted_LSF_err)) # print LSF

        #Fit PNe with updated PSF
        gen_params(FWHM=fitted_FWHM, beta=fitted_beta, em_dict=emission_dict) # set params up with fitted FWHM and beta values
        print("Fitting PNe again with fitted PSF values.")
        run_minimiser(PNe_3D_params) # run fitting section again with new values

    else:
        print("Plotting each PNe spectra with best-fit model.")

    # Plot out each full spectrum with fitted peaks
    for p in np.arange(0, len(x_PNe)):
        plt.figure(figsize=(20,10))
        plt.plot(wavelength, np.sum(PNe_spectra[p],0), alpha=0.7, c="k") # data
        plt.plot(wavelength, np.sum(model_spectra_list[p],0), c="r") # model
        plt.axhline(residuals_list[p], c="b", alpha=0.6)
        plt.xlabel("Wavelength ($\AA$)", fontsize=18)
        plt.ylabel("Flux Density ($10^{-20}$ $erg s^{-1}$ $cm^{-2}$ $\AA^{-1}$ $arcsec^{-2}$)", fontsize=18)
        plt.ylim(-1500,3000)
        plt.savefig("Plots/"+ galaxy_data["Galaxy name"] +"/full_spec_fits/PNe_{}.pdf".format(p))

        plt.clf()

    #Run the rest of the analysis

    print("PNe analysis complete.")

    # Plot galaxy A/rN map with circled sources, indicating filter status of object, plus masked out area.



    # PNLF



    # Alpha value calculation








elif fit_3D == "n":
    print("This is the end of PNe analysis script. Goodbye")
