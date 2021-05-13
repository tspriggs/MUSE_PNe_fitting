import numpy as np
import lmfit as lmfit
from lmfit import Parameters
from functions.MUSE_Models import Moffat, Gauss

# Set up PSF params
def generate_PSF_params(sel_PNe, amp, mean, n_pixels=9):
    """ Generate the parameter object, from LMfit, for the PSF fitting routine. This produces an LMfit parameter that has entires for each PNe from sel_PNe.

    Parameters
    ----------
    sel_PNe : list
        Index ID for the PNe selected for PSF fitting.
    amp : float
        Initial guess for the PNe's 2D amplitudes.
    mean : float
        Initial guess for the location of the [OIII] in wavelength space, adjusted by redshift.
    n_pixels : int, optional
        pixel width of PNe minicubes, by default 9

    Returns
    -------
    dict
        LMfit parameter object of PSF fitting parameters.
    """
    
    param_obj = Parameters()
    for n in np.arange(0,len(sel_PNe)):
        param_obj.add("moffat_amp_{:03d}".format(n), value=amp, min=0.01)
        param_obj.add("x_{:03d}".format(n), value=(n_pixels/2.), min=(n_pixels/2.) -4, max=(n_pixels/2.) +4)
        param_obj.add("y_{:03d}".format(n), value=(n_pixels/2.), min=(n_pixels/2.) -4, max=(n_pixels/2.) +4)
        param_obj.add("wave_{:03d}".format(n), value=mean, min=mean-20., max=mean+20.)
        param_obj.add("gauss_bkg_{:03d}".format(n),  value=0.001, vary=True)
        param_obj.add("gauss_grad_{:03d}".format(n), value=0.001, vary=True)

    return param_obj

def run_PSF_analysis(sel_PNe, PNe_spectra, obj_err, wavelength, x_fit, y_fit, z, n_pixels=9., run_CI=False):
    """Fits multiple PNe simultaneously to evaluate the Point Spread Function (PSF) and Line Spread Function,
    of the galaxy (pointing dependant).

    Parameters
    ----------
    sel_PNe : list
        list of selected PNe, by index, for simultaneous PSF fitting.
    PNe_spectra : list / array
        residual data containing the emission lines of PNe. A list of minicubes, one for each PNe to be fitted for the PSF.
    obj_err : list / array
        objective function error, made during the spaxel-by-spaxel fitting stage.
    wavelength : array
        Wavelength array
    x_fit : list /array
        x array of matrix sized n_pixel x n_pixel.
    y_fit : list / array
        y array of matrix sized n_pixel x n_pixel.
    z : float
        Redshift value
    n_pixels : int, optional
        pixel width of PNe minicubes, by default 9.

    Returns
    -------
    PSF_results, 
        LMfit minimisation results, dictionary object.
    PSF_ci,
        PSF confidence intervals, from LMfit.
    """

    print("\n")
    print("################################################################")
    print("########################## Fitting PSF #########################")
    print("################################################################")
    print("\n")
    print(f"Using PNe: {sel_PNe}")

    selected_PNe = PNe_spectra[sel_PNe]
    selected_PNe_err = obj_err[sel_PNe]

    # Create parameters for each PNe: one set of entries per PN
    PSF_params = generate_PSF_params(sel_PNe, amp=750.0, mean=5006.77*(1+z))

    # Add in the PSF and LSF paramters
    PSF_params.add('FWHM', value=4.0, min=0.01, vary=True)
    PSF_params.add("beta", value=2.5, min=1.00, vary=True)
    PSF_params.add("LSF",  value=3.0, min=0.01, vary=True)

    # minimisation of the PSF functions
    PSF_min = lmfit.Minimizer(PSF_residuals_3D, PSF_params, fcn_args=(wavelength, x_fit, y_fit, selected_PNe, selected_PNe_err, z), nan_policy="propagate")
    PSF_results = PSF_min.minimize()
    # use LMfits confidence interval functionality to map out the errors between the 3 different parameters.
    print("Calculating Confidence Intervals for FWHM, beta and LSF")
    if run_CI == True:
        PSF_ci = lmfit.conf_interval(PSF_min, PSF_results, p_names=["FWHM", "beta", "LSF"], sigmas=[1,2])
    else:
        PSF_ci = []
    
    print("FWHM: ", round(PSF_results.params["FWHM"].value, 4), "+/-", round(PSF_results.params["FWHM"].stderr, 4), "(", round(PSF_results.params["FWHM"].stderr / PSF_results.params["FWHM"].value, 4)*100, "%)")
    print("Beta: ", round(PSF_results.params["beta"].value, 4), "+/-", round(PSF_results.params["beta"].stderr, 4), "(", round(PSF_results.params["beta"].stderr / PSF_results.params["beta"].value, 4)*100, "%)")
    print("LSF: " , round(PSF_results.params["LSF"].value , 4), "+/-", round(PSF_results.params["LSF"].stderr , 4), "(", round(PSF_results.params["LSF"].stderr / PSF_results.params["LSF"].value, 4)*100, "%)")

    return PSF_results, PSF_ci


def PSF_residuals_3D(params, lam, x_2D, y_2D, data, err, z):
    """Objective function for the 3D fitting of multiple PNe, simultaneously. Takes in PNe residual spectra (containing only the emission lines), 
    and fits them simultaneously, though holds the PSF parameters the same for all models.

    Parameters
    ----------
    params : dict
        LMfit parameter object for PSF fitting.
    lam : list / array
        Wavelength array for fitting the emission lines.
    x_2D : list / array
        x array of matrix sized n_pixel x n_pixel.
    y_2D : list / array
        y array of matrix sized n_pixel x n_pixel.
    data : list / array
        List of PNe residual minicubes, containing the emission lines.
    err : list / array
        List of the associated PN error spectra, for use in the objective function (data-model/error)
    z : float
        Redshift value of the galaxy.

    Returns
    -------
    list / array
        List of the residuals from the objective function: (data-model)/error. 
    """
    FWHM = params['FWHM']
    beta = params["beta"]
    LSF = params["LSF"]

    def gen_model(x, y, moffat_amp, FWHM, beta, g_LSF, bkg, grad, wave, z, x_2D, y_2D):
        """ Generate the models for comparison, used only within the PSF_residuals_3D function """
        # 2 dimensional flux distribution, derived from the 2D Moffat function
        F_OIII_xy = Moffat(moffat_amp, FWHM, beta, x, y, x_2D, y_2D)
        
        # calculate the Gaussian standard deviation (LSF) from the given FWHM value.
        Gauss_std = g_LSF / 2.35482

        # 2D array of ampltiudes, as converted from the Moffat flux distribution.
        A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))

        # using the 2D amplitude array, create a series of model spectra, forming a minicube of Guassians that will be fitted to the data.
        model_spectra = [Gauss(lam, Amp, wave, g_LSF, bkg, grad, z) for Amp in A_OIII_xy]

        return model_spectra

    # create a dictionary for storing the models for multiple PNe
    models = {}
    # for each PNe, store the model spectra under a string name "model_000" in the models dictionary.
    for k in np.arange(0, len(data)):
        models["model_{:03d}".format(k)] = gen_model(params["x_{:03d}".format(k)], params["y_{:03d}".format(k)],
                                                       params["moffat_amp_{:03d}".format(k)], FWHM, beta, LSF,
                                                       params["gauss_grad_{:03d}".format(k)], params["gauss_bkg_{:03d}".format(k)],
                                                       params["wave_{:03d}".format(k)], z, x_2D, y_2D)
    
    # repeating the above process, though here we will calcuate the residuals from (data-model)/err .
    resid = {}
    for m in np.arange(0, len(data)):
        resid["resid_{:03d}".format(m)] = ((data[m] - models["model_{:03d}".format(m)]) / err[m])
    
    # if the number of PNe used for simultaneous PSF fitting is greater than one, concatenate the results.
    if len(resid) > 1.:
        # resid is a dictionary, so a sort statement is needed to form a concatinated list of the resid results.
        return np.concatenate([resid[x] for x in sorted(resid)],0)
    # Otherwise, return the first, and only instance inside the resid dictionary.
    else:
        return resid["resid_000"]