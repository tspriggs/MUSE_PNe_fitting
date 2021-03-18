import lmfit
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm.notebook import trange, tqdm
from astropy.io import fits
from sklearn.utils import resample
from lmfit import Parameters
from functions.file_handling import paths

np.random.seed(42)

def ecdf(data):
    """Compute Emprical Cumulative Distribution Function (ECDF)

    Parameters
    ----------
    data : [list / array]
        list or array of recorded m_5007 values from PNe.

    Returns
    -------
    [list, list]
        x, eCDF: x aray is the sorted list of the input 'data', while the eCDF is calculated from the 'data'.
    """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return (x,y)

def calc_PNLF(m_star, mag, c_1=1.0, c_2=0.307, c_3=3.):
    """ Calculate the Plnaetary Nebulae Luminosity Function (PNLF), using the input paramters.

    Parameters
    ----------
    m_star : [float]
        theoretical cut-off value for the PNLF, normally using M_5007_star (-4.53) - dM.
    mag : [list / array]
        magnitude array over which to calculate the PNLF.
    c_1 : [float], optional
        Normalisation parameter, little to no observed impact in our method, by default 1.0.
    c_2 : [float], optional
        c_2 paramter controls the tail of the PNLF, and mainly influenced by fainter PN of a sample, by default 0.307.
    c_3 : [float], optional
        varied duirng experimentation, haven't changed back to normal since, by default 3.

    Returns
    -------
    [list / array]
        Calcualted PNLF for the given magnitude range.
    """    
    N_m = c_1 * np.exp(c_2*mag) * (1-np.exp(c_3*(m_star-mag)))
    return N_m

def reconstructed_image(galaxy_name, loc):
    DIR_dict = paths(galaxy_name, loc)
    
    with fits.open(DIR_dict["RAW_DATA"]) as hdu:
        hdr  = hdu[1].header
        s    = np.shape(hdu[1].data)
        wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']   
        cond = (wave >= 4900.0) & (wave <= 5100.0)
        data = np.sum(hdu[1].data[cond,:,:], axis=0)

    return data, wave, hdr

def form_PNLF_CDF(PNLF, data, dM, obs_comp, M_5007, m_5007):
    """Take in a formed PNLF, completeness correct it and calculate the Cumulative distribution function of the incompleteness corrected PNLF.

    Parameters
    ----------
    PNLF : [list / array]
        PNLF as calculated from a given dM and c2, over a series of m_5007 and M_5007 values
    data : [list / array]
        PNe magnitudes
    dM : [float]
        distance modulus value for the PNLF
    obs_comp : [list / array]
        observed completeness profile, supplied as a list of ratio's across a given m_5007.
    M_5007 : [list / array]
        Absolute magnitude, in [OIII], array (-4.53 to 0.53).
    m_5007 : [list / array]
        Apparent magnitude, in [OIII], array (26.0 to 31.0).

    Returns
    -------
    [list / array]
        Cumulative distribution function of the PNLF provided, at dM
    """    
    sorted_data = np.sort(data)

    PNLF_comp_corr = np.array(np.interp(m_5007, M_5007+dM, PNLF)*obs_comp)
    PNLF_comp_corr[PNLF_comp_corr < 0] = 0.0
    PNLF_CDF = np.array(np.interp(sorted_data, m_5007, np.nancumsum(PNLF_comp_corr)/np.nansum(PNLF_comp_corr)))

    return PNLF_CDF

def PNLF_fitter(params, data, obs_comp, M_5007, m_5007, comp_lim=False):
    """LMfit minimisation function. Creates a PNLF using paramters, forms the PNLF CDF and PNe empricial CDF.

    Parameters
    ----------
    params : [dictionary]
        LMfit parameter instance
    data : [list / array]
        PNe apparent magnitudes, in [OIII]
    obs_comp : [list / array]
        Observed completeness profile / ratio for the galaxy
    M_5007 : [list / array]
        Absolute magnitude, in [OIII], array (-4.53 to 0.53).
    m_5007 : [list / array]
        Apparent magnitude, in [OIII], array (26.0 to 31.0).

    Returns
    -------
    [list / array]
        LMfit - residual = abs( data - model )
    """    
    M_star = params["M_star"]
    dM = params["dM"]
    c1 = params["c1"]
    c2 = params["c2"]
    c3 = params["c3"]
    
    PNLF = calc_PNLF(m_star=M_star+dM, mag=M_5007+dM, c_1=c1, c_2=c2, c_3=c3) 

    if comp_lim == True:
        completeness_lim_mag = m_5007[obs_comp>=0.5].max()
        PNLF_CDF = form_PNLF_CDF(PNLF, data, dM, obs_comp, M_5007, m_5007)
        PNLF_CDF = PNLF_CDF[data<completeness_lim_mag]
        x, PNe_CDF = ecdf(data)
        PNe_CDF = PNe_CDF[data<completeness_lim_mag]

    elif comp_lim == False:
        PNLF_CDF = form_PNLF_CDF(PNLF, data, dM, obs_comp, M_5007, m_5007)
        x, PNe_CDF = ecdf(data)

    return np.abs(PNe_CDF-PNLF_CDF) 


def PNLF_analysis(galaxy_name, loc, data, obs_comp, M_5007, m_5007, dM_in=31.5, c2_in=0.307,
                 vary_dict={"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}, mcmc=False, comp_lim=False):  
    """This is the execution function to fit the PNLF to a given data set of m_5007 of PNe.
    First, setup the paramters class from LMfit, then make a lmfit.Minimizer function.
    Second, use the .minimize method to execute the fitting.
    There is an if statement to check if mcmc is to be used.
    Otherwise, returns an LMfit dict object that contains the best-fit results from the LMfit package.

    Parameters
    ----------
    galaxy_name : [str]
        Name of the galaxy.
    loc : [str]
        pointing location: center, halo or middle.
    data : [list / array]
        PNe apparent magnitudes, in [OIII].
    obs_comp : [list / array]
        Observed completeness profile / ratio for the galaxy.
    M_5007 : [list / array]
        Absolute magnitude, in [OIII], array (-4.53 to 0.53).
    m_5007 : [list / array]
        Apparent magnitude, in [OIII], array (26.0 to 31.0).
    dM_in : float, optional
        Starting guess for the value of distance modulus, by default 31.5.
    c2_in : float, optional
        Starting guess for the c2 paramter, by default 0.307.
    vary_dict : dict, optional
        A dictionary with a series of boolean switches, deciding which paramters will be fitted.
        By default {"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}.
    mcmc : bool, optional
        A check of whether an mcmc minimisation should also be run on the PNLF fitting function.
        By default False.

    Returns
    -------
    [dict]
        LMfit object containing the fit results from the PNLF and PNe CDF minimisation.
    """
    PNLF_params = Parameters()

    if loc == "center":
        PNLF_params.add("dM", value=dM_in, min=30.0, max=32.5, vary=vary_dict["dM"])
    elif loc in ["middle", "halo"]:
        gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
        center_dM = gal_df.loc[(galaxy_name, "center"), "PNLF dM"]
        PNLF_params.add("dM", value=center_dM, min=29.5, max=33.0, vary=False)

    PNLF_params.add("c1", value=1, min=0.00, vary=vary_dict["c1"])
    PNLF_params.add("c2", value=c2_in, min=-1.5, max=1.5, vary=vary_dict["c2"])
    PNLF_params.add("c3", value=3., min=0.0001, max=10, vary=vary_dict["c3"])
    PNLF_params.add("M_star", value=-4.53, min=-4.7, max=-4.3, vary=vary_dict["M_star"])

    PNLF_minimizer = lmfit.Minimizer(PNLF_fitter, PNLF_params, fcn_args=(data, obs_comp, M_5007, m_5007, comp_lim), nan_policy="propagate")
    PNLF_results = PNLF_minimizer.minimize()

    ## testing
    if mcmc == True:
        PNLF_results.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(20))
        res = lmfit.minimize(PNLF_fitter, args=(data, obs_comp, M_5007, m_5007), method='emcee', nan_policy='omit', nwalkers=200, burn=700, steps=3000, thin=20,
                     params=PNLF_results.params, is_weighted=False, progress=True)
        
        return res

    else:
        return PNLF_results


def MC_PNLF_runner(galaxy_name, loc, data, data_err_up, data_err_lo, obs_comp, c2_in=0.307, n_runs=1000,
                    vary_dict={"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}, comp_lim=False):
    """Using an observed sample of Planetary Nebulae m_5007 values, along with the associated errors (both upper and lower bounds), 
    we use MC sampling and bootstrapping to form a distribution of best-fit distance modulus values, from which we can determine the errors on our PNLF distances.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy.
    loc : str
        pointing location: center, halo or middle.
    data : list / array
        PNe apparent magnitudes, in [OIII].
    data_err_up : list / array
        PNe apparent mangitude errors, upper margin.
    data_err_lo : list / array
        PNe apparent mangitude errors, lower margin.
    obs_comp : list / array
        Observed completeness profile / ratio for the galaxy.
    c2_in : float, optional
        c2 parameter to be used, allows for switching between varying and fixing paramter, by default 0.307
    n_runs : int, optional
        Number of iterations the function is to run, by default 1000
    vary_dict : dict, optional
        dictionary of boolean switches for varying different parameters. by default {"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}

    Returns
    -------
    [list / array, list / array, LMfit object (dict)]
        array of dM values, array of c2 values (default to 0.307 for all values), PNLF results from the last iteration.
    """    
    data_err = np.median([data_err_up, data_err_lo],0) # average error

    step = 0.001
    M_star = -4.53
    M_5007 = np.arange(M_star, 0.53, step)
    m_5007 = np.arange(26, 31, step)
    
    MC_dM = []
    MC_c2 = []

    PN_m_distrs = np.array([np.random.normal(m_PN, m_PN_err, size=400) for m_PN, m_PN_err in zip(data, data_err)])

    for _ in range(n_runs):
        n_PNe_synth = len(data)

        rand_indx = resample(range(0, len(data), 1), replace=True, n_samples=n_PNe_synth, )

        synth_PNe = np.array([resample(PN_m_distrs[i], n_samples=1)[0] for i in rand_indx])

        PNLF_results = PNLF_analysis(galaxy_name, loc, synth_PNe, obs_comp, M_5007, m_5007, c2_in=c2_in, vary_dict=vary_dict, comp_lim=comp_lim)
        MC_dM.append(PNLF_results.params["dM"].value)
        MC_c2.append(PNLF_results.params["c2"].value)

    return np.array(MC_dM), np.array(MC_c2), PNLF_results 


def calc_PNLF_interp_comp(dM, c2, obs_comp):
    """ Using given dM and c2 best-fit values, along with the completeness profile, 
    this function forms a best fit PNLF, an interpolated PNLF to the observed PNe's m_5007 values, 
    along with completeness corrected, interpolated PNLF.

    Parameters
    ----------
    dM : [float]
        best-fit dM from PNLF fitting
    c2 : [float]
        c2 value to use, normaly kept at 0.307.
    obs_comp : [list / array]
        Observed completeness profile / ratio for the galaxy.

    Returns
    -------
    [lists]
        PNLF_best_fit, PNLF_interp, PNLF_comp_corr
    """    
    step = 0.001
    M_star = -4.53
    M_5007 = np.arange(M_star, 0.53, step)
    m_5007 = np.arange(26, 31, step)

    PNLF_best_fit = calc_PNLF(m_star=M_star+dM, mag=M_5007+dM, c_2=c2)
    PNLF_interp = np.array(np.interp(m_5007, M_5007+dM, PNLF_best_fit))
    PNLF_comp_corr = np.array(np.interp(m_5007, M_5007+dM, PNLF_best_fit)*obs_comp)

    return PNLF_best_fit, PNLF_interp, PNLF_comp_corr


def scale_PNLF(data, PNLF, comp_PNLF, bw, step):
    """Function to scale the PNLf for plotting purposes.
    binwidth * PNLF * (N_data / sum(completeness_corrected_PNLF)*stepsize )

    Parameters
    ----------
    data : [list /array]
        PNe m_5007 values.
    PNLF : [list / array]
        best-fit PNLF array, using values for dM and c2 normally found from fitting.
    comp_PNLF : [list / array]
        Completeness corrected PNLF
    bw : [float]
        binwidth used on the PNe m_5007 histogram. Normally 0.2.
    step : [float]
        Step size, normally the same as used to form M_5007 and m_5007 = 0.001.

    Returns
    -------
    [list / array]
        scale_PNLF: the PNLF scaled to the data, for plotting purposes mainly.
    """    
    scale_PNLF = bw * PNLF *  (len(data) / (np.sum(comp_PNLF)*step))
    return scale_PNLF
