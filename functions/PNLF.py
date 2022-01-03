import lmfit
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm.notebook import trange, tqdm
from astropy.io import fits
from sklearn.utils import resample
from lmfit import Parameters
from functions.file_handling import paths
from scipy import stats


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
    y = np.linspace(0.0, 1.0, len(data))
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
        wave = hdr['CRVAL3']+(np.arange(s[0]))*hdr['CD3_3']   
        cond = (wave >= 4900.0) & (wave <= 5100.0)
        data = np.sum(hdu[1].data[cond,:,:], axis=0)

    return data, wave, hdr

def form_PNLF_CDF(data, PNLF, dM, obs_comp, M_5007, m_5007):
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

def PNLF_fitter(params, data, data_err, obs_comp, M_5007, m_5007, min_stat="KS_1samp", comp_lim=False):
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

    if min_stat == "chi2":
        if comp_lim == True:
            completeness_lim_mag = m_5007[obs_comp>=0.5].max()
            PNLF_CDF = form_PNLF_CDF(data, PNLF, dM, obs_comp, M_5007, m_5007)
            PNLF_CDF = PNLF_CDF[data<completeness_lim_mag]
            x, PNe_CDF = ecdf(data)
            PNe_CDF = PNe_CDF[data<completeness_lim_mag]

        elif comp_lim == False:
            PNLF_CDF = form_PNLF_CDF(data, PNLF, dM, obs_comp, M_5007, m_5007)
            x, PNe_CDF = ecdf(data)

        return np.abs(PNe_CDF-PNLF_CDF) #/ np.sort(data_err)

    elif min_stat == "KS_1samp":
        KS_stat, pvalue = stats.ks_1samp(data, form_PNLF_CDF, args=(PNLF, dM, obs_comp, M_5007, m_5007 ))
        return KS_stat



def PNLF_analysis(galaxy_name, loc, data, data_err, obs_comp, M_5007, m_5007, dM_in=31.5, c2_in=0.307,
                 vary_dict={"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}, mcmc=False, min_stat="KS_1samp", comp_lim=False):  
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
        PNLF_params.add("dM", value=dM_in, min=dM_in-2, max=dM_in+2, vary=vary_dict["dM"], brute_step=0.001)
    elif loc in ["middle", "halo"]:
        gal_df = pd.read_csv("exported_data/galaxy_dataframe.csv", index_col=("Galaxy", "loc"))
        center_dM = gal_df.loc[(galaxy_name, "center"), "PNLF dM"]
        PNLF_params.add("dM", value=center_dM, min=29.5, max=33.0, vary=False)

    PNLF_params.add("c1", value=1, min=0.00, vary=vary_dict["c1"], brute_step=0.001)
    PNLF_params.add("c2", value=c2_in, min=c2_in-1.5, max=c2_in+1.5, vary=vary_dict["c2"], brute_step=0.001)
    PNLF_params.add("c3", value=3., min=0.0001, max=10, vary=vary_dict["c3"], brute_step=0.001)
    PNLF_params.add("M_star", value=-4.53, min=-4.7, max=-4.3, vary=vary_dict["M_star"], brute_step=0.001)

    PNLF_minimizer = lmfit.Minimizer(PNLF_fitter, PNLF_params, fcn_args=(data, data_err, obs_comp, M_5007, m_5007, min_stat, comp_lim), nan_policy="propagate")
    if min_stat == "chi2":
        PNLF_results = PNLF_minimizer.minimize()
    elif min_stat == "KS_1samp":
        PNLF_results = PNLF_minimizer.scalar_minimize(method="slsqp", tol=1e-6)#, options={"initial_simplex":np.array([[c2_in-0.3, dM_in-0.2] ,  [c2_in+0.3, dM_in] , [c2_in-0.3, dM_in+0.2] ])})

    ## testing
    if mcmc == True:
        PNLF_results.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(20))
        res = lmfit.minimize(PNLF_fitter, args=(data, data_err, obs_comp, M_5007, m_5007, min_stat, comp_lim), method='emcee', nan_policy='omit', nwalkers=200, burn=700, steps=3000, thin=20,
                     params=PNLF_results.params, is_weighted=False, progress=True)
        
        return res

    else:
        return PNLF_results


def MC_PNLF_runner(galaxy_name, loc, data, data_err, obs_comp, dM_in, c2_in=0.307, n_runs=1000,
                    vary_dict={"dM":True, "M_star":False, "c1":False, "c2":False, "c3":False}, min_stat="KS_1samp", comp_lim=False):
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
    data_err : list / array
        PNe apparent mangitude errors.
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

    slope, intercept, r_value, p_value, std_err = stats.linregress(data, data_err)
    data_shift_err = (slope*data) + intercept

    step = 0.001
    M_star = -4.53
    m_5007 = np.arange(22, 34, step)
    M_5007 = np.arange(M_star, M_star+12, step)
    
    MC_dM = []
    MC_c2 = []


    PNLF = calc_PNLF(M_star+dM_in, M_5007+dM_in, c_2=c2_in)
    PNLF_comp_corr = np.array(np.interp(m_5007, M_5007+dM_in, PNLF)*obs_comp)

    for _ in range(n_runs):

        n_PNe_synth = np.random.poisson(len(data))
        while n_PNe_synth < 5:
            n_PNe_synth = np.random.poisson(len(data))

        PNLF_sample = np.interp(np.random.uniform(0,1,n_PNe_synth), np.nancumsum(PNLF_comp_corr)/np.sum(PNLF_comp_corr), m_5007)
        PNLF_sample_err = (slope*PNLF_sample) + intercept
        PNLF_m_distrs = np.array([np.random.normal(m_PN, m_PN_err, size=1000) for m_PN, m_PN_err in zip(PNLF_sample, PNLF_sample_err)])
        synth_PNe = np.array([resample(m_distr, n_samples=1)[0] for m_distr in PNLF_m_distrs])

        PNLF_results = PNLF_analysis(galaxy_name, loc, synth_PNe, data_err, obs_comp, M_5007, m_5007, c2_in=c2_in, vary_dict=vary_dict, min_stat=min_stat, comp_lim=comp_lim)
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
    m_5007 = np.arange(22, 34, step)
    M_5007 = np.arange(M_star, M_star+12, step)

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
