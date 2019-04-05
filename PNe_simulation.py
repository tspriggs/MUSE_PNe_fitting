import numpy as np
import lmfit
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import scipy as sp
from scipy.stats import norm
from scipy import stats
import pdb as pdb
from astropy.io import fits
from MUSE_Models import data_cube_y_x

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

    
def Moffat(amplitude, x_0, y_0, FWHM, beta):
    r_d = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2. + (y_fit - y_0)**2.) / r_d**2.
    return amplitude * (1. + rr_gg)**(-beta)

def Gaussian(x, amplitude, mean, LSF):
    stddev = LSF / 2.35482
    return (np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.))) + (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399))** 2 / (stddev**2.))

def gen_noise(mu, std):
    noise = np.random.normal(mu, std, len(wavelength))
    return noise

def gen_data(wave, amp, G_FWHM, mu, std, p_n, s_n):
    ## Gauss
    gauss_models = np.array([Gaussian(wave, a, wave[219], G_FWHM) + gen_noise(mu, std) for a in amp[p_n]])
    ## error cube
    list_of_std = [np.abs(np.std(spec)) for spec in gauss_models]
    error_cube = [np.repeat(list_of_std[i], len(wave)) for i in range(0,len(list_of_std))]

    return gauss_models, error_cube

def gen_params(M_FWHM, beta, G_FWHM, M_FWHM_v = False, beta_v = False):
    sim_params.add('moffat_amp', value=70., min=0.01)
    sim_params.add('x_0', value=((n_pixels//2.) +1), min=0.01, max=n_pixels)
    sim_params.add('y_0', value=((n_pixels//2.) +1), min=0.01, max=n_pixels)
    sim_params.add('FWHM', value=M_FWHM, min=0.01, max=14.0, vary=M_FWHM_v)
    sim_params.add("beta", value=beta, min=0.01, max=14.0, vary=beta_v)
    sim_params.add("mean", value=5030., min=5000, max=5070)
    sim_params.add("Gauss_LSF", value=G_FWHM, vary=False)
    sim_params.add("Gauss_bkg", value=0.001)
    sim_params.add("Gauss_grad", value=0.001)

def sim_residual(params, x, data, error, PNe_number, run):
    # List of parameters
    moffat_amp = params['moffat_amp']
    x_0 = params['x_0']
    y_0 = params['y_0']
    FWHM = params['FWHM']
    beta = params["beta"]
    mean = params["mean"]
    Gauss_LSF = params["Gauss_LSF"]
    Gauss_bkg = params["Gauss_bkg"]
    Gauss_grad = params["Gauss_grad"]
    #Moffat model
    gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2. + (y_fit - y_0)**2.) / gamma**2.
    F_OIII_xy = moffat_amp * (1. + rr_gg)**(-beta)
    # Convert Moffat flux to amplitude
    A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * 1.19))
    Gauss_std = Gauss_LSF / 2.35482
    #Construct model gaussian profiles for each amplitude value in cube
    model_spectra = [(Gauss_bkg + (Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - mean)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

    # Store things
    list_of_fitted_flux[PNe_number, run] = F_OIII_xy
    list_of_total_fitted_flux[PNe_number, run] = np.sum(F_OIII_xy) * 1e-20
    list_of_resids[PNe_number, run] = np.std(data - model_spectra)
    list_of_A_OIII[PNe_number, run] = np.max(A_OIII_xy)

    return (data - model_spectra) / error

hdulist = fits.open("FCC167_data/FCC167_residuals_list.fits")
hdr = hdulist[0].header
data = hdulist[0].data
y_data, x_data, n_data = data_cube_y_x(len(data))

wavelength = np.exp(hdulist[1].data)
M_5007 = np.arange(-4.5, 0.0, 0.05)
dM = 5 * np.log10(18.687) + 25
m_5007 = M_5007 + dM

total_flux_list = 10.**((m_5007 + 13.74) / -2.5)
flux = total_flux_list / 1e-20

n_pixels = 9
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

n_PNe = len(m_5007)

n_sim = 10

print("Number of PNe:", n_PNe)
print("Number of Simulations:", n_sim)
print("Total number of simulations to run:", n_PNe * n_sim)

in_FWHM   = 3.8
in_beta   = 2.4
in_G_FWHM = 3.12

mu = 0.7634997156910988
std = 5.764539800007041

sum_initial = np.sum(Moffat(1., n_pixels/2., n_pixels/2., in_FWHM, in_beta))
input_moff_A = flux / sum_initial

# Make moffat models = F_5007 (x,y)
Moffat_models = np.array([Moffat(moff_A, n_pixels/2, n_pixels/2, in_FWHM, in_beta) for moff_A in input_moff_A])

# A_5007 (x,y)
Amp_x_y = ((Moffat_models) / (np.sqrt(2.*np.pi) * 1.19))

## Storage
list_of_A_OIII = np.zeros((n_PNe,n_sim))
list_of_resids = np.zeros((n_PNe,n_sim))
list_of_total_fitted_flux = np.zeros((n_PNe,n_sim))
list_of_fitted_flux = np.zeros((n_PNe, n_sim, n_pixels**2))

list_of_M_amp = np.zeros((n_PNe,n_sim))
list_of_FWHM = np.zeros((n_PNe,n_sim))
list_of_beta = np.zeros((n_PNe,n_sim))
##

## Generate Parameters
sim_params = Parameters()
gen_params(M_FWHM=in_FWHM, beta=in_beta, G_FWHM=in_G_FWHM, M_FWHM_v=False, beta_v=False)

for p in range(n_PNe):
    progbar(p, n_PNe, 20)
    for s in range(n_sim):
        sim_data, sim_error = gen_data(wavelength, Amp_x_y, in_G_FWHM, mu, std, p, s)
        result = minimize(sim_residual, sim_params, args=(wavelength, sim_data, sim_error, p, s))
        list_of_M_amp[p, s] = result.params["moffat_amp"]
        list_of_FWHM[p, s] = result.params["FWHM"]
        list_of_beta[p, s] = result.params["beta"]
        
A_by_rN = list_of_A_OIII / list_of_resids

# Calculate Deltas

m_5007_out = -2.5 * np.log10(list_of_total_fitted_flux) - 13.74

#M_5007_out = m_5007_out - dM

# create plots and delta params, out minus in
delta_moff_amp = np.zeros((n_PNe,n_sim))
for pne in np.arange(0, n_PNe):
    for sim in np.arange(0, n_sim):
        delta_moff_amp[pne, sim] = input_moff_A[pne] - list_of_M_amp[pne, sim]
        #delta_moff_amp[pne, sim] = ((input_moff_A[pne] - list_of_M_amp[pne, sim])/input_moff_A[pne])*100

delta_total_F = np.zeros((n_PNe,n_sim))
for pne in np.arange(0, n_PNe):
    for sim in np.arange(0, n_sim):
        delta_total_F[pne, sim] =  (flux[pne]) - (list_of_total_fitted_flux[pne, sim]/1e-20)
        #delta_total_F[pne, sim] =  ((flux[pne] - (list_of_total_fitted_flux[pne, sim]/1e-20))/flux[pne])*100
        
delta_m_5007 = np.zeros((n_PNe,n_sim))
for pne in np.arange(0, n_PNe):
    for sim in np.arange(0, n_sim):
        delta_m_5007[pne, sim] = app_m_list[pne] - m_5007_out[pne, sim]
        #delta_m_5007[pne, sim] = ((app_m_list[pne] - m_5007_out[pne, sim])/app_m_list[pne])*100
        
        
delta_FWHM = init_FWHM - list_of_FWHM
#delta_FWHM = ((init_FWHM - list_of_FWHM)/init_FWHM)*100
delta_beta = init_beta - list_of_beta
#delta_beta = ((init_beta - list_of_beta) / init_beta)*100

delta_total_F = delta_total_F*1e-20


