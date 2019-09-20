import numpy as np
import lmfit
from lmfit import minimize, Minimizer, report_fit, Model, Parameters
import scipy as sp
from scipy.stats import norm
from scipy import stats
import pdb as pdb
from astropy.io import fits
from tqdm import tqdm
import sys
import yaml
from MUSE_Models import robust_sigma

vary = sys.argv[1]
if vary == "True":
    vary_PSF = True
elif vary == "False":
    vary_PSF = False

with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
galaxy_data = galaxy_info["FCC167"]
    
def moffat(amplitude, x_0, y_0, FWHM, beta):
    gamma = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2. + (y_fit - y_0)**2.) / gamma**2.
    return amplitude * (1. + rr_gg)**(-beta)

def Gaussian(x, amplitude, mean, LSF, z):
    stddev = LSF / 2.35482
    return (np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.))) + (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / (stddev**2.))

def gen_noise(mu, std):
    noise = np.random.normal(mu, std, len(wavelength))
    return noise

def gen_data(wave, amp, mean, G_FWHM, z, mu, std, p_n, s_n):
    ## Gauss
    gauss_models = np.array([Gaussian(wave, a, mean, G_FWHM, z) + gen_noise(mu, std) for a in amp[p_n]])
    ## error cube
    list_of_std = [np.abs(robust_sigma(spec)) for spec in gauss_models]
    error_cube = [np.repeat(list_of_std[i], len(wave)) for i in range(0,len(list_of_std))]

    return gauss_models, error_cube

def gen_params(M_FWHM, beta, G_FWHM, M_FWHM_v = False, beta_v = False, G_FWHM_v=False):
    sim_params.add('moffat_amp', value=70., min=0.01)
    sim_params.add('x_0', value=((n_pixels//2.) +1),min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    sim_params.add('y_0', value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    sim_params.add('FWHM', value=M_FWHM, min=0.01, max=14.0, vary=M_FWHM_v)
    sim_params.add("beta", value=beta, min=0.01, max=14.0, vary=beta_v)
    sim_params.add("mean", value=5030., min=5015, max=5045)
    sim_params.add("Gauss_LSF", value=G_FWHM, min=G_FWHM-0.5, max=G_FWHM+0.5, vary=G_FWHM_v)
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
    Gauss_std = Gauss_LSF / 2.35482
    A_OIII_xy = ((F_OIII_xy) / (np.sqrt(2*np.pi) * Gauss_std))
    #Construct model gaussian profiles for each amplitude value in cube
    model_spectra = [(Gauss_bkg + (Gauss_grad * x) + Amp * np.exp(- 0.5 * (x - mean)** 2 / Gauss_std**2.) +
             (Amp/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

    # Store things
    list_of_fitted_flux[PNe_number, run] = F_OIII_xy
    list_of_total_fitted_flux[PNe_number, run] = np.sum(F_OIII_xy) * 1e-20
    list_of_resids[PNe_number, run] = robust_sigma(data - model_spectra)
    list_of_A_OIII[PNe_number, run] = np.max(A_OIII_xy)
    list_of_model_spectra[PNe_number, run] = model_spectra
    
    return (data - model_spectra) / error

def chi_square(sim_data, flux_array, wavelength, mean_wave, gal_LSF, G_bkg, G_grad, dof, n_pix, z):
    # chi square analysis
    PNe_n = np.copy(sim_data) # data
    flux_1D = np.copy(flux_array) # flux list
    A_n = ((flux_1D) / (np.sqrt(2*np.pi) * (gal_LSF/ 2.35482)))
    
    def gaussian(x, amplitude, mean, FWHM, bkg, grad, z):
        stddev = FWHM/ 2.35482
        return ((bkg + grad*x) + np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.)) +
                (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / (stddev**2.)))

    list_of_gauss = [gaussian(wavelength, A, mean_wave, gal_LSF, G_bkg, G_grad, z) for A in A_n]
    for kk in range(len(PNe_n)):
        temp = np.copy(list_of_gauss[kk])
        idx  = np.where(PNe_n[kk] == 0.0)[0]
        temp[idx] = 0.0
        PNe_n[kk,idx] = 1.0
        list_of_gauss[kk] = np.copy(temp)
    rN   = robust_sigma(sim_data - list_of_gauss) #data
    res  = PNe_n - list_of_gauss #data
    Chi2 = np.sum((res**2)/(rN**2))
    s    = np.shape(PNe_n)
    redchi = Chi2/((len(wavelength)*n_pix**2) - dof) #dof
    Chi_sqr = Chi2 

    return redchi, Chi_sqr
    
hdulist = fits.open("galaxy_data/FCC167_data/FCC167_residuals_list.fits")
hdr = hdulist[0].header

wavelength = np.exp(hdulist[1].data)
M_5007 = np.arange(-4.52, -1.00, 0.05)
dM = 5 * np.log10(18.739) + 25
m_5007 = M_5007 + dM

c = 299792458.0 # speed of light
gal_vel = 1878
z = gal_vel*1e3 / c

total_flux_list = 10.**((m_5007 + 13.74) / -2.5)
flux = total_flux_list / 1e-20

n_pixels = 9
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

n_PNe = len(m_5007)

n_sim = 200

print("Number of PNe:", n_PNe)
print("Number of Simulations:", n_sim)
print("Total number of simulations to run:", n_PNe * n_sim)

init_g_FWHM = 3.281
init_wave = wavelength[219]
init_FWHM = 3.9
init_beta = 2.4

mu = 0.7634997156910988
std = 5.890678560485462

sum_initial = np.sum(moffat(1., n_pixels/2., n_pixels/2., init_FWHM, init_beta))
input_moff_A = flux / sum_initial

# Make moffat models = F_5007 (x,y)
Moffat_models = np.array([moffat(moff_A, n_pixels/2, n_pixels/2, init_FWHM, init_beta) for moff_A in input_moff_A])

# A_5007 (x,y)
Amp_x_y = ((Moffat_models) / (np.sqrt(2.*np.pi) * (init_g_FWHM / 2.35482)))

## Storage
list_of_A_OIII = np.zeros((n_PNe,n_sim))
list_of_resids = np.zeros((n_PNe,n_sim))
list_of_total_fitted_flux = np.zeros((n_PNe,n_sim))
list_of_fitted_flux = np.zeros((n_PNe, n_sim, n_pixels**2))
list_of_wave = np.zeros((n_PNe,n_sim))
list_of_model_spectra = np.zeros((n_PNe, n_sim, n_pixels**2, len(wavelength)))

list_of_M_amp = np.zeros((n_PNe,n_sim))
list_of_FWHM = np.zeros((n_PNe,n_sim))
list_of_beta = np.zeros((n_PNe,n_sim))
list_of_g_FWHM = np.zeros((n_PNe,n_sim))
list_of_g_grad = np.zeros((n_PNe,n_sim))
list_of_g_bkg  = np.zeros((n_PNe,n_sim)) 

list_of_red_chi = np.zeros((n_PNe,n_sim))
list_of_chi_sqr = np.zeros((n_PNe,n_sim))

list_of_red_chi_lmfit = np.zeros((n_PNe,n_sim))
list_of_chi_sqr_lmfit = np.zeros((n_PNe,n_sim))
##

## Generate Parameters
sim_params = Parameters()
gen_params(M_FWHM=init_FWHM, beta=init_beta, G_FWHM=init_g_FWHM, M_FWHM_v=vary_PSF, beta_v=vary_PSF,
           G_FWHM_v=vary_PSF)
gal_LSF = galaxy_data["LSF"]

gauss_models = np.zeros((n_PNe,n_sim,n_pixels**2,len(wavelength)))
for i in np.arange(0, n_PNe):
    for j in np.arange(0, n_sim):
        gauss_models[i,j] = np.array([Gaussian(wavelength, amp, init_wave, init_g_FWHM, z)+ gen_noise(mu, std) for amp in Amp_x_y[i]]) 
        
#construct error cube
error_cube = np.zeros((n_PNe, n_sim, n_pixels**2, len(wavelength)))

for PNe_num in np.arange(0, n_PNe):
    for sim_num in np.arange(0, n_sim):
        list_of_std = [np.abs(robust_sigma(spec)) for spec in gauss_models[PNe_num, sim_num]]
        error_cube[PNe_num, sim_num] = [np.repeat(list_of_std[i], len(wavelength)) for i in np.arange(0,len(list_of_std))]


for p in tqdm(range(n_PNe)):
    for s in range(n_sim):
        #sim_data, sim_error = gen_data(wavelength, Amp_x_y, init_wave, init_g_FWHM, z, mu, std, p, s)
        result = minimize(sim_residual, sim_params, args=(wavelength, gauss_models[p,s], error_cube[p,s], p, s))
        
        list_of_M_amp[p, s] = result.params["moffat_amp"].value
        list_of_FWHM[p, s] = result.params["FWHM"].value
        list_of_beta[p, s] = result.params["beta"].value
        list_of_g_FWHM[p, s] = result.params["Gauss_LSF"].value
        list_of_wave[p, s] = result.params["mean"].value
        list_of_g_grad[p,s] = result.params["Gauss_grad"].value
        list_of_g_bkg[p,s] = result.params["Gauss_bkg"].value
        list_of_red_chi_lmfit[p,s] = result.redchi
        list_of_chi_sqr_lmfit[p,s] = result.chisqr
        
        list_of_red_chi[p,s], list_of_chi_sqr[p,s] = chi_square(gauss_models[p,s], list_of_fitted_flux[p,s],
                                                                wavelength, result.params["mean"].value, 
                                                                init_g_FWHM, result.params["Gauss_bkg"].value,
                                                                result.params["Gauss_grad"].value, result.nvarys,
                                                                n_pixels, z)


A_by_rN = list_of_A_OIII / list_of_resids

if vary_PSF == True:
    
    #### Save files ####
    np.save("exported_data/simulations/script/Free_PSF/input_moff_A", input_moff_A)
    np.save("exported_data/simulations/script/Free_PSF/output_moff_A", list_of_M_amp)
    np.save("exported_data/simulations/script/Free_PSF/output_FWHM", list_of_FWHM)
    np.save("exported_data/simulations/script/Free_PSF/output_beta", list_of_beta)
    np.save("exported_data/simulations/script/Free_PSF/A_by_rN", A_by_rN)
    np.save("exported_data/simulations/script/Free_PSF/total_F", list_of_total_fitted_flux)
    np.save("exported_data/simulations/script/Free_PSF/output_wave", list_of_wave)
    np.save("exported_data/simulations/script/Free_PSF/output_g_FWHM", list_of_g_FWHM)
    np.save("exported_data/simulations/script/Free_PSF/red_chi", list_of_red_chi)
    np.save("exported_data/simulations/script/Free_PSF/chi_sqr", list_of_chi_sqr)
    np.save("exported_data/simulations/script/Free_PSF/red_chi_lmfit", list_of_red_chi_lmfit)
    np.save("exported_data/simulations/script/Free_PSF/chi_sqr_lmfit", list_of_chi_sqr_lmfit)
    print("Free PSF result files saved.")


elif vary_PSF == False:
    
    #### save files ####
    np.save("exported_data/simulations/script/Fixed_PSF/input_moff_A", input_moff_A)
    np.save("exported_data/simulations/script/Fixed_PSF/output_moff_A", list_of_M_amp)
    np.save("exported_data/simulations/script/Fixed_PSF/output_FWHM", list_of_FWHM)
    np.save("exported_data/simulations/script/Fixed_PSF/output_beta", list_of_beta)
    np.save("exported_data/simulations/script/Fixed_PSF/A_by_rN", A_by_rN)
    np.save("exported_data/simulations/script/Fixed_PSF/total_F", list_of_total_fitted_flux)
    np.save("exported_data/simulations/script/Fixed_PSF/output_wave", list_of_wave)
    np.save("exported_data/simulations/script/Fixed_PSF/output_g_FWHM", list_of_g_FWHM)
    np.save("exported_data/simulations/script/Fixed_PSF/red_chi", list_of_red_chi)
    np.save("exported_data/simulations/script/Fixed_PSF/chi_sqr", list_of_chi_sqr)
    np.save("exported_data/simulations/script/Fixed_PSF/red_chi_lmfit", list_of_red_chi_lmfit)
    np.save("exported_data/simulations/script/Fixed_PSF/chi_sqr_lmfit", list_of_chi_sqr_lmfit)
    print("Fixed PSF result files saved.")
