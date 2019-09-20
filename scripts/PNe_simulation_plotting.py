import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
from MUSE_Models import robust_sigma
from astropy.io import fits


vary = sys.argv[1]
if vary == "True":
    vary_PSF = True
elif vary == "False":
    vary_PSF = False

with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
    
galaxy_data = galaxy_info["FCC167"]
    
def Moffat(amplitude, x_0, y_0, FWHM, beta):
    r_d = FWHM / (2. * np.sqrt(2.**(1./beta) - 1.))
    rr_gg = ((x_fit - x_0)**2. + (y_fit - y_0)**2.) / r_d**2.
    return amplitude * (1. + rr_gg)**(-beta)

def Gaussian(x, amplitude, mean, LSF, z):
    stddev = LSF / 2.35482
    return (np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.))) + (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399)*(1+z))** 2 / (stddev**2.))

def gen_noise(mu, std):
    noise = np.random.normal(mu, std, len(wavelength))
    return noise

def gen_data(wave, amp, G_FWHM, z, mu, std, p_n, s_n):
    ## Gauss
    gauss_models = np.array([Gaussian(wave, a, wave[219], G_FWHM, z) + gen_noise(mu, std) for a in amp[p_n]])
    ## error cube
    list_of_std = [np.abs(np.std(spec)) for spec in gauss_models]
    error_cube = [np.repeat(list_of_std[i], len(wave)) for i in range(0,len(list_of_std))]

    return gauss_models, error_cube

def gen_params(M_FWHM, beta, G_FWHM, M_FWHM_v = False, beta_v = False):
    sim_params.add('moffat_amp', value=70., min=0.01)
    sim_params.add('x_0', value=((n_pixels//2.) +1),min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    sim_params.add('y_0', value=((n_pixels//2.) +1), min=((n_pixels//2.) +1)-3, max=((n_pixels//2.) +1)+3)
    sim_params.add('FWHM', value=M_FWHM, min=0.01, max=14.0, vary=M_FWHM_v)
    sim_params.add("beta", value=beta, min=0.01, max=14.0, vary=beta_v)
    sim_params.add("mean", value=5030., min=5015, max=5045)
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
             (Amp/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / Gauss_std**2.)) for Amp in A_OIII_xy]

    # Store things
    list_of_fitted_flux[PNe_number, run] = F_OIII_xy
    list_of_total_fitted_flux[PNe_number, run] = np.sum(F_OIII_xy) * 1e-20
    list_of_resids[PNe_number, run] = np.std(data - model_spectra)
    list_of_A_OIII[PNe_number, run] = np.max(A_OIII_xy)
    
    return (data - model_spectra) / error

def chi_square(data, flux_array, wavelength, mean_wave, gal_LSF, G_bkg, G_grad, dof, n_pix):
    # chi square analysis
    gauss_list, redchi, Chi_sqr = [], [], []
    PNe_n = np.copy(data) # data
    flux_1D = np.copy(flux_array) # flux list
    A_n = ((flux_1D) / (np.sqrt(2*np.pi) * 1.19))
    
    def gaussian(x, amplitude, mean, FWHM, bkg, grad):
        stddev = FWHM/ 2.35482
        return ((bkg + grad*x) + np.abs(amplitude) * np.exp(- 0.5 * (x - mean)** 2 / (stddev**2.)) +
                (np.abs(amplitude)/2.85) * np.exp(- 0.5 * (x - (mean - 47.9399*(1+z)))** 2 / (stddev**2.)))
    
    list_of_gauss = [gaussian(wavelength, A, mean_wave, gal_LSF, G_bkg, G_grad) for A in A_n]
#     for kk in range(len(PNe_n)):
    temp = np.copy(list_of_gauss)
    idx  = np.where(data == 0.0)[0] # data
    temp[idx] = 0.0
    PNe_n[idx] = 1.0
    list_of_gauss = np.copy(temp)
    rN   = robust_sigma(data - list_of_gauss) #data
    res  = data - list_of_gauss #data
    Chi2 = np.sum((res**2)/(rN**2))
    s    = np.shape(data)
    redchi.append(Chi2/(len(wavelength)*n_pix**2 - dof)) #dof
    gauss_list.append(list_of_gauss)
    Chi_sqr.append(Chi2)
        
    return redchi, Chi_sqr


if vary_PSF == True:
    #### load files ####
    input_moff_A  = np.load("exported_data/simulations/script/Free_PSF/input_moff_A.npy")
    list_of_M_amp = np.load("exported_data/simulations/script/Free_PSF/output_moff_A.npy")
    list_of_FWHM  = np.load("exported_data/simulations/script/Free_PSF/output_FWHM.npy")
    list_of_beta  = np.load("exported_data/simulations/script/Free_PSF/output_beta.npy")
    A_by_rN       = np.load("exported_data/simulations/script/Free_PSF/A_by_rN.npy")
    list_of_total_fitted_flux = np.load("exported_data/simulations/script/Free_PSF/total_F.npy")
    list_of_wave   = np.load("exported_data/simulations/script/Free_PSF/output_wave.npy")
    list_of_g_FWHM = np.load("exported_data/simulations/script/Free_PSF/output_g_FWHM.npy")
    red_chi        = np.load("exported_data/simulations/script/Free_PSF/red_chi.npy")
    chi_sqr        = np.load("exported_data/simulations/script/Free_PSF/chi_sqr.npy")
    red_chi_lmfit  = np.load("exported_data/simulations/script/Free_PSF/red_chi_lmfit.npy")
    chi_sqr_lmfit  = np.load("exported_data/simulations/script/Free_PSF/chi_sqr_lmfit.npy")
    
elif vary_PSF == False:
    #### load files ####
    input_moff_A  = np.load("exported_data/simulations/script/Fixed_PSF/input_moff_A.npy")
    list_of_M_amp = np.load("exported_data/simulations/script/Fixed_PSF/output_moff_A.npy")
    list_of_FWHM  = np.load("exported_data/simulations/script/Fixed_PSF/output_FWHM.npy")
    list_of_beta  = np.load("exported_data/simulations/script/Fixed_PSF/output_beta.npy")
    A_by_rN       = np.load("exported_data/simulations/script/Fixed_PSF/A_by_rN.npy")
    list_of_total_fitted_flux = np.load("exported_data/simulations/script/Fixed_PSF/total_F.npy")
    list_of_wave   = np.load("exported_data/simulations/script/Fixed_PSF/output_wave.npy")
    list_of_g_FWHM = np.load("exported_data/simulations/script/Fixed_PSF/output_g_FWHM.npy")
    red_chi        = np.load("exported_data/simulations/script/Fixed_PSF/red_chi.npy")
    chi_sqr        = np.load("exported_data/simulations/script/Fixed_PSF/chi_sqr.npy")
    red_chi_lmfit  = np.load("exported_data/simulations/script/Fixed_PSF/red_chi_lmfit.npy")
    chi_sqr_lmfit  = np.load("exported_data/simulations/script/Fixed_PSF/chi_sqr_lmfit.npy")



hdulist = fits.open("galaxy_data/FCC167_data/FCC167_residuals_list.fits")
hdr = hdulist[0].header
data = hdulist[0].data
x_data = hdr["XAXIS"]
y_data = hdr["YAXIS"]

wavelength = np.exp(hdulist[1].data)
M_5007 = np.arange(-4.52, 0.00, 0.05)
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

n_PNe = len(input_moff_A)

n_sim = len(A_by_rN[0])

print("Number of PNe:", n_PNe)
print("Number of Simulations:", n_sim)
print("Total number of simulations to run:", n_PNe * n_sim)

init_g_FWHM = 3.281
init_wave = wavelength[219]
init_FWHM = 3.9
init_beta = 2.4

mu = 0.7634997156910988
std = 5.890678560485462

sum_initial = np.sum(Moffat(1., n_pixels/2., n_pixels/2., init_FWHM, init_beta))
input_moff_A = flux / sum_initial

# Make moffat models = F_5007 (x,y)
Moffat_models = np.array([Moffat(moff_A, n_pixels/2, n_pixels/2, init_FWHM, init_beta) for moff_A in input_moff_A])

# A_5007 (x,y)
Amp_x_y = ((Moffat_models) / (np.sqrt(2.*np.pi) * 1.19))


if vary_PSF == True:
    #### load files ####
    input_moff_A  = np.load("exported_data/simulations/script/Free_PSF/input_moff_A.npy")
    list_of_M_amp = np.load("exported_data/simulations/script/Free_PSF/output_moff_A.npy")
    list_of_FWHM  = np.load("exported_data/simulations/script/Free_PSF/output_FWHM.npy")
    list_of_beta  = np.load("exported_data/simulations/script/Free_PSF/output_beta.npy")
    A_by_rN       = np.load("exported_data/simulations/script/Free_PSF/A_by_rN.npy")
    list_of_total_fitted_flux = np.load("exported_data/simulations/script/Free_PSF/total_F.npy")
    list_of_wave   = np.load("exported_data/simulations/script/Free_PSF/output_wave.npy")
    list_of_g_FWHM = np.load("exported_data/simulations/script/Free_PSF/output_g_FWHM.npy")
    red_chi        = np.load("exported_data/simulations/script/Free_PSF/red_chi.npy")
    chi_sqr        = np.load("exported_data/simulations/script/Free_PSF/chi_sqr.npy")
    red_chi_lmfit  = np.load("exported_data/simulations/script/Free_PSF/red_chi_lmfit.npy")
    chi_sqr_lmfit  = np.load("exported_data/simulations/script/Free_PSF/chi_sqr_lmfit.npy")
    
elif vary_PSF == False:
    #### load files ####
    input_moff_A  = np.load("exported_data/simulations/script/Fixed_PSF/input_moff_A.npy")
    list_of_M_amp = np.load("exported_data/simulations/script/Fixed_PSF/output_moff_A.npy")
    list_of_FWHM  = np.load("exported_data/simulations/script/Fixed_PSF/output_FWHM.npy")
    list_of_beta  = np.load("exported_data/simulations/script/Fixed_PSF/output_beta.npy")
    A_by_rN       = np.load("exported_data/simulations/script/Fixed_PSF/A_by_rN.npy")
    list_of_total_fitted_flux = np.load("exported_data/simulations/script/Fixed_PSF/total_F.npy")
    list_of_wave   = np.load("exported_data/simulations/script/Fixed_PSF/output_wave.npy")
    list_of_g_FWHM = np.load("exported_data/simulations/script/Fixed_PSF/output_g_FWHM.npy")
    red_chi        = np.load("exported_data/simulations/script/Fixed_PSF/red_chi.npy")
    chi_sqr        = np.load("exported_data/simulations/script/Fixed_PSF/chi_sqr.npy")
    red_chi_lmfit  = np.load("exported_data/simulations/script/Fixed_PSF/red_chi_lmfit.npy")
    chi_sqr_lmfit  = np.load("exported_data/simulations/script/Fixed_PSF/chi_sqr_lmfit.npy")
    
m_5007_out = -2.5 * np.log10(list_of_total_fitted_flux) - 13.74


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
        delta_m_5007[pne, sim] = m_5007[pne] - m_5007_out[pne, sim]
        #delta_m_5007[pne, sim] = ((app_m_list[pne] - m_5007_out[pne, sim])/app_m_list[pne])*100
        
        
# delta_FWHM = init_FWHM - list_of_FWHM
# delta_FWHM = ((init_FWHM - list_of_FWHM)/init_FWHM)*100
# delta_beta = init_beta - list_of_beta
# delta_beta = ((init_beta - list_of_beta) / init_beta)*100

delta_FWHM = init_FWHM - list_of_FWHM
delta_beta = init_beta - list_of_beta
delta_wave = init_wave - list_of_wave
delta_g_FWHM = init_g_FWHM - list_of_g_FWHM
delta_g_stddev = delta_g_FWHM[:] / 2.35482

de_z_means = np.array(list_of_wave / (1 + z))
list_of_v = (c * (de_z_means - 5006.77) / 5006.77)/1000.
init_de_z = np.array(init_wave / (1 + z))
init_v = (c * (init_de_z - 5006.77) / 5006.77)/1000.

delta_v = init_v - list_of_v
delta_sigma = ((c * delta_g_stddev) / list_of_wave)/1000.

delta_total_F = delta_total_F*1e-20

if vary_PSF == True:
    #### Plotting with Free PSF ####
        
    ## Bin data here
    q = 1
    n_bins = np.arange(0,np.ceil(np.max(A_by_rN)),0.5)
    digitized = np.digitize(A_by_rN, n_bins)
    
    def mean_16_84(data):
        binned_data_mean = [np.percentile(data[digitized == i], 50) for i in range(q, len(n_bins))]
        binned_data_16 = [np.percentile(data[digitized == i], 16) for i in range(q, len(n_bins))]
        binned_data_84 = [np.percentile(data[digitized == i], 84) for i in range(q, len(n_bins))]
        
        return binned_data_mean, binned_data_16, binned_data_84
    
    binned_F, binned_F_16, binned_F_84 = mean_16_84(delta_total_F)
    binned_FWHM, binned_FWHM_16, binned_FWHM_84 = mean_16_84(delta_FWHM)
    binned_beta, binned_beta_16, binned_beta_84 = mean_16_84(delta_beta)
    binned_m5007, binned_m5007_16, binned_m5007_84 = mean_16_84(delta_m_5007)
    binned_v, binned_v_16, binned_v_84 = mean_16_84(delta_v)
    binned_sigma, binned_sigma_16, binned_sigma_84 = mean_16_84(delta_sigma)
    
    x_axis = n_bins[q:]
    
    ## Plot data here
    fig = plt.figure(1, figsize=(15,30))
    fig.subplots_adjust(hspace=0.3)
    label_f_s = 20
    
    ## FLUX
    ax0 = plt.subplot(7,1,1)
    plt.scatter(A_by_rN, delta_total_F, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel("Delta $F_{5007} \ {}_{(10^{-18} \ erg \ s^{-1} cm^{-2})}$", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    #ax0.annotate("Under estimated", xy=(1,1e-17), xytext=(11,1e-17), fontsize=14)
    #ax0.annotate("Over estimated", xy=(1,1e-17), xytext=(11,-1e-17), fontsize=14)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_F, label="median", c="r")
    plt.plot(x_axis, binned_F, c="r")
    plt.xlim(-0.5,14)
    plt.fill_between(x_axis, binned_F_16, binned_F_84, color="red", alpha=0.4, linestyle="None")
    plt.ylim(-0.5e-17,0.5e-17)
    ax0.yaxis.offsetText.set_visible(False)#.set_fontsize(20)
    
    ## m5007
    ax1 = plt.subplot(7,1,2)
    plt.scatter(A_by_rN, delta_m_5007, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel("Delta $m_{5007}$", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    #ax0.annotate("Under estimated", xy=(1,1e-17), xytext=(11,1e-17), fontsize=14)
    #ax0.annotate("Over estimated", xy=(1,1e-17), xytext=(11,-1e-17), fontsize=14)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_m5007, label="median", c="r")
    plt.plot(x_axis, binned_m5007, c="r")
    plt.xlim(-0.5,14)
    plt.fill_between(x_axis, binned_m5007_16, binned_m5007_84, color="red", alpha=0.4, linestyle="None")
    plt.ylim(-1,1)
    #ax0.yaxis.offsetText.set_fontsize(20)
    
    
    ## FWHM
    ax2 = plt.subplot(7,1,3)
    plt.scatter(A_by_rN, delta_FWHM, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel("Delta FWHM", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    plt.ylim(-4,4)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_FWHM, label="median", c="r")
    plt.plot(x_axis, binned_FWHM, c="r")
    plt.xlim(-0.5,14)
    plt.fill_between(x_axis, binned_FWHM_16, binned_FWHM_84, color="red", alpha=0.4, linestyle="None")
    
    ## Beta
    
    ax3 = plt.subplot(7,1,4)
    plt.scatter(A_by_rN, delta_beta, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel(r"Delta $\beta$", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    plt.ylim(-12,3)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_beta, label="median", c="r")
    plt.plot(x_axis, binned_beta, c="r")
    plt.xlim(-0.5,14)
    plt.fill_between(x_axis, binned_beta_16, binned_beta_84, color="red", alpha=0.4, linestyle="None")
    
    ax4 = plt.subplot(7,1,5)
    plt.scatter(A_by_rN, delta_v, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel(r"Delta radial V $\ (km \ s^{-1})$", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    plt.ylim(-100,100)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_v, label="median", c="r")
    plt.plot(x_axis, binned_v, c="r")
    plt.xlim(-0.5,14)
    plt.fill_between(x_axis, binned_v_16, binned_v_84, color="red", alpha=0.4, linestyle="None")
    
    ax5 = plt.subplot(7,1,6)
    plt.scatter(A_by_rN, delta_sigma, s=5, rasterized=True)
    plt.xlabel("A/rN", fontsize=label_f_s)
    plt.ylabel(r"Delta $\sigma \ (km \ s^{-1})$ ", fontsize=label_f_s)
    plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    plt.axvline(x=3, c="k", ls="dashed")
    #plt.ylim(-0.4,0.6)
    plt.tick_params(labelsize = label_f_s)
    ## binned median
    plt.scatter(x_axis, binned_sigma, label="median", c="r")
    plt.plot(x_axis, binned_sigma, c="r")
    plt.xlim(-0.5,14)
    plt.ylim(-14,14)
    plt.fill_between(x_axis, binned_sigma_16, binned_sigma_84, color="red", alpha=0.4, linestyle="None")
    #ax1 = plt.subplot(2,1,1)
    #plt.scatter(A_by_rN, red_chi)
    #plt.ylabel("reduced Chi_sqr")
    #
    #ax2 = plt.subplot(2,1,2)
    #plt.scatter(A_by_rN, chi_sqr)
    #plt.ylabel("Chi_sqr")
    #
    #plt.show()
    
    plt.savefig("Plots/simulations/free_PSF_sims.pdf", bbox_inches='tight')
    plt.savefig("Plots/simulations/free_PSF_sims.png", bbox_inches='tight')#, dpi=500)

elif vary_PSF == False:
    ## Bin data here
    q = 1
    n_bins = np.arange(0,np.ceil(np.max(A_by_rN)))
    digitized = np.digitize(A_by_rN, n_bins)
    
    def mean_16_84(data):
        binned_data_mean = [np.percentile(data[digitized == i], 50) for i in range(q, len(n_bins))]
        binned_data_16 = [np.percentile(data[digitized == i], 16) for i in range(q, len(n_bins))]
        binned_data_84 = [np.percentile(data[digitized == i], 84) for i in range(q, len(n_bins))]
        
        return binned_data_mean, binned_data_16, binned_data_84
    
    binned_F, binned_F_16, binned_F_84 = mean_16_84(delta_total_F)
    binned_m5007, binned_m5007_16, binned_m5007_84 = mean_16_84(delta_m_5007)
    binned_v, binned_v_16, binned_v_84 = mean_16_84(delta_v)
    
    x_axis = n_bins[q:]
    label_f_s = 20
    
    ## Plot data here
    fig = plt.figure(1, figsize=(15,10)) #15,20
    fig.subplots_adjust(hspace=0.3)
    
    ax1 = plt.subplot(2,1,1)
    plt.scatter(A_by_rN, red_chi)
    plt.ylabel("reduced Chi_sqr")
    
    ax2 = plt.subplot(2,1,2)
    plt.scatter(A_by_rN, chi_sqr)
    plt.ylabel("Chi_sqr")
    
    plt.show()
    
    
    ## FLUX
    #ax0 = plt.subplot(3,1,1)
    #plt.scatter(A_by_rN, delta_total_F, s=5, rasterized=True)
    #plt.xlabel("A/rN", fontsize=label_f_s)
    #plt.ylabel("Delta $F_{5007} \ {}_{(10^{-18} \ erg \ s^{-1} cm^{-2})}$", fontsize=label_f_s)
    #plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    #plt.axvline(x=3, c="k", ls="dashed")
    ##ax0.annotate("Under estimated", xy=(1,1e-17), xytext=(11,1e-17), fontsize=14)
    ##ax0.annotate("Over estimated", xy=(1,1e-17), xytext=(11,-1e-17), fontsize=14)
    #plt.tick_params(labelsize = label_f_s)
    ### binned median
    #plt.scatter(x_axis, binned_F, label="median", c="r")
    #plt.plot(x_axis, binned_F, c="r")
    #plt.fill_between(x_axis, binned_F_16, binned_F_84, color="red", alpha=0.4, linestyle="None")
    #plt.ylim(-0.5e-17,0.5e-17)
    #ax0.yaxis.offsetText.set_visible(False) #set_fontsize(label_f_s)
    #
    ### m5007
    #ax1 = plt.subplot(3,1,2)
    #plt.scatter(A_by_rN, delta_m_5007, s=5, rasterized=True)
    #plt.xlabel("A/rN", fontsize=label_f_s)
    #plt.ylabel("Delta $m_{5007}$", fontsize=label_f_s)
    #plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    #plt.axvline(x=3, c="k", ls="dashed")
    ##ax0.annotate("Under estimated", xy=(1,1e-17), xytext=(11,1e-17), fontsize=14)
    ##ax0.annotate("Over estimated", xy=(1,1e-17), xytext=(11,-1e-17), fontsize=14)
    #plt.tick_params(labelsize = label_f_s)
    ### binned median
    #plt.scatter(x_axis, binned_m5007, label="median", c="r")
    #plt.plot(x_axis, binned_m5007, c="r")
    #plt.fill_between(x_axis, binned_m5007_16, binned_m5007_84, color="red", alpha=0.4, linestyle="None")
    #plt.ylim(-1,1)
    ##ax0.yaxis.offsetText.set_fontsize(20)
    #
    #ax4 = plt.subplot(3,1,3)
    #plt.scatter(A_by_rN, delta_v, s=5, rasterized=True)
    #plt.xlabel("A/rN", fontsize=label_f_s)
    #plt.ylabel(r"Delta radial V $\ (km \ s^{-1})$", fontsize=label_f_s)
    #plt.axhline(y=0, c="k", ls="dashed", alpha=0.5)
    #plt.axvline(x=3, c="k", ls="dashed")
    #plt.ylim(-100,100)
    #plt.tick_params(labelsize = label_f_s)
    ### binned median
    #plt.scatter(x_axis, binned_v, label="median", c="r")
    #plt.plot(x_axis, binned_v, c="r")
    #plt.fill_between(x_axis, binned_v_16, binned_v_84, color="red", alpha=0.4, linestyle="None")
    
    #plt.savefig("Plots/simulations/fixed_PSF_sims.pdf", bbox_inches='tight')
    #plt.savefig("Plots/simulations/fixed_PSF_sims.png", bbox_inches='tight')