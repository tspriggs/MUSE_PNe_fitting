from astropy.io import fits
import numpy as np
import pdb
import copy
import matplotlib.pyplot as plt
from spectral_resampling import spectres
from multiprocessing import Queue, Process

def workerGANDALF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for all_spec, best_fit, emission, i\
        in iter(inQueue.get, 'STOP'):

        spec_er = run_gandalf(all_spec, best_fit, emission, i)

        outQueue.put((spec_er))

def run_gandalf(all_spec, best_fit, emission, i):

#    spe  = spectres(lm_new, old_spec_wavs, all_spec, spec_errs=res_fits)
#
#    spec_er = np.zeros(ss)
#    err_n = spe[1]
#    spec_er[overlp] = err_n**2
#    er_l_lvl = np.nanstd(err_n[0:180])**2 
#    er_r_lvl = np.nanstd(err_n[-180:-1])**2 
#
#    spec_er[noverlp_lf] = spec_er[noverlp_lf]*0. + er_l_lvl
#    spec_er[noverlp_rt] = spec_er[noverlp_rt]*0. + er_r_lvl
    spec_er = all_spec - (best_fit - emission)
    
    return(spec_er)

# Read the original cube ... 

#gal_id = 'FCC083'
#gal_id = 'FCC147'
#gal_id = 'FCC148'
#gal_id = 'FCC161'
#gal_id = 'FCC179'
#gal_id = 'FCC184'
gal_id = 'FCC170'
#gal_id = 'FCC193'
#gal_id = 'FCC219'
#gal_id = 'FCC276'
#gal_id = 'FCC290'
#gal_id = 'FCC308'
#gal_id = 'FCC310'
#gal_id = 'FCC312'
#
cvel      = 299792.458

hdu_pars_used = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+'USED_PARAMS.fits')
#
params = hdu_pars_used[0].header
vsys   = float(params['REDSHIFT'])
#
# Applying some wavelength cuts
hdu_orig = fits.open('./'+ gal_id+'/'+gal_id+'center.fits')
data = hdu_orig[1].data
hdr   = hdu_orig[1].header
s      = np.shape(data)
spec  = np.reshape(data,[s[0],s[1]*s[2]])
wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']
#
idx       = (wave >= float(params['LMIN'])*(1.0 + vsys/cvel)) & \
            (wave <= float(params['LMAX'])*(1.0 + vsys/cvel))
#
flux_or = spec
spec      = spec[idx,:]
idx_good = np.where( np.median(spec, axis=0) > 0.0 )[0]
#

# Read the GANDALF residuals sp. by sp. 

hdu_res = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_gandalf-residuals_SPAXEL.fits')

res_fits = hdu_res[1].data['RESIDUALS']

hdu_bf = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_ppxf-bestfit_SPAXELS.fits')

bf_fits = hdu_bf[1].data['BESTFIT']

hdu_emi = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_gandalf-emission_SPAXEL.fits')

emi_fits = hdu_emi[1].data['EMISSION']

hdu_res = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_ppxf-bestfit_rN.fits')

# Linear Vor bins .. 

#hdu_lin_Vor = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_VorSpectra_linear.fits')
#lam_lin = hdu_lin_Vor[2].data
#lam_lin = lam_lin['LOGLAM']

# Log Vor bins .. 

hdu_log_Vor = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_VorSpectra.fits')
lam_log = hdu_log_Vor[2].data
lam_log = lam_log['LOGLAM']

# Table file 

#hdu_tab = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_table.fits')

# Read all spectra 

hdu_all_spec = fits.open('./'+ gal_id+'/'+gal_id+'center_center/'+gal_id+'center_AllSpectra.fits')

all_spec = hdu_all_spec[1].data.SPEC

# Stat S/N 

#stat_sn = hdu_tab[1].data['SNR']
#flux = hdu_tab[1].data['FLUX'] 

# S/rN

#res_lvl = []
#sig_lvl = []
#sig_rN = []
#

log_l_to_l_lin = np.exp(lam_log)
noverlp_lf = np.where((wave < log_l_to_l_lin[1]) )[0]
noverlp_rt = np.where((wave > log_l_to_l_lin[-2]) )[0]
#
overlp = np.where( (wave >= log_l_to_l_lin[1]) & (wave <= log_l_to_l_lin[-2]) )[0]
#
spec_er = np.zeros(s[0])
old_spec_wavs = log_l_to_l_lin
lm_new = wave[overlp]

#
#i  = 0
#spe  = spectres(lm_new, old_spec_wavs, all_spec[i,:], spec_errs=res_fits[:,i])
#
#spec_er = np.zeros(s[0])
#err_n = spe[1]
#spec_er[overlp] = err_n**2
#er_l_lvl = np.nanstd(err_n[0:180])**2 
#er_r_lvl = np.nanstd(err_n[-180:-1])**2 
#
#spec_er[noverlp_lf] = spec_er[noverlp_lf]*0. + er_l_lvl
#spec_er[noverlp_rt] = spec_er[noverlp_rt]*0. + er_r_lvl
#pdb.set_trace()

ss     = np.array(np.shape(data))
s = np.copy(ss)
free_points = s[1]*s[2]
empty = np.zeros((len(lam_log),free_points))
empty.fill(np.nan)
noise_tot    = np.copy(empty)
noise_tot_good   = np.zeros((len(lam_log),len(idx_good)))
#

# Create Queues
inQueue  = Queue()
outQueue = Queue()
    
# Create worker processes
ps = [Process(target=workerGANDALF, args=(inQueue, outQueue))
      for _ in range(31)]
    
# Start worker processes
for p in ps: p.start()
    
## Fill the queue
for i in range(len(idx_good)-1):
    inQueue.put( ( all_spec[i,:], bf_fits[i,:], emi_fits[i,:], i) )

_tmp = [outQueue.get() for _ in range(len(idx_good)-1)]

# send stop signal to stop iteration
for _ in range(31): inQueue.put('STOP')

# stop processes
for p in ps: p.join()

for i in range(0, len(idx_good)-1): noise_tot_good[:,i] = _tmp[i]

#var = hdu_orig[2].data
#var = np.reshape(var,[s[0],s[1]*s[2]])
#var = var[:,idx_good]
#plt.plot(wave, var[:,-1])
#plt.plot(wave,spec_er)

noise_tot[:,idx_good] = np.copy(noise_tot_good)
noise_tot_rs  = np.copy(np.reshape(noise_tot,[len(lam_log), s[1], s[2]]))

Res_prim_HDU = fits.PrimaryHDU(noise_tot_rs)
hdu_wave = fits.ImageHDU(data=lam_log, name='LOG_LAM')
Res_hdul = fits.HDUList([Res_prim_HDU, hdu_wave])

#noise_list = np.copy(np.swapaxes(noise_tot, 1, 0)) # shape swapped to x*y, lambda
#noise_tot_rs  = noise_tot_rs.astype(np.float32)

#hdu_err = copy.copy(hdu_orig)
#hdr = copy.copy(hdu_orig[1].header)
#hdr['EXTNAME'] = 'STAT'
#hdr['HDUCLAS2'] = 'ERROR'
#hdr['HDUCLAS3'] = 'MSE'
#hdr['SCIDATA'] = 'DATA'
#del hdr['ERRDATA']
#hdr['OBJECT'] = gal_id +' (STAT)'

#STAT_EXT = fits.ImageHDU(data=noise_tot_rs, header=hdr, name='STAT')

#hdu_err.append(STAT_EXT)


Res_hdul.writeto('./'+ gal_id+'/'+gal_id+'center_RES_cube.fits')

#plt.errorbar(np.arange(len(hdu[1].data[:,250,250])), hdu[1].data[:,250,250], yerr=np.sqrt(hdu[2].data[:,250,250]), fmt='.')
#for i in np.arange(len(flux)): 
#    n_lvl = np.std(res_fits[:,i])
#    s_lvl = flux[i]
#    res_lvl.append(n_lvl)
#    sig_lvl.append(s_lvl)
#    sig_rN.append(s_lvl/n_lvl)

#lm_new = np.arange(wave_b[0]+hdr_r['CDELT3'], wave_b[-1]-hdr_r['CDELT3'], hdr_r['CDELT3'])
#spe  = spectres.spectres(lm_new, old_spec_wavs, spec_b[:,i], spec_errs=espec_b[:,i])

#sig_rN = np.array(sig_rN)
#res_lvl = np.array(res_lvl)
#sig_lvl = np.array(sig_lvl)

#sig_rN.dump('sig_rN,pkl')
#res_lvl.dump('res_lvl,pkl')
#sig_lvl.dump('sig_lvl,pkl')
#stat_sn.dump('stat_sn,pkl')


