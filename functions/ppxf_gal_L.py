from astropy.io import fits
from scipy import ndimage
import numpy as np
from time import clock
import glob
import pyphot
import matplotlib.pylab as plt
from   ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from os import path

# def ppxf_L_tot(file, gal_mask_params, star_mask_params, redshift, vel, dist_mod):#, mask=False):
def ppxf_L_tot(int_spec, header, redshift, vel, dist_mod, dM_err=[0.1,0.1] ,calc_L_bol=True):#, mask=False):

    """
    Input: File - file name of the original MUSE cube (after data reduction)
                - redshift of galaxy, taken from Simbad
                - velocity, in km/s, of galaxy, also from Simbad
                - dist_mod: Distance Modulus
                - dM_err: [0.1,0.1] upper and lower error bounds
                - calc_L_bol: Default=True, switch to False to not calculate L_bol
                              This only returns the last 3 items (mag_g, mag_r, mag_v)
    Returns:    
                - lum_bol_g
                - [lum_bol_g_upper, lum_bol_g_lower] errors
                - lum_bol_r 
                - [lum_bol_r_upper, lum_bol_r_lower] errors
                - lum_bol_v
                - [lum_bol_v_upper, lum_bol_v_lower] errors
                - mag_g
                - mag_r
                - mag_v
    """
    
    # Read a galaxy spectrum and define the wavelength range
    
    
    lamRange1 = header['CRVAL3'] + np.array([0., header['CD3_3']*(header['NAXIS3'] - 1)]) #IMPORTANTE: EL RANGO DE LAMBDAS ESTA EN ESCALA LOGARITMICA
    #Transformamos los pixeles en lambdas:
    #lam=np.linspace(lamRange1[0],lamRange[1],len(gal_lin[0,:]))
    FWHM_gal = 2.81  # SAURON has an instrumental resolution FWHM of 4.2A.

    # If the galaxy is at a significant redshift (z > 0.03), one would need to apply
    # a large velocity shift in PPXF to match the template to the galaxy spectrum.
    # This would require a large initial value for the velocity (V > 1e4 km/s)
    # in the input parameter START = [V,sig]. This can cause PPXF to stop!
    # The solution consists of bringing the galaxy spectrum roughly to the
    # rest-frame wavelength, before calling PPXF. In practice there is no
    # need to modify the spectrum before the usual LOG_REBIN, given that a
    # red shift corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    # z = 1.23 # Initial estimate of the galaxy redshift
    # lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    # FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, int_spec)
    cond = np.exp(logLam1) <= 6900
    # Getting the apparent magnitude of the galaxy in the g, r and V bands
    mag_g, Flux_g = library(np.exp(logLam1[cond]),galaxy[cond]*1.0e-20,filt="SDSS", band="g")
    mag_r, Flux_r = library(np.exp(logLam1[cond]),galaxy[cond]*1.0e-20,filt="SDSS", band="r")
    mag_v, Flux_v = library(np.exp(logLam1[cond]),galaxy[cond]*1.0e-20,filt="GROUND_JOHNSON", band="V")

    # Converting to absolute magnitude
    M_g = mag_g - dist_mod #
    M_g_u = mag_g - (dist_mod + dM_err[0]) # upper distance M_g
    M_g_l = mag_g - (dist_mod - dM_err[1]) # lower distance M_g
    
    M_r = mag_r - dist_mod
    M_r_u = mag_r - (dist_mod + dM_err[0]) # upper distance M_g
    M_r_l = mag_r - (dist_mod - dM_err[1])
    
    M_v = mag_v - dist_mod
    M_v_u = mag_v - (dist_mod + dM_err[0]) # upper distance M_g
    M_v_l = mag_v - (dist_mod - dM_err[1])
    
    if calc_L_bol == True:
        galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
        noise = np.full_like(galaxy, redshift)           # Assume constant noise per pixel here
    
        # Read the list of filenames from the Single Stellar Population library
        # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
        # of the library is included for this example with permission
        vazdekis = glob.glob('emiles/Ekb1.30*') # PUT HERE THE DIRECTORY TO EMILES_STARS
        FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
        velscale_ratio = 2  # adopts 2x higher spectral sampling for templates than for galaxy
        miles_path = path.expanduser("emiles/Ekb1.30Z*.fits")
        miles = lib.miles(miles_path, velscale, FWHM_tem)
        stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
        reg_dim = miles.templates.shape[1:]
        # Extract the wavelength range and logarithmically rebin one spectrum
        # to a velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
        # the size needed for the array which will contain the template spectra.
        #
        hdu = fits.open(vazdekis[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
        templates = np.empty((sspNew.size, len(vazdekis))) # PUT HERE THE DIRECTORY TO MILES_STARS
        
    
    #     # Extract the wavelength range and logarithmically rebin one spectrum
    #     # to a velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    #     # the size needed for the array which will contain the template spectra.
    #     #
    #     hdu = fits.open(vazdekis[0])
    #     ssp = hdu[0].data
    #     h2 = hdu[0].header
    #     lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    #     sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
    #     templates = np.empty((sspNew.size, len(vazdekis)))
    
        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the SAURON and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.
    
        # Quadratic sigma difference in pixels Vazdekis --> SAURON
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        #
        FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
    
        for j, vazd in enumerate(vazdekis):
            hdu = fits.open(vazd)
            ssp = hdu[0].data
            ssp = ndimage.gaussian_filter1d(ssp, sigma)
            sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
            templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates
    
        # The galaxy and the template spectra do not have the same starting wavelength.
        # For this reason an extra velocity shift DV has to be applied to the template
        # to fit the galaxy spectrum. We remove this artificial shift by using the
        # keyword VSYST in the call to PPXF below, so that all velocities are
        # measured with respect to DV. This assume the redshift is negligible.
        # In the case of a high-redshift galaxy one should de-redshift its
        # wavelength to the rest frame before using the line below (see above).
        #
        c = 299792.458
        #c = 299792458.0 # speed of light
        dv = (logLam2[0] - logLam1[0])*c  # km/s
        z = np.exp(vel/c) - 1   # Relation between velocity and redshift in pPXF
        
        cond = np.exp(logLam1) <= 7810#6900
        logLam1 = logLam1[cond]
        galaxy = galaxy[cond]; noise = noise[cond]
        goodPixels = util.determine_goodpixels(logLam1, lamRange2, z)
    
        # Here the actual fit starts. The best fit is plotted on the screen.
        # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
        #
        start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
        t = clock()
    
        galaxy[np.where(np.isfinite(galaxy)==False)]       = 0.0
        noise[np.where(np.isfinite(noise)==False)]         = 0.0
        templates[np.where(np.isfinite(templates)==False)] = 0.0
    
        pp = ppxf(templates, galaxy, noise*0.0+1.0, velscale, start,
                  goodpixels=goodPixels, plot=True, moments=4,
                  mdegree=15, vsyst=dv, velscale_ratio=velscale_ratio)
    
        weights = pp.weights
        normalized_weights = weights / np.sum( weights )
        
        # Use miles_utils to get metallicity
        weights = pp.weights
        weights = weights.reshape(reg_dim)/weights.sum()  # Normalized
        miles.mean_age_metal(weights)
        
        
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]
    
        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
    
        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
    
        # Pass the optimal template to get the bolometric correction for the g-band
        BC_g = transmission(np.exp(logLam2), optimal_template, band="g")
        BC_r = transmission(np.exp(logLam2), optimal_template, band="r")
        BC_v = transmission(np.exp(logLam2), optimal_template, band="V")
    
        # Obtaining the bolometric correction of the Sun
        BC_sun_g, M_sun_g = library(0,0,filt="SDSS", band="g",get_sun='Y')
        BC_sun_r, M_sun_r = library(0,0,filt="SDSS", band="r",get_sun='Y')
        BC_sun_v, M_sun_v = library(0,0,filt="GROUND_JOHNSON", band="V",get_sun='Y')
        
    
        # Getting the bolometric luminosity (in solar luminosity) for the g-band
        lum_bol_g = 10.0**(-0.4*(M_g-M_sun_g)) * 10.0**(-0.4*(BC_g-BC_sun_g))
        lum_bol_g_u = 10.0**(-0.4*(M_g_u-M_sun_g)) * 10.0**(-0.4*(BC_g-BC_sun_g)) # upper dM L
        lum_bol_g_l = 10.0**(-0.4*(M_g_l-M_sun_g)) * 10.0**(-0.4*(BC_g-BC_sun_g)) # lower dM L

        lum_bol_r = 10.0**(-0.4*(M_r-M_sun_r)) * 10.0**(-0.4*(BC_r-BC_sun_r))        
        lum_bol_r_u = 10.0**(-0.4*(M_r_u-M_sun_r)) * 10.0**(-0.4*(BC_r-BC_sun_r)) # upper dM L
        lum_bol_r_l = 10.0**(-0.4*(M_r_l-M_sun_r)) * 10.0**(-0.4*(BC_r-BC_sun_r)) # lower dM L

        lum_bol_v = 10.0**(-0.4*(M_v-M_sun_v)) * 10.0**(-0.4*(BC_v-BC_sun_v))
        lum_bol_v_u = 10.0**(-0.4*(M_v_u-M_sun_v)) * 10.0**(-0.4*(BC_v-BC_sun_v)) # upper dM L
        lum_bol_v_l = 10.0**(-0.4*(M_v_l-M_sun_v)) * 10.0**(-0.4*(BC_v-BC_sun_v)) # lower dM L

        
        return lum_bol_g, [lum_bol_g_u, lum_bol_g_l], lum_bol_r, [lum_bol_r_u, lum_bol_r_l], lum_bol_v, [lum_bol_v_u, lum_bol_v_l], mag_g, mag_r, mag_v
    
    else:
        
        return mag_g, mag_r, mag_v


# ====================================
#        GETTING THE MAGNITUDES
# ====================================
def library(lamb, spectra, filt, band='g', get_sun='N'):

    # Internal default library of passbands filters
    lib = pyphot.get_library()

    # Defining the filter band library
    f = lib[filt+'_'+band]

    if get_sun == 'Y':
        sun_obs  = pyphot.Sun(flavor='observed') # Getting the solar spectrum
        wave_sun = sun_obs.wavelength.magnitude
        spec_sun = sun_obs.flux.magnitude

        BC_sun = transmission(wave_sun, spec_sun, band)

        # Getting the Sun absolute magnitude
        fluxes = f.get_flux(wave_sun, spec_sun)

        # Convert to vega magnitudes
        mags  = -2.5 * np.log10(fluxes) - f.Vega_zero_mag
        dist  = 1.49597871e11/(3.0857e16) # Astronomical unit in parsec
        M_sun = mags - 5.0*np.log10(dist/10.0) # Absolute magnitude of the Sun

        return BC_sun, M_sun

    else:
        # Compute the integrated flux through the filter f
        # note that it work on many spectra at once
        fluxes = f.get_flux(lamb, spectra)

        # Convert to vega magnitudes
        mags = -2.5 * np.log10(fluxes) - f.Vega_zero_mag

        return mags, fluxes

# ====================================
#        TRANSMISSION FUNCTION
# ====================================
def transmission(lamb, spectra, band):

    """ This function convolves the transmission function
    of the g-band for the SLOAN filters with the provided
    spectra. """

    if (band =="r") | (band == "g"):
        file = np.loadtxt(f"config/Paranal_OmegaCAM.{band}_SDSS.dat") # PUT HERE THE PATH TO YOUR FILTER
    else:
        file = np.loadtxt(f"config/Paranal_OmegaCAM.{band}.dat")
    # Getting the response function for different wavelenghts
    l_band, response_band = file[:,0], file[:,1]
    

    # Apply an interpolation to get the transmission function
    aa = np.interp(lamb,l_band,response_band)
    #aa[np.where((lamb > max(l_rband)) | (lamb < min(l_rband)))] = 0
    
    spectra_band = aa*spectra # Apply the transmission filter to the spectra
    dl = lamb[1]-lamb[0] # Spectral resolution

    total_flux = np.sum(spectra)*dl; total_flux_band = np.sum(spectra_band)*dl

    BC = total_flux/total_flux_band # Computing the bolometric correction

    return -2.5*np.log10(BC)
