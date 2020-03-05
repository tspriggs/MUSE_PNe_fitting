from astropy.io import fits
import numpy as np
import yaml

from functions.PNe_functions import PNe_spectrum_extractor


def paths(galaxy_name, loc):
    """
    Change the paths in this function to reflect where your data is stored.
    
    Parameters:
        - galaxy_name - string - name of galaxy (FCC000 for example).
        - loc         - string - location of observation (center, middle etc.). Leave empty if not required.
        
    Returns a dictionary containing:
        - RAW_DIR    - directory of raw data files.
        - RAW_DATA   - directory of raw data fits file.
        - DATA_DIR   - directory of galaxy data, local to scripts.
        - EXPORT_DIR - directory of exported files from script output.
        - PLOT_DIR   - directory for plots to be saved in.
        - YAML       - Path to yaml galaxy info file
        
    """
    DIR_dict = {
        "RAW_DIR"    : f"/local/tspriggs/Fornax_data_cubes/{galaxy_name}",
        "RAW_DATA"   : f"/local/tspriggs/Fornax_data_cubes/{galaxy_name}{loc}.fits",
        "DATA_DIR"   : f"galaxy_data/{galaxy_name}_data/{galaxy_name}{loc}",
        "EXPORT_DIR" : f"exported_data/{galaxy_name}/{galaxy_name}{loc}",
        "PLOT_DIR"   : f"Plots/{galaxy_name}/{galaxy_name}{loc}",
        "YAML"       : "config/galaxy_info.yaml"
    }
    
    return DIR_dict


def open_data(galaxy_name, loc, DIR_dict):
    """
    Load up both the yaml and residual data associated with input galaxy and location
    Parameters:
        - galaxy - string - name of galaxy - FCC000
        - loc    - string - location       - center, middle or halo
        - DIR_dict - dict - dict object containing all directories
    
    Returns:
        - residual_data
        - wavelength
        - residual_shape
        - galaxy_info
        - x_data
        - y_data
    """
    # Load in the residual data, in list form
    
    # Open the yaml config file for galaxy_info
    with open(DIR_dict["YAML"], "r") as yaml_data:
        yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
    galaxy_info = yaml_info[f"{galaxy_name}_{loc}"]

    # Open the residual list fits file for the selected galaxy and location
    with fits.open(DIR_dict["DATA_DIR"]+"_residuals_list.fits") as hdulist:# Path to data
        residual_data = np.copy(hdulist[0].data)
        residual_hdr = hdulist[0].header # extract header from residual cube
        wavelength = np.exp(hdulist[1].data)
    
    # store the shape of the data, should be two dimensional
    residual_shape = np.shape(residual_data)
    
    x_data = residual_hdr["XAXIS"]
    y_data = residual_hdr["YAXIS"]
    
    return residual_data, wavelength, residual_shape, x_data, y_data, galaxy_info

def open_PNe(galaxy_name, loc, DIR_dict):
    """
    Load up both the yaml and residual data associated with input galaxy and location
    Parameters:
        - galaxy - string - name of galaxy - FCC000
        - loc    - string - location       - center, middle or halo
        - DIR_dict - dict - dict object containing all directories
    
    Returns:
        - PNe_spectra
        - hdr
        - wavelength
        - obj_err
        - res_err        
        - residual_shape
        - x_data
        - y_data
        - galaxy_info
    """
    # Load in the residual data, in list form
    
    # Open the yaml config file for galaxy_info
    with open(DIR_dict["YAML"], "r") as yaml_data:
        yaml_info = yaml.load(yaml_data, Loader=yaml.FullLoader)
        
    galaxy_info = yaml_info[f"{galaxy_name}_{loc}"]

    # Open the residual list fits file for the selected galaxy and location
    with fits.open(DIR_dict["DATA_DIR"]+"_PNe_spectra.fits") as hdulist: # Path to data
        PNe_spectra = np.copy(hdulist[1].data)
        hdr = hdulist[1].header # extract header from residual cube
        wavelength = np.copy(hdulist[2].data)
        obj_err = np.copy(hdulist[3].data)
        res_err = np.copy(hdulist[4].data)
    
    # store the shape of the data, should be three dimensional: N_PNe, n_pixels**2, len(wavelength)
    residual_shape = np.shape(PNe_spectra)
    
    x_data = hdr["XAXIS"]
    y_data = hdr["YAXIS"]
    
    return PNe_spectra, hdr, wavelength, obj_err, res_err, residual_shape, x_data, y_data, galaxy_info


def reconstructed_image(galaxy_name, loc):
    DIR_dict = paths(galaxy_name, loc)
    
    with fits.open(DIR_dict["RAW_DATA"]) as hdu:
        hdr  = hdu[1].header
        s    = np.shape(hdu[1].data)
        wave = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']   
        cond = (wave >= 4900.0) & (wave <= 5100.0)
        data = np.sum(hdu[1].data[cond,:,:],axis=0)

    return data, wave, hdr


def prep_impostor_files(galaxy_name, DIR_dict, galaxy_info, PNe_spectra, model_spectra_list, short_wave, n_PNe, n_pixels, x_PNe, y_PNe):
    """
    Prepare files for impostor checks, no return, only saved files.
    
    Parameters:
        - galaxy_name - string
        - galaxy_info - dict of configs, used for PSF params (M_FWHM, beta, etc.)
    """
    
    ############# WEIGHTED MUSE data PNe ##############
    def PSF_weight(MUSE_p, model_p, r_wls, spaxels=81):
           
        coeff = np.polyfit(r_wls, np.clip(model_p[0, :], -50, 50), 1) # get continuum on first spaxel, assume the same across the minicube
        poly = np.poly1d(coeff)
        tmp = np.copy(model_p)
        for k in np.arange(0,spaxels):
             tmp[k,:] = poly(r_wls)
                
        res_minicube_model_no_continuum = model_p - tmp # remove continuum
        
        # PSF weighted minicube
        sum_model_no_continuum = np.nansum(res_minicube_model_no_continuum, 0)
        weights = np.nansum(res_minicube_model_no_continuum, 1)
        nweights = weights / np.nansum(weights) # spaxel weights
        weighted_spec = np.dot(nweights, MUSE_p) # dot product of the nweights and spectra
    
        return weighted_spec
    
    with fits.open(DIR_dict["RAW_DATA"]) as raw_hdulist:
        raw_data = raw_hdulist[1].data
        raw_hdr = raw_hdulist[1].header
        raw_shape = raw_hdulist[1].data.shape # (lambda, y, x)
        full_wavelength = raw_hdr['CRVAL3']+(np.arange(raw_shape[0])-raw_hdr['CRPIX3'])*raw_hdr['CD3_3']

        cube_list = np.copy(raw_data).reshape(raw_shape[0], raw_shape[1]*raw_shape[2]) # (lambda, list of len y*x)
        cube_list = np.swapaxes(cube_list, 1,0) # (list of len x*y, lambda)

        if len(raw_hdulist) == 3:
            stat_list = np.copy(raw_hdulist[2].data).reshape(raw_shape[0], raw_shape[1]*raw_shape[2])
            stat_list = np.swapaxes(stat_list, 1,0)
        elif len(raw_hdulist) == 2:
            stat_list = np.ones_like(cube_list)

    
    raw_minicubes = np.array([PNe_spectrum_extractor(x, y, n_pixels, cube_list, raw_shape[2], full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    # stat_minicubes = np.ones_like(raw_minicubes)
    stat_minicubes = np.array([PNe_spectrum_extractor(x, y, n_pixels, stat_list, raw_shape[2], full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    
    sum_raw  = np.nansum(raw_minicubes, 1)
    sum_stat = np.nansum(stat_minicubes, 1)
    
    hdu_raw_minicubes = fits.PrimaryHDU(sum_raw,raw_hdr)
    hdu_stat_minicubes = fits.ImageHDU(sum_stat)
    hdu_long_wavelength = fits.ImageHDU(full_wavelength)
    
    raw_hdu_to_write = fits.HDUList([hdu_raw_minicubes, hdu_stat_minicubes, hdu_long_wavelength])
    
    raw_hdu_to_write.writeto(DIR_dict["EXPORT_DIR"]+"_MUSE_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_MUSE_PNe.fits file saved.")
    
    
    ##### Residual .fits file ################
    residual_hdu = fits.PrimaryHDU(PNe_spectra)
    wavelenth_residual = fits.ImageHDU(short_wave)
    resid_hdu_to_write = fits.HDUList([residual_hdu, wavelenth_residual])
    resid_hdu_to_write.writeto(DIR_dict["EXPORT_DIR"]+"_residuals_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_residuals_PNe.fits file saved.")
    
    
    ####### 3D model .fits file ##################
    models_hdu = fits.PrimaryHDU(model_spectra_list)
    wavelenth_models = fits.ImageHDU(short_wave)
    model_hdu_to_write = fits.HDUList([models_hdu, wavelenth_models])
    model_hdu_to_write.writeto(DIR_dict["EXPORT_DIR"]+"_3D_models_PNe.fits", overwrite=True)
    print(f"{galaxy_name}_residuals_PNe.fits file saved.")
    
    weighted_PNe = np.ones((n_PNe, n_pixels**2, len(full_wavelength)))  #N_PNe, spaxels, wavelength length
    
    for p in np.arange(0, n_PNe):
        weighted_PNe[p] = PSF_weight(raw_minicubes[p], model_spectra_list[p], short_wave, n_pixels**2)
    
    sum_weighted_PNe = np.nansum(weighted_PNe, 1)
    
    hdu_weighted_minicubes = fits.PrimaryHDU(sum_weighted_PNe, raw_hdr)
    hdu_weighted_stat = fits.ImageHDU(np.nansum(stat_minicubes,1))
    
    weight_hdu_to_write = fits.HDUList([hdu_weighted_minicubes, hdu_stat_minicubes, hdu_long_wavelength])
    
    weight_hdu_to_write.writeto(f"../../gist_PNe/inputData/{galaxy_name}MUSEPNeweighted.fits", overwrite=True)
    print(f"{galaxy_name}_MUSE_PNe_weighted.fits file saved. Ready for GIST to run.")
