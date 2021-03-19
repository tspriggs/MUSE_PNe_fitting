from astropy.io import fits
import numpy as np
import yaml

from functions.PNe_functions import PNe_minicube_extractor


def paths(galaxy_name, loc):
    """Read from "dir_config.yaml" file for the pathnames for your data directory.

    Parameters
    ----------
    galaxy_name : str
        Name of galaxy (FCC000 for example).

    loc : str
        Location of observation (center, middle etc.). Leave empty if not required.

    Returns
    -------
    Dict
        RAW_DIR    - directory of raw data files.
        RAW_DATA   - directory of raw data fits file.
        DATA_DIR   - directory of galaxy data, local to scripts.
        EXPORT_DIR - directory of exported files from script output.
        PLOT_DIR   - directory for plots to be saved in.
        YAML       - Path to yaml galaxy info file
    """
    
    with open("config/dir_config.yaml", "r") as yaml_dir_file:
        yaml_dir = yaml.load(yaml_dir_file, Loader=yaml.FullLoader)
    
    for item in yaml_dir:
        yaml_dir[f"{item}"] =  yaml_dir[f"{item}"].format(galaxy=galaxy_name, loc=loc)

    return yaml_dir


def open_data(galaxy_name, loc, DIR_dict):
    """Load up both the yaml and residual data associated with input galaxy and location

    Parameters
    ----------
    galaxy_name : str
        name of galaxy (FCC000 for example).

    loc : str
        location of observation (center, middle etc.). Leave empty if not required.

    DIR_dict : Dict
        Dictionary containing the directories of key files and folers. Generated from 'paths' function.

    Returns
    -------
    list
        residual_data, wavelength, residual_shape, galaxy_info, x_data, y_data
    """

    # Load in the residual data, in list form
    
    # Open the yaml config file for galaxy_info
    with open(DIR_dict["YAML"], "r") as yaml_file:
        yaml_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
    galaxy_info = yaml_info[f"{galaxy_name}_{loc}"]

    # Open the residual list fits file for the selected galaxy and location
    with fits.open(DIR_dict["DATA_DIR"]+"_residual_cube.fits") as hdulist:# Path to data
        residual_data = np.copy(hdulist[1].data)
        residual_hdr = hdulist[1].header # extract raw data header info, including wcs, from residual cube
        wavelength = np.copy(hdulist[2].data)
    
    # store the shape of the data, should be two dimensional
    residual_shape = np.shape(residual_data)
    
    x_data = residual_hdr["NAXIS1"]
    y_data = residual_hdr["NAXIS2"]
    
    return residual_data, residual_hdr, wavelength, residual_shape, x_data, y_data, galaxy_info

def open_PNe(galaxy_name, loc, DIR_dict):
    """Load up both the yaml and residual data associated with input galaxy and location

    Parameters
    ----------
    galaxy_name : str
        name of galaxy (FCC000 for example).

    loc : str
        location of observation (center, middle etc.). Leave empty if not required.

    DIR_dict : Dict
        Dictionary containing the directories of key files and folers. Generated from 'paths' function.

    Returns
    -------
    list
        PNe_spectra, hdr, wavelength, obj_err, res_err , residual_shape, x_data, y_data, galaxy_info
    """
        
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


def prep_impostor_files(galaxy_name, DIR_dict, PNe_spectra, model_spectra_list, short_wave, n_pixels, x_PNe, y_PNe):
    """[summary]

    Parameters
    ----------
    galaxy_name : [str]
        Name of galaxy

    DIR_dict : [dict]
        Dictionary of useful directories for file opening and saving.

    PNe_spectra : [array]
        list of PNe residual spectra (containing [OIII] emissions).

    model_spectra_list : [list]
        list of best fit models to the PNe (Gaussian models).

    short_wave : [array]
        wavelength array.

    n_pixels : [int]
        number of pixels on a side (n_pixel by n_pixel).

    x_PNe : [list]
        list of x coordinate for the PNe.

    y_PNe : [list]
        list of y coordinate for the PNe.


    Returns
    -------
    [None]
        No return, just saves the files and prints out the locations of the files.
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
    
    with fits.open(DIR_dict["RAW_DATA"], memmap=True) as raw_hdulist:
        raw_data = raw_hdulist[1].data
        raw_hdr = raw_hdulist[1].header
        raw_shape = raw_hdulist[1].data.shape # (lambda, y, x)
        full_wavelength = raw_hdr['CRVAL3']+(np.arange(raw_shape[0])-raw_hdr['CRPIX3'])*raw_hdr['CD3_3']

        if len(raw_hdulist) == 3:
            stat_list = raw_hdulist[2].data
        elif len(raw_hdulist) == 2:
            stat_list = np.ones_like(raw_data)

    
    raw_minicubes = np.array([PNe_minicube_extractor(x, y, n_pixels, raw_data, full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    stat_minicubes = np.array([PNe_minicube_extractor(x, y, n_pixels, stat_list, full_wavelength) for  x,y in zip(x_PNe, y_PNe)])
    
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
    n_PNe = len(PNe_spectra)
    weighted_PNe = np.ones((n_PNe, n_pixels**2, len(full_wavelength)))  #N_PNe, spaxels, wavelength length
    
    for p in np.arange(0, n_PNe):
        weighted_PNe[p] = PSF_weight(raw_minicubes[p], model_spectra_list[p], short_wave, n_pixels**2)
    
    sum_weighted_PNe = np.nansum(weighted_PNe, 1)
    
    hdu_weighted_minicubes = fits.PrimaryHDU(sum_weighted_PNe, raw_hdr)
    hdu_weighted_stat = fits.ImageHDU(np.nansum(stat_minicubes,1))
    
    weight_hdu_to_write = fits.HDUList([hdu_weighted_minicubes, hdu_stat_minicubes, hdu_long_wavelength])
    
    # weight_hdu_to_write.writeto(f"../../gist_PNe/inputData/{galaxy_name}MUSEPNeweighted.fits", overwrite=True)
    weight_hdu_to_write.writeto(DIR_dict["EXPORT_DIR"]+"MUSEPNeweighted.fits", overwrite=True)
    print(f"{galaxy_name}_MUSE_PNe_weighted.fits file saved. Ready for GIST to run.")
