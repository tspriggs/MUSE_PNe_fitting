import pytest
import numpy as np
from astropy.io import fits
from functions.MUSE_Models import PNe_residuals_3D, PSF_residuals_3D, spaxel_by_spaxel
from functions.PNe_functions import open_data, robust_sigma, PNe_spectrum_extractor


def test_open_data():
    galaxy_name = "FCCtest"
    loc = "center"
    res_data, wavelength, res_shape, x_data, y_data, galaxy_data = open_data(galaxy_name, loc)
    
    assert len(np.shape(res_data)) == 2


def test_PNe_spectrum_extractor():
    galaxy_name = "FCCtest"
    loc="center"
    EXPORT_DIR = f"exported_data/{galaxy_name}/{galaxy_name}{loc}"
    DATA_DIR = f"galaxy_data/{galaxy_name}_data/{galaxy_name}{loc}"
    
    # Load in the residual data, in list form
    with fits.open(DATA_DIR+"_residuals_list.fits") as hdulist:# Path to data
        res_data = np.copy(hdulist[0].data)
        res_hdr = hdulist[0].header # extract header from residual cube
        wavelength = np.exp(hdulist[1].data)
    
    x_y_list = np.load(EXPORT_DIR+"_PNe_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])
    
    
    sources = np.array([PNe_spectrum_extractor(x, y, 9, res_data, res_hdr["XAXIS"], wavelength) for x,y in zip(x_PNe, y_PNe)])

    assert len(np.shape(sources)) == 3
    assert np.shape(sources)[0] == len(x_PNe)
    
