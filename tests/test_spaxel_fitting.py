import pytest
import numpy as np
from astropy.io import fits
from functions.MUSE_Models import PNe_residuals_3D, PSF_residuals_3D, spaxel_by_spaxel
from functions.PNe_functions import robust_sigma, PNe_spectrum_extractor
from functions.file_handling import paths, open_data


def test_dir_dict():
    galaxy_name = "FCCtest"
    loc = "center"
    
    DIR_dict = paths(galaxy_name, loc)
    
    assert len(DIR_dict) > 0 # test to verify DIR_dict isn't empty
    

def test_open_data():
    galaxy_name = "FCCtest"
    loc = "center"
    DIR_dict = paths(galaxy_name, loc)
    res_data, wavelength, res_shape, x_data, y_data, galaxy_data = open_data(galaxy_name, loc, DIR_dict)
    
    assert len(np.shape(res_data)) == 2 # test data is in list format, i.e. 2 dimensions


def test_PNe_spectrum_extractor():
    galaxy_name = "FCCtest"
    loc = "center"
    
    DIR_dict = paths(galaxy_name, loc)
    
    # Load in the residual data, in list form
    res_data, wavelength, res_shape, x_data, y_data, galaxy_data = open_data(galaxy_name, loc, DIR_dict)
    
    x_y_list = np.load(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])
    
    sources = np.array([PNe_spectrum_extractor(x, y, 9, res_data, x_data, wavelength) for x,y in zip(x_PNe, y_PNe)])

    assert len(np.shape(sources)) == 3  # test sources is 3 dimensional array
    assert np.shape(sources)[0] == len(x_PNe) # test number of sources matches number of PNe in x_y_list
    

