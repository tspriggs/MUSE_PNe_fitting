import numpy as np
from functions.PNe_functions import PNe_minicube_extractor, generate_mask, D_to_dM, dM_to_D
from functions.file_handling import paths, open_data

def test_dir_dict():
    galaxy_name = "FCCtest"
    loc = "center"
    
    DIR_dict = paths(galaxy_name, loc)
    
    assert len(DIR_dict) > 0 # test that DIR_dict isn't empty
    #assert that all the values in the DIR_dict dictionary are string types
    assert [isinstance(entry, str) for entry in DIR_dict.values() ] == [True] * len(list(DIR_dict.values())) 
    

def test_open_data():
    galaxy_name = "FCCtest"
    loc = "center"
    DIR_dict = paths(galaxy_name, loc)
    res_data, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_data = open_data(galaxy_name, loc, DIR_dict)
    
    assert len(np.shape(res_data)) == 3 # test data is in list format, i.e. 2 dimensions


def test_PNe_minicube_extractor():
    galaxy_name = "FCCtest"
    loc = "center"
    
    DIR_dict = paths(galaxy_name, loc)
    
    # Load in the residual data, in list form
    res_data, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_data = open_data(galaxy_name, loc, DIR_dict)

    x_y_list = np.load(DIR_dict["EXPORT_DIR"]+"_PNe_x_y_list.npy")
    x_PNe = np.array([x[0] for x in x_y_list])
    y_PNe = np.array([y[1] for y in x_y_list])
    
    sources = np.array([PNe_minicube_extractor(x, y, 9, res_data, wavelength) for x,y in zip(x_PNe, y_PNe)])

    assert len(np.shape(sources)) == 3  # test sources is 3 dimensional array
    assert np.shape(sources)[0] == len(x_PNe) # test number of sources matches number of PNe in x_y_list
    

def test_generate_mask():
    galaxy_name = "FCCtest"
    loc = "center"
    DIR_dict = paths(galaxy_name, loc)
    res_cube, res_hdr, wavelength, res_shape, x_data, y_data, galaxy_info = open_data(galaxy_name, loc, DIR_dict)

    ellip_mask = generate_mask(img_shape=[y_data, x_data], mask_params=galaxy_info["gal_mask"], mask_shape="ellipse")
    circle_mask = [generate_mask(img_shape=[y_data, x_data], mask_params=star_mask, mask_shape="circle") for star_mask in galaxy_info["star_mask"]]

    assert np.shape(ellip_mask) == (y_data, x_data) # Normally only one mask for each galaxy.
    assert np.shape(circle_mask) == (len(galaxy_info["star_mask"]), y_data, x_data) # There can be more than one mask for stars in a FOV.

def test_D_to_dM():
    test_distance = 20.0 # Mpc

    assert round(D_to_dM(test_distance), 3) == 31.505

def test_dM_to_D():
    test_dM = 31.505 # mag

    assert round(dM_to_D(test_dM), 3) == 19.999