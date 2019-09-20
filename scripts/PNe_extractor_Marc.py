import numpy as np
from astropy.io import fits, ascii
import sys

# TO Make
# 3D minicubes of residuals
# 3D minicubes of models
# 3D minicubes of original MUSE data

# Include wavelength ranges for each.

galaxy_name = sys.argv[1]#input("Please choose a galaxy, in the form of FCC000: ")

def extract(x,y,n_pix, data, x_d):
    xc = round(x)
    yc = round(y)
    offset = n_pix // 2
    
    y_range = np.arange(yc - offset, (yc - offset)+n_pix, 1, dtype=int)
    x_range = np.arange(xc - offset, (xc - offset)+n_pix, 1, dtype=int)

    ind = [i * x_d +x_range for i in y_range]

    return data[np.ravel(ind)]


x_y_list = np.load("exported_data/"+galaxy_name+"/"+galaxy_name+"_PNe_x_y_list.npy")
x_PNe = np.array([x[0] for x in x_y_list])
y_PNe = np.array([y[1] for y in x_y_list])

n_pixels = 9
coordinates = [(n,m) for n in range(n_pixels) for m in range(n_pixels)]
x_fit = np.array([item[0] for item in coordinates])
y_fit = np.array([item[1] for item in coordinates])

# raw MUSE data minicubes
hdulist = fits.open("/local/tspriggs/Fornax_data_cubes/"+galaxy_name+"center.fits")

hdr = hdulist[1].header
s = hdulist[1].data.shape # (lambda, y, x)
full_wavelength = hdr['CRVAL3']+(np.arange(s[0])-hdr['CRPIX3'])*hdr['CD3_3']

cube_list = hdulist[1].data.reshape(s[0], s[1]*s[2]) # (lambda, list of len y*x)
cube_list = np.swapaxes(cube_list, 1,0) # (list of len x*y, lambda)

raw_minicubes = np.array([extract(x,y,n_pixels, cube_list, s[2]) for  x,y in zip(x_PNe, y_PNe)])

hdu_raw_minicubes = fits.PrimaryHDU(raw_minicubes,hdr)
hdu_long_wavelength = fits.ImageHDU(full_wavelength)

hdu_to_write = fits.HDUList([hdu_raw_minicubes, hdu_long_wavelength])

hdu_to_write.writeto("exported_data/"+galaxy_name+"/"+galaxy_name+"_MUSE_PNe.fits", overwrite=True)
print("Raw minicubes saved")
# 3D minicubes of residuals

hdulist = fits.open("galaxy_data/"+galaxy_name+"_data/"+galaxy_name+"_residuals_list.fits")

hdr = hdulist[0].data

wavelength = np.exp(hdulist[1].data)

residual_minicubes = np.array([extract(x,y,n_pixels,  hdulist[0].data, s[2]) for  x,y in zip(x_PNe, y_PNe)])

hdu_resid_minicubes = fits.PrimaryHDU(residual_minicubes)
hdu_resid_wavelength = fits.ImageHDU(wavelength)

hdu_to_write = fits.HDUList([hdu_resid_minicubes, hdu_resid_wavelength])

hdu_to_write.writeto("exported_data/"+galaxy_name+"/"+galaxy_name+"residual_PNe.fits", overwrite=True)
