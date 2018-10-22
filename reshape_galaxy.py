import numpy as np
from astropy.io import fits
import yaml

with open("galaxy_info.yaml", "r") as yaml_data:
    galaxy_info = yaml.load(yaml_data)
    
choose_galaxy = input("Please type which Galaxy you want to analyse, use FCC000 format: ")
galaxy_data = galaxy_info[choose_galaxy]

# Read in Raw data cube for axis control
raw_hdu = fits.open(galaxy_data["Galaxy name"] + "_data/" + galaxy_data["Galaxy name"] + "_DATACUBE_center.fits")
raw_hdr = raw_hdu[1].header
raw_shape = np.shape(raw_hdu[1].data)
xaxis = np.arange(raw_shape[2])*raw_hdr['CD2_2']*3600.0
yaxis = np.arange(raw_shape[1])*raw_hdr['CD2_2']*3600.0

# Read in table fits file for x,y (in respect to above xaxis and yaxis numbers) values
table_hdu = fits.open(galaxy_data["Galaxy name"] + "_data/" + galaxy_data["Galaxy name"] + "_table.fits")
table_data = table_hdu[1].data

# store as independant x and y variables
x_pix = table_data["X"]
y_pix = table_data["Y"]

# check where the index location of the x,y coordinates of the fitted pixels are, relative to the xaxis and yaxis coordinate systems
index_pix = np.zeros((len(x_pix),2))

for n,(i,j) in enumerate(zip(x_pix, y_pix)):
    index_pix[n] = [np.squeeze(np.where(xaxis == i)), np.squeeze(np.where(yaxis==j))]

clean_hdulist = fits.open(galaxy_data["clean cube"])
wavelength = np.exp(clean_hdulist[2].data.LOGLAM)

# read in the residual cube from gandalf and order the data as (list of spectra, wavelength range length)
res_hdu = fits.open(galaxy_data["Galaxy name"] + "_data/" + galaxy_data["Galaxy name"] + "_gandalf-residuals_SPAXEL.fits")
res_data_list = res_hdu[1].data.RESIDUALS
res_data_list = np.swapaxes(res_data_list, 1, 0)

# Make an empty pointing of the same shape as the input raw data cube: y, x, lambda
empty_gal = np.zeros((raw_shape[1], raw_shape[2], len(wavelength)))

# Fill in the empty pointing, using the x,y index values of the fitted pixel locations.
for n, i in enumerate(index_pix):
    empty_gal[int(i[1])][int(i[0])] = res_data_list[n]

reshaped_gal = np.transpose(empty_gal, (2,0,1)) 

# Save galaxy
write_hdu = fits.PrimaryHDU()
write_hdu.data = reshaped_gal
write_hdu.header['CRVAL3'] = min(wavelength)
write_hdu.header['CRPIX3'] = 1.0
write_hdu.header['CDELT3'] = wavelength[1]-wavelength[0]
write_hdu.writeto(galaxy_data["Galaxy name"]+"_data/" + galaxy_data["Galaxy name"] + "_residual_cube.fits", overwrite=True)


