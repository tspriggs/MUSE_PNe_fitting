galaxy_name = "FCC143"
hdu = fits.open("/local/tspriggs/Fornax_data_cubes/FCC143center.fits")
s = np.shape(hdu[1].data)
#flat_cube = np.sum(hdu[1].data, 0)
plt.imshow(np.log10(flat_cube), origin="lower", vmin=2, vmax=6)
gal_centre_pix = Table.read("exported_data/galaxy_centre_pix.dat", format="ascii")
gal_x = gal_centre_pix[1]["x_pix"]# 1 for FCC143
gal_y = gal_centre_pix[1]["y_pix"]

plt.axhline(gal_y)
plt.axvline(gal_x)

Y, X = np.mgrid[:s[1], :s[2]]
xe, ye, length, width, alpha = [gal_x, gal_y, 350, 250, 2.1]

elip_mask = (((X-xe) * np.cos(alpha) + (Y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((X-xe) * np.sin(alpha) - (Y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 1 


star_mask_params = [[0,0,0]]
star_mask_sum = np.sum([(Y - yc)**2 + (X - xc)**2 <= rc**2 for xc,yc,rc in star_mask_params],0).astype(bool)
    
    
total_mask = ((np.isnan(hdu[1].data[1,:,:])==False) & (elip_mask==False) & (star_mask_sum==False))
indx_mask = np.where(total_mask==True)

plt.imshow(np.log10(flat_cube), origin="lower")
plt.imshow(total_mask,origin="lower", alpha=0.4)




sum_mask = (elip_mask==True)&(np.isnan(hdu[1].data[1,:,:])==False)
plt.imshow(sum_mask, origin="lower")

indx_mask = np.where(sum_mask==True)




raw_data_cube = np.copy(hdu[1].data)
h1 = hdu[1].header
good_spectra = np.zeros((s[0], len(indx_mask[0])))

for i, (y, x)  in enumerate(zip(tqdm(indx_mask[0]), indx_mask[1])):
    good_spectra[:,i] = raw_data_cube[:,y,x]

print("Collapsing cube now....")    
    
gal_lin = np.nansum(good_spectra, 1)
        
print("Cube has been collapsed...")
dM = 31.220

c = 299792458.0 
gal_vel = 1379
z = gal_vel*1e3 / c

L_bol = ppxf_L_tot(int_spec=gal_lin, header=h1, redshift=z, vel=gal_vel, dist_mod=dM)


#FCC255 xe, ye, length, width, alpha = [gal_x, gal_y, 400, 200, -0.1]
#FCC143 xe, ye, length, width, alpha = [gal_x, gal_y, 350, 250, 2.1]
