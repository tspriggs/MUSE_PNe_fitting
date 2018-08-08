# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:17:50 2017

@author: TSpriggs
"""
#%% Gaussian 2D fitting routine.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.modeling import fitting, Fittable2DModel, Parameter
from astropy.table import Table
import pandas as pd
from MUSE_Models import Moffat2D_OIII
#%% Functions and fits

# Function to determine bounding limits to make an array of interest.

def AOI(x, y, size):
    x = round(x)
    y = round(y)
    far_left = x - size
    far_right = x + size
    top = y - size
    bottom = y + size
    return int(far_left), int(far_right), int(top), int(bottom)

# Function returning an array containing the area of interest.

def AOI_Array_selection(array, far_left, far_right, top, bottom):
    return array[top:bottom, far_left:far_right]

# Function to flatten an array.

flatten = lambda l: [item for sublist in l for item in sublist]

# Fitter

fitter = fitting.LevMarLSQFitter()

# Function for second fitter preparation phase.

def second_third_prep(array, fitted_model):
    res_flat = flatten(np.abs(array - fitted_model))
    rN = np.std(res_flat)
    array_flat = flatten(array)
    weights_flat = []
    outlier_marking = []
    
    for item, i in zip(res_flat, array_flat):
        if (np.abs(item) > 3*rN) or (i == 0):
            weights_flat.append(1e-6)
        else:
            weights_flat.append(1)

    for item, i in zip(res_flat, array_flat):
        if item > 3*rN:
            outlier_marking.append(i)
        else:
            outlier_marking.append(None)

    return weights_flat, outlier_marking, rN


def chi_square_test(residuals, error):
    chi_square = np.sum((residuals / error)**2) # Chi-square statistic test
    ndof = len(residuals) - 4. # Number of degrees of freedom, 4 being the number of free parameters used for fitting.
    chi_2_r = chi_square / ndof # Reduced Chi Square Statistic, used for goodness of fit evaluation.
    
    return chi_square, chi_2_r


def rounding_func(data):
    return round(data,2)

# Function to perfom fits
def fitting_process(list_of_x_y):
   
    M_5007_list = []
    m_5007_list = []
    f_5007_list =[]
    fit_A = []
    fit_rN = []
    A_err = []
    fit_chi_sq = []
    fit_chi_sq_r = []
    PNe_vel = []
    run_number = 1
    
    for item in x_y_list:
        AOI_coords = AOI(item[0],item[1], 8)
        
        # Select Flux Area Of Interest
        AOI_array = AOI_Array_selection(Gauss_F_shape, AOI_coords[0], AOI_coords[1], AOI_coords[2], AOI_coords[3]) # Return Array containing Area of Interest
        AOI_array_flat = flatten(AOI_array)
        
        # Select mean wavelenght Area Of Interest
        AOI_mean = AOI_Array_selection(z_vel, AOI_coords[0], AOI_coords[1], AOI_coords[2], AOI_coords[3])
        flat_AOI_mean = flatten(AOI_mean)
        
        # First set of weights created from F_weights using AOI coordinates
        AOI_weights = AOI_Array_selection(F_weights, AOI_coords[0], AOI_coords[1], AOI_coords[2], AOI_coords[3])
        AOI_weights_flat = flatten(AOI_weights)
        
        # 1st fit
        m_init = Moffat2D_OIII(amplitude=np.max(AOI_array), x_0=8., y_0=8., bkg=1., gamma=4.225, alpha=1.423, fixed={"gamma":True, "alpha":True}) # 7.16, 8.68,,,, 4.78,2.91
        m_fit = fitter(m_init, x_fit, y_fit, AOI_array_flat, maxiter=10000000, weights=AOI_weights_flat)
        # Include a check for it fitted amplitude is 3*rN, else assign Null values

        # 2nd fit
        outlier_removal_1 = second_third_prep(np.array(AOI_array), np.array(m_fit(X_AOI,Y_AOI)))
        m_fit_2nd = fitter(m_init, x_fit, y_fit, AOI_array_flat, weights=outlier_removal_1[0], maxiter=100000000)
        
        # 3rd fit
        outlier_removal_2 = second_third_prep(np.array(AOI_array), np.array(m_fit_2nd(X_AOI,Y_AOI)))
        m_fit_3rd = fitter(m_init, x_fit, y_fit, AOI_array_flat, weights=outlier_removal_2[0], maxiter=100000000)
        
        fit_x = round(m_fit_3rd.x_0.value)
        fit_y = round(m_fit_3rd.y_0.value)
#        vel_finder = int((fit_y * len(AOI_mean)) + fit_x )      
#        
#        PNe_vel.append(flat_AOI_mean[vel_finder])
        
        fit_A.append(m_fit_3rd.amplitude.value)
        fit_rN.append(outlier_removal_2[2])
         
        A_err.append(np.sqrt(fitter.fit_info["cov_x"][0][0]))
        
        model_minus_bkg = (m_fit_3rd(X_AOI, Y_AOI) - np.abs(m_fit_3rd.bkg.value)) # fit minus background
        f_5007 = np.sum(model_minus_bkg) * 1e-20 # total flux of PNe
        #f_5007_err = (np.sum(model_mins_bkg) * (A_err / fit_A)) * 1e-20   # Error on the flux values.
        m_5007 = -2.5 * np.log10(f_5007) - 13.74 # apparent magnitude
        dM = 5 * np.log10(14.5) + 25 # Distance Modulus assuming 14.5 MPc
        M_5007 = m_5007 - dM # Absolute Magnitude
        M_5007_list.append(M_5007)
        m_5007_list.append(m_5007)
        f_5007_list.append(f_5007)
                
        AOI_F_err = AOI_Array_selection(F_err_shape, AOI_coords[0], AOI_coords[1], AOI_coords[2], AOI_coords[3])
        AOI_F_err_flat = flatten(AOI_F_err)
        
        AOI_array_flat[AOI_array_flat == np.nan] = 0.0
        
        Chi_results = chi_square_test(np.array(fitter.fit_info["fvec"]), np.array(AOI_F_err_flat))
        fit_chi_sq.append(Chi_results[0])
        fit_chi_sq_r.append(Chi_results[1])
        
        x_c_m_3rd = m_fit_3rd.x_0.value
        y_c_m_3rd = m_fit_3rd.y_0.value
        r_moff_3rd = np.sqrt((x_fit - x_c_m_3rd)**2 + (y_fit - y_c_m_3rd)**2)

        wspace=0.5
        hspace=0.3
        # 3rd fit Plotting code
#        plt.figure("Detailed Analysis of third fit" , figsize=(16,4))
#        plt.clf()
        
        #Data plot
#        plt.subplot(1,3,1)
#        plt.subplots_adjust(wspace=wspace, hspace=hspace)
#        plt.imshow(AOI_array, origin="lower", cmap="CMRmap", vmin=0, vmax=100)
#        cb = plt.colorbar(fraction=0.05, pad=0.1)
#        cb.set_label("$erg  s^-1  cm^-1$")
#        plt.xlabel("x (pixels)")
#        plt.ylabel("y (pixels)")
#        plt.title("Flux data")
#               
#        # Redshift velocity plot.
#        plt.subplot(1,3,2)
#        plt.hist(AOI_mean, bins=2)
#        plt.xlabel("x (pixels)")
#        plt.ylabel("y (pixels)")
#        plt.title("Redshifted velocity.")
        
        
        # Redshift velocity plot.
#        plt.subplot(1,3,3)
#        plt.imshow(AOI_mean, origin="lower", cmap="RdBu", vmin=-300, vmax=300)
#        cb = plt.colorbar(fraction=0.05, pad=0.1)
#        cb.set_label("velocity $km s^{-1}$")
#        plt.xlabel("x (pixels)")
#        plt.ylabel("y (pixels)")
#        plt.title("Redshifted velocity.")
        
        
#        # Model plot
#        plt.subplot(1,3,2)
#        plt.subplots_adjust(wspace=wspace, hspace=hspace)
#        plt.imshow(m_fit_3rd(X_AOI,Y_AOI), origin="lower", cmap="CMRmap", vmin=0, vmax=100)
#        cb = plt.colorbar(fraction=0.05, pad=0.1)
#        cb.set_label("$erg  s^-1  cm^-1$")
#        plt.xlabel("x (pixels)")
#        plt.ylabel("y (pixels)")
#        plt.title("Moffat Model")        
        plt.figure(run_number)
        plt.clf()
        plt.scatter(r_moff_3rd, AOI_array_flat, marker=".", label="Data")
        plt.scatter(r_moff_3rd, outlier_removal_2[1], marker="x", color="k", label="Outlier")
        plt.scatter(r_moff_3rd, m_fit_3rd(x_fit,y_fit), marker="x", label="Moffat", c="r")
        plt.axvline(x=3, c='k', ls="dashed")
        plt.axhline(y=m_fit_3rd.bkg.value, c='k', ls="dashed")
        plt.xlabel("r")
        plt.ylabel("Flux")
        plt.legend(loc="upper right")
        plt.ylim(0,200)
        
        
        
#        plt.savefig("Plots/Detailed plot analysis of run %s.png" % run_number)
        
        run_number +=1
        
    fit_A_by_rN = [A / rN for A, rN in zip(fit_A, fit_rN)]
        
    return fit_A, fit_rN, fit_A_by_rN, f_5007_list, m_5007_list, M_5007_list, fit_chi_sq, fit_chi_sq_r, m_fit_3rd


#%% Read in and prepare data from 1d Gaussian fit.
    
Gaussian1D_data = ascii.read("exported_data/Gaussian1D_data.txt")

F_err = np.array(Gaussian1D_data["Flux error"])
F_err_shape = F_err.reshape((318,315))
F_weights = np.array(Gaussian1D_data["Flux weights"]).reshape((318,315))

# Prepare data for plotting and 2D fitting
Gauss_F_shape = np.array(Gaussian1D_data["Gaussian Fluxes"]).reshape((318,315))
A_by_rN_shape = np.array(Gaussian1D_data["A/rN"]).reshape((318,315))
Gauss_mean_shape = np.array(Gaussian1D_data["mean wavelength"]).reshape((318,315))

# v = (c * (obs - emit)/emit) / 1000   redshift velocity of sources in km/s
c = 299792458
z_vel = (c * (Gauss_mean_shape - 5006.9) / 5006.9) / 1000

Moffat2D_data = ascii.read("exported_data/Fit results table.txt")

# Read in x and y coordinates, in reagrds to 318x315 map.
x_PNe = np.array(Moffat2D_data["x"])
y_PNe = np.array(Moffat2D_data["y"])
x_y_list = [[i,j] for i,j in zip(x_PNe, y_PNe)]
#%%
A_rN = np.load("exported_data/M87/A_rN.npy")
#%% Plot A / rN to select potential PNe

fig = plt.figure("PNe selection plot",figsize=(16,14))

ax = plt.gca()
plt.clf()
plt.imshow(A_rN.reshape(318, 315), interpolation="spline16",origin="lower", cmap="CMRmap", vmin=1.6, vmax=6)
cb = plt.colorbar()
cb.set_label("A/rN", fontsize=22)
plt.xlabel("x (pixels)", fontsize=22)
plt.ylabel("y (pixels)", fontsize=22)
plt.title("$Log_{10}(A / rN)$.", fontsize=22)
print("Please select points, then press <ENTER> to proceed.")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.pause(0.001)
x_y_list = plt.ginput(n=100,timeout=0, show_clicks=True)
x_y_list = np.array(x_y_list)
#number = range(1,len(x_y_list))
#%%
for i, item in enumerate(x_y_list):
    ax = plt.gca()
    circ = plt.Circle((item[0],item[1]),6, color="white", fill=False)
    ax.add_artist(circ)
    ax.annotate(i+1, (item[0]-8, item[1]-16), color="white", size=25)
    plt.draw()



#%%


#%% Fitting of selected area - 30 x 30
x_y_size = 16
# Area of Interest coordinates setup: meshgrid for plotting, x_fit and y_fit for fitting 30x30 grid.
X_AOI, Y_AOI = np.mgrid[:x_y_size,:x_y_size]

coordinates = [(n,m) for n in range(x_y_size) for m in range(x_y_size)]

x_fit = [item[0] for item in coordinates]
y_fit = [item[1] for item in coordinates]

#%% Execution of Fitting function, returning all the appropriate values for evaluation.

fit_A, fit_rN, fit_A_by_rN, f_5007_list, m_5007_list, M_5007_list, fit_chi_sq, fit_chi_sq_r, m_fit_3rd = fitting_process(x_y_list)

print("%d of the %d fits of selected potential PNe are good fits." % (sum(i<2 for i in fit_chi_sq_r), len(fit_A)))
PNe_number = list(range(1,len(fit_A)+1))

#%% pandas dataframe

PNe_df = pd.DataFrame(columns=("PNe number","Amplitude", "rN", "A/rN","A/rN > 3", 
                               "Total Flux", "m_5007", "M_5007", "chi_sq", "chi_sq_r"))

PNe_df["PNe number"] = PNe_number
PNe_df["Amplitude"] = fit_A
PNe_df["rN"] = fit_rN
PNe_df["A/rN"] = fit_A_by_rN
PNe_df["A/rN > 3"] = PNe_df["A/rN"] > 3
PNe_df["Total Flux"] = f_5007_list
PNe_df["m_5007"] = m_5007_list
PNe_df["M_5007"] = M_5007_list
PNe_df["chi_sq"] = fit_chi_sq
PNe_df["chi_sq_r"] = fit_chi_sq_r

# Count how many PNe are above the 4 * rN criteria
A_by_rN_test = (PNe_df["A/rN > 3"] == True).sum()
print("%d of %d selected sources are above the threshold of 3 times the residual noise." % (A_by_rN_test, len(fit_A)))


#%% Round off appropriate values, then append them to a .txt file table.

x = [round(item[0]) for item in x_y_list]
y = [round(item[1]) for item in x_y_list]

f_5007_round = ["{0:.3g}".format(item) for item in f_5007_list]
m_5007_round = [rounding_func(item) for item in m_5007_list]
M_5007_round = [rounding_func(item) for item in M_5007_list]
chi_r_round = [rounding_func(item) for item in fit_chi_sq_r]


fit_results_table = Table(data=(PNe_number, x, y, f_5007_round, m_5007_round, M_5007_round, chi_r_round), 
                          names=("PNe number", "x", "y", "Total Flux (erg $/ s^-1 / cm^-1$)", "$m_{5007}$", "$M_{5007}$", "${\chi^2}_r$"))

check = input("Do you wish to overwrite previous data? Y/N:  ")

if check == "y":
    ow = True
    ascii.write(fit_results_table, "exported_data/data_file_for_write_up.csv", format="csv", overwrite=ow)
    ascii.write(fit_results_table, "exported_data/LaTex table output.txt", format="latex", overwrite=ow)
    ascii.write(fit_results_table, "exported_data/Fit results table.txt", overwrite=ow)
    print("fit_results_table data overwritten.")
elif check == "n":
    ow = False
    print("fit_results_table data was not overwritten")



#%% Plot of A/rN vs. reduced chi square

to_be_crossed_A_by_rN = PNe_df["A/rN"].loc[PNe_df["A/rN > 3"]==False]
to_be_crossed_chi_sq_r = PNe_df["chi_sq_r"].loc[PNe_df["A/rN > 3"]==False]

fig = plt.figure("Reduced Chi square plot", figsize=(12,10))
ax1 = plt.gca()
plt.clf()
plt.scatter(fit_A_by_rN, fit_chi_sq_r, c=fit_A_by_rN, cmap="viridis", marker=".", s=100)
plt.scatter(list(to_be_crossed_A_by_rN), list(to_be_crossed_chi_sq_r),c="r", marker="x", s=70)
plt.xlabel("A/rN from fit")
plt.ylabel("reduced chi square")
plt.title("A/rN vs. Reduced Chi Square statistic")
plt.axhline(y=1, c='k', ls="dashed")
plt.ylim(0, 2)
plt.show()

label = range(0,len(x_y_list))

for label, x, y in zip(label, fit_A_by_rN, fit_chi_sq_r):
    ax1=plt.gca()
    ax1.annotate(label+1, xy=(x+.05,y-0.015), color="black", size=10)
    plt.draw()

#%%

plt.figure("Magnitude Histogram", figsize=(12,10))
plt.clf()
PNe_df["M_5007"].loc[PNe_df["A/rN > 3"] == True].plot(kind="hist", bins=4,edgecolor="black",linewidth=0.8)
plt.xlabel("Abosulte Magnitude, M_5007")
plt.ylabel("Occurance")
plt.title("Absolute Magnitude histogram")
plt.xlim(-5, 0)

#%% Work on stellar dispersion comparison plot
stellar_vel = 0. # km/s
stellar_dis = 360. # km/s
PNe_vel_ratio = (PNe_vel - stellar_vel)/ stellar_dis

plt.figure(44, figsize=(12,10))
plt.clf()
plt.scatter(M_5007_list, PNe_vel_ratio, s=200)
plt.xlabel("$M_{5007}$", fontsize=20)
plt.ylabel("$V_{PNe} - V_{*} / \sigma_*$", fontsize=20)
plt.axhline(y=1, c="k", ls="dashed")
plt.axhline(y=0, c="k", ls="dashed")
plt.axhline(y=-1, c="k", ls="dashed")
plt.ylim(-2,2)
plt.xlim(-5,1)

#%% Luminosity function tests
#  N(M) ~~  exp(0.307M) * (1- exp(3(M_star - M))), M_star = -4.47mag
M_star = np.min(M_5007_list)
PNLF_list = []
for i in M_5007_list:
    PNLF_list.append(np.exp(0.307*i)*(1-np.exp(3*(M_star - i))))
#%% Single Case Study section for evaluation on a single PNe
'''
fig = plt.figure(2222, figsize=(12,10))
plt.clf()
plt.imshow(np.log10(Gauss_F_shape), origin="lower",cmap="CMRmap", vmin=1., vmax=3.)
cb = plt.colorbar()
cb.set_label("Flux (unit)")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("$Log_{10}(F)$.")
plt.show()
'''