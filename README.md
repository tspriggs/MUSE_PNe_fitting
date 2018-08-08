# Detection of Planetary Nebulae within the central arcminute of M87.

Having fitted the rescaled, continuum emission subtracted data cube of M87, as observed with the MUSE Integral Field Spectrograph, attached to the VLT in Chile, I was able to find isolated peak sources of [OIII] emission using an amplitude diviced by residual noise map that highlights such feautures.

[OIII] emissions are found as emission doublets at 4959 and 5007 angstrom, where the first peak is 1/3 the amplitude of the second peak.

## Modelling the PNe
For modelling the PNe and retrieving their Absolute magnitude in 5007 angstrom, a custom routine has been written in python to characterize the properties of both a 2D Moffat and 1D Gaussian profile, hence allowing for the total flux calculation and magnitude conversion.

Here a 3D model has been constructed, and utilised by a chi-square minimizaition program provided by the 3rd party python package LMfit (link here).
First the parameters are initialised and estimated, with relevant limits and wether or not the parameter will be held fixed for the process. Then a 2D moffat profile is setup, extracting each pixel's value of flux, converting this to an amplitude. The disregard for a background estimation means that the spread of amplitudes in the subsequent 1D gaussian array per pixel will be spatially spread out in a moffat profile; at the outer reaches of the datacube the useful amplitude should be near to or exactly zero.
Once all the 1D Gaussian profiles are created with unique noise added on, the whole model is compared with the data:   data - model / error

Once minimization of the source is complete, the relevant interesting values are calculated and stored in a Pandas dataframe.