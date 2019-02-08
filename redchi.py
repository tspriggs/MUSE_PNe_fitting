def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417

    """
    y = np.ravel(y)
    d = y if zero else y - np.nanmedian(y)

    mad = np.nanmedian(np.abs(d))
    u2 = (d/(9.0*mad))**2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good]*u1**2)**2).sum()
    den = (u1*(1.0 - 5.0*u2[good])).sum()
    sigma = np.sqrt(num/(den*(den - 1.0)))  # see note in above reference

    return sigma

def calc_redchi(redchi,histogram='y'):
    # Using robust sigma to determine the the standard deviation of the chisqr distribution
    sig    = robust_sigma(redchi)
    median = np.median(redchi)
    chilim = [median-3.0*sig,median+3.0*sig] # We consider 3 times the standard deviation of the distribution

    redchi = redchi[(redchi >= chilim[0]) & (redchi <= chilim[1])]

    if histogram == 'y':
        plt.figure(figsize=(10,10))
        bins  = plt.hist(redchi, edgecolor="black", linewidth=0.8, alpha=0.5)
        chi = np.arange(-2.0,2.0,0.001)
        #gauss = max(bins[0])*np.exp(-(chi - cent)**2/(2.0*np.var(PNe_df_cen["redchi"].loc[(PNe_df_cen["redchi"] < chilim[1]) & (PNe_df_cen["redchi"] > chilim[0])])))
        gauss = max(bins[0])*np.exp(-(chi - median)**2/(2.0*np.var(redchi)))

        plt.plot(chi,gauss)
        plt.xlim(median-5.0*sig,median+5.0*sig)
        plt.axvline(median, color='red')
        plt.axvline(chilim[0]); plt.axvline(chilim[1])
        plt.annotate(s=r'$3\sigma$',xy=(median+3.5*sig,30.0),xytext=(median+3.5*sig,30.0))
        plt.annotate(s=r'$-3\sigma$',xy=(median-3.5*sig,30.0),xytext=(median-3.5*sig,30.0))
        plt.show()
        input('Press ENTER to close it')
        plt.close()

    return chilim
