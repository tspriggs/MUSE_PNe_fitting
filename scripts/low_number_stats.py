import numpy
from scipy.special import bdtri, ndtr, pdtri
from sys import exit

def poissonLimits(k, cl=None, sigma=False):
    """
    NAME:
          poissonLimits
    AUTHOR:
          Tim Haines, thaines.astro@gmail.com
    PURPOSE:
          This function computes the single-sided upper and lower
          confidence limits for the Poisson distribution.
    CATEGORY:
          Statistics and probability
    CALLING SEQUENCE:
          (u,l) = poissonLimits(k, [cl [, sigma]])
    INPUTS:
          k:      A strictly nonnegative integer that specifies the
                  number of observed events. Can be a list or numpy array.
    OPTIONAL INPUTS:
          cl:     The confidence level in the interval [0, 1]. The default
                  is 0.8413 (i.e., 1 sigma)
    OPTIONS:
          sigma:  If this is true, then cl is assumed to be a
                  multiple of sigma, and the actual confidence level
                  is computed from the standard normal distribution with
                  parameter cl.
    RETURNS:
           Two lists: the first containing the upper limits, and the second
           containing the lower limits. If the input is a numpy array, then
           numpy arrays are returned instead of lists. If the input is a scalar,
           then scalars are returned.
    REFERENCES:
          N. Gehrels. Confidence limits for small numbers of events in astrophysical
          data. The Astrophysical Journal, 303:336-346, April 1986.
    EXAMPLE:
          Compute the confidence limits of seeing 20 events in 8
          seconds at the 2.5 sigma.
              (u,l) = poissonLimits(20, 2.5, sigma=True)
                  u = 34.1875
                  l = 10.5711
          However, recall that the Poisson parameter is defined as the
          average rate, so it is necessary to divide these values by
          the time (or space) interval over which they were observed.
          Since these are the confidence limits, the fraction would be
          reported as
              2.5 (+4.273, -1.321) observations per second
    """
    if cl is None:
        cl = 1.0
        sigma = True
    if sigma:
        cl = ndtr(cl)
    # Since there isn't any syntactical advantage to using
    # numpy, just convert it to a list and carry on.
    isNumpy = False
    if isinstance(k,numpy.ndarray):
        k = k.tolist()
        isNumpy = True
    # Box single values into a list
    isScalar = False
    if not isinstance(k,list):
        k = [k]
        isScalar = True
    upper = []
    lower = []
    for x in k:
        upper.append(pdtri(x,1-cl))
        # See Gehrels (1986) for details
        if x == 0:
            lower.append(0.0)
        else:
            lower.append(pdtri(x-1,cl))
    if isNumpy:
        upper = numpy.array(upper)
        lower = numpy.array(lower)
    # Scalar-in/scalar-out
    if isScalar:
        return (upper[0], lower[0])
    return (upper,lower)