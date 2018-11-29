'''Module containing all psychometric curves used in this package'''
import numpy
from scipy.special import erf

def logistic(x: float,
             alpha: float,
             beta: float,
             gamma: float,
             lamb: float) -> float:
    '''TODO ADD DOCUMENTATION'''
    return gamma + (1.0 - gamma - lamb) * (1.0 + numpy.exp((-1.0) * (numpy.float64(x) - alpha) * beta)) ** (-1.0)

def weibull(x: float,
            alpha: float,
            beta: float ,
            gamma: float,
            lamb: float) -> float:
    '''TODO ADD DOCUMENTATION'''
    return gamma + (1 - gamma - lamb) * (1 - numpy.exp(-(x / alpha) ** beta))

def gaussian(x: float,
             alpha: float,
             beta: float,
             gamma: float,
             lamb: float) -> float:
    '''TODO ADD DOCUMENTATION'''

    return gamma + (1.0 - gamma - lamb) * ( 1.0 + erf((x - alpha) / numpy.sqrt(2.0 * beta ** 2.0))) / 2.0
