import math
import numpy as np

a = 1

def getF(x):
    denominator = 1 + np.power( math.e, (-a * x) )
    return 1/denominator;



def getFPrime(x):
    return  a * getF(x,a) * (1 - getF(x,a) )