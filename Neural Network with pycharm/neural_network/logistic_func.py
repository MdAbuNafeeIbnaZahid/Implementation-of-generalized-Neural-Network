import math

def getF(x, a):
    denominator = 1 + math.e ** (-a * x)
    return 1/denominator;



def getFPrime(x, a):
    return  a * getF(x,a) * (1 - getF(x,a) )