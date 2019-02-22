import numpy as np

def minkowski_distance(a, b, p):
    return sum(map(lambda x: abs(x)**p, a-b))**(1/p)
