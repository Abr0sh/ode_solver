from typing import Any
import numpy as np
class Oscillator:
    """ 
    - vertical oscillator: massless spring k with dampending beta. (d^2/dt^2) Y = g -k/m Y - beta/m dY/dt
    - vectorized form:
        d/dt [u0,u1] = [u1, g -(k/m)u0 - beta/m u1]
    """
    def __init__(self,m,g,beta,k):
        self.m = m 
        self.g = g
        self.beta = beta
        self.k = k 
    def __call__(self, t,y):
        y0, y1 = y
        m , beta, k, g = self.m,self.beta,self.k,self.g
        return np.array([y1,g-(k/m)*y0-(beta/m)*y1])
    
