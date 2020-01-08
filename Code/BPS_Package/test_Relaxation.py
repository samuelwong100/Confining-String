# -*- coding: utf-8 -*-
"""
File Name: test_Relaxation
Purpose: test relaxation class
Author: Samuel Wong
"""
import numpy as np
from numpy import pi, sin, cos, e, exp
import matplotlib.pyplot as plt
from Relaxation import Relaxation

def test1():
    z0 = 0
    zf = 5
    bound0 = np.array([1-(5**3)/6])
    boundf = np.array([1])
    
    def g(z,f):
        return z
    
    R = Relaxation(g,z0,zf,bound0,boundf)
    R.solve(100)
    
    #theoretical solution
    y = (1/6)*(R.sol_z)**3 + (1-(5**3)/6)
    
    plt.figure()
    plt.plot(R.sol_z,R.sol_f[0],label='numerical')
    plt.plot(R.sol_z,y,label='theoretical')
    plt.legend()
    plt.show()
    
def test2():
    z0 = 0
    zf = 2*pi
    bound0 = np.array([0])
    boundf = np.array([0])
    
    def g(z,f):
        return -f
    
    R = Relaxation(g,z0,zf,bound0,boundf)
    R.solve(10,tol=0.1)
    
    #theoretical solution
    y=sin(R.sol_z)
    
    plt.figure()
    plt.plot(R.sol_z,R.sol_f[0],label='numerical')
    plt.plot(R.sol_z,y,label='theoretical')
    plt.legend()
    plt.show()

def test3():
    z0 = 0
    zf = 1
    bound0 = np.array([1])
    boundf = np.array([e])
    
    def g(z,f):
        return f
    
    R = Relaxation(g,z0,zf,bound0,boundf)
    R.solve(100, tol = 0.00001)
    
    #theoretical solution
    y=exp(R.sol_z)
    
    plt.figure()
    plt.plot(R.sol_z,R.sol_f[0],label='numerical')
    plt.plot(R.sol_z,y,label='theoretical')
    plt.legend()
    plt.show()

#test1()
#test2()
test3()