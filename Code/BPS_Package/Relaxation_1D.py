# -*- coding: utf-8 -*-
"""
File Name: Relaxation_1D.py
Purpose: General class for solving second order ODE by relaxation method
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class Relaxation_1D():
    """
    Solve the boundary value problem of a set of coupled, second order,
    ordinary differential equation with m components using relaxation method.
    
    The ODE has to be of the form
    d^2(f(z))/dz^2 = g(f(z))
    where f can be a vector.
    """
    def __init__(self,g,z0,zf,bound0,boundf):
        # source function, which take (num,m) array: num points, each m dim.
        # g must returns (num,m) array
        self.g = g
        # first and last point of dependent variable
        self.z0 = z0
        self.zf = zf
        # boundary values, of shape (m,), where m is the number of fields
        self.bound0 = bound0
        self.boundf = boundf
        self.m = bound0.size  # dimension of f / number of ODE
        self.sol_z = None # solution - independent variable
        self.sol_f = None # solution - dependent variable
        
    def _update(self,num,f_old,h,z):
        # replace each element of f_old with sum of left and right neighbors,
        # plus a source term
        f_new = deepcopy(f_old)
        source = self.g(f_new)
        for k in range(1,num-1):
            f_new[k] = (f_new[k-1] + f_new[k+1] - source[k]*(h**2))/2
        # set the boundary values unchanged
        f_new[0]= self.bound0
        f_new[-1]= self.boundf
        return f_new
        
    def _relaxation(self,num,tol,f,h,z,diagnose):
        # initialize error to be larger than tolerance
        error = tol+1
        loop = 0
        while error>tol:
            f_new = self._update(num,f,h,z)
            error = np.sum(np.abs(f_new - f))/(num*self.m)
            f = f_new
            if diagnose:
                if loop % 100 == 0:
                    print("loop =",loop,"error =",error)
                    plt.figure()
                    for i in range(self.m):
                        plt.plot(z,np.real(f[:,i]))
                        plt.plot(z,np.imag(f[:,i]))
                    plt.show()
                loop = loop + 1
        return f
    
    def _set_f0(self,f0,num,R,top):
        # define a initial f if none was given
        if f0 is None:
            f0 = np.ones(shape=(num,self.m),dtype=complex)*complex(1,1)
        elif f0 == "special kink":
            f0 = np.ones(shape=(num,self.m),dtype=complex)
            half = num // 2
            f0[0:half,:] = self.bound0 + 0.1
            f0[half:-1,:] = self.boundf + 0.1
        elif f0 == "kink with predicted width":
            f0 = np.ones(shape=(num,self.m),dtype=complex)
            #predict the width to be d/2 = y(R/2) = ln(R/2+1)-R/(R+2)
            height = np.log(R/2+1)-R/(R+2) #equivalent to d/2
            print("height = ",height)
            ratio = height/np.abs(self.zf - self.z0)
            print("zf = ",self.zf)
            print("z0 = ",self.z0)
            print("zf-z0 =", self.zf - self.z0)
            print("ratio = ",ratio)
            if top:
                kink_pixel_number = int((1-ratio)*num)
            else:
                kink_pixel_number = int(ratio*num)
            f0[0:kink_pixel_number,:] = self.bound0
            f0[kink_pixel_number:-1,:] = self.boundf
        elif f0 == "special kink without +0.1":
            f0 = np.ones(shape=(num,self.m),dtype=complex)
            half = num // 2
            f0[0:half,:] = self.bound0
            f0[half:-1,:] = self.boundf
        f0[0]= self.bound0
        f0[-1]= self.boundf
        return f0
        
    def solve(self,R,top,num,tol=1e-5,f0=None,diagnose=True):
        """
        num = number of points in grid
        tol = tolerance of error
        f0 = initial f; must be of shape (num,m); must obey boundary conditions
        """
        #define variables
        h = (self.zf - self.z0)/num # pixel
        z = np.linspace(self.z0,self.zf,num) # grid for z
        f0 = self._set_f0(f0,num,R,top)
        
        # solve
        f = self._relaxation(num,tol,f0,h,z,diagnose)
        self.sol_z = z
        self.sol_f = f
