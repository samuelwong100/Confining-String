# -*- coding: utf-8 -*-
"""
File Name: equations_tools.py
Purpose: Useful classes for constants and functions related to SU(N) gauge
         group and superpotential
Author: Samuel Wong
"""

"""
===============================================================================
                                Conventions
                        
1. The general complex field, x, is stored as a (m x n) array. Each of the m
rows is a new point. So if x is only storing one field at one point, then its
shape is [1,n], ie., x = array([[x_1, x_2, ..., x_n]]), where x_i are complex. 
Each of the n columns are the component of the fields. In SU(N), n = N-1.

2. One exception is when we are storing a few constant vectors in SU(N) class.
In this case, even though there can be n components to these vectors, and there
might be m of these at a time, if m = 1, we store it as array([c1,c2,...,cn]).
For example, SU.rho = array([rho_1, rho_2, ...., rho_n]), since there is only
one rho ever.
===============================================================================
"""

import numpy as np
from numpy import sqrt, pi, exp
from copy import deepcopy
from scipy.misc import derivative

"""
==================================Functions====================================
"""

def delta(i,j):
    #kronecker delta
    if i==j:
        return 1
    else:
        return 0

def grad(f, points, dx = 1e-5):
    """
    NAME:
        grad
    PURPOSE:
        Calculate the numerical value of gradient for an array of points, using
        a function that is able to take an array of points
    INPUT:
        f = a differentiable function that takes an array of m points, each with n
            dimensions, and returns an (m,1) array
        points = (m,n) array, representing m points, each with n dimensions
    OUTPUT:
        (m,n) array, each row being a gradient
    """
    n = np.shape(points)[1]
    increment = dx*np.identity(n)
    df = []
    for row in increment:
        df.append((f(points + row) - f(points-row))/(2*dx))
    return np.array(df).T[0]        

def derivative_sample(x,h):
    """
    return the derivative of a sample of a function, x, which can have multiple
    components (column), and the points are stored as rows.
    """
    #get the derivaitve and fix the boundary issues
    first = (x[1] - x[0])/h
    last = (x[-1] - x[-2])/h
    dxdz = (np.roll(x,-1,axis=0) - np.roll(x,1,axis=0))/(2*h)
    dxdz[0] = first
    dxdz[-1] = last
    return dxdz

"""
==================================Classes======================================
"""

class SU:
    """
    A class representing an SU(N) gauge group. It is used to compute and store
    constants such as fundamental weights and weights of the fundamental 
    representation.
    
    Convention: sets of vectors are stored as rows of a matrix. To acess the
    first nu vector, for example, call SU.nu[0,:].
    """
    def __init__(self, N):
        self.N = N # number of color
        self.nu = self._define_nu() # weight of fundamental representation
        self.alpha = self._define_alpha() # simple roots
        self.w = self._define_w() # fundamental weights
        self.rho = self._define_rho() # Weyl's vector

    def _theta(self,a,A):
        # a=1,...,N-1
        # A=1,...,N
        if a<1 or a==self.N:
            raise Exception
        if a>=A:
            return 1
        elif a<A:
            return 0

    def _lambda(self,a,A):
        # the mathematical definition of lambda
        return (1/sqrt(a*(a+1)))*(self._theta(a,A) - a*delta(a+1,A))

    def _lambda_modified(self,A,a):
        # to use labmda to create matrix, first switch rows & columns, then
        # add 1 since python start counting at 0 but lambda start at 1
        return self._lambda(a+1,A+1)

    def _define_nu(self):
        # create a N by N-1 matrix that is defined by lambda
        # there are N nu vectors, each N-1 dimensional
        # the rows are the nu vectors
        nu = np.empty(shape=(self.N,self.N-1))
        for A in range(self.N):
            for a in range(self.N-1):
                nu[A][a] = self._lambda_modified(A,a)
        return nu

    def _define_alpha(self):
        #initialize an empty list
        ls=[]
        summation= 0
        # define the first N-1 alpha vectors
        for a in range(0,self.N-1):
            ls.append(self.nu[a,:] - self.nu[a+1,:])
            summation += self.nu[a,:] - self.nu[a+1,:]
        # the last alpha is the negative of the sum of the previous alpha
        ls.append(-1*summation)
        alpha = np.array(ls)
        return alpha

    def _define_w(self):
        # the i-th w is the sum of the first i nu
        # it is only defined up to the N-1th W
        # w is N-1 by N-1 matrix.
        w = np.zeros(shape=(self.N-1,self.N-1))
        summation = np.zeros(self.N-1)
        for A in range(0,self.N-1):
            # for every row of w, add up increasingly more nu vectors
            summation += self.nu[A,:]
            w[A,:] = deepcopy(summation)        
        return w

    def _define_rho(self):
        summation= np.zeros(self.N-1)
        # rho is a N-1 dimensional row vector
        for A in range(0,self.N-1):
            summation += self.w[A,:]
        return summation

class Superpotential():
    """
    A class representing a superpotential, working under SU(N).
    """

    def __init__(self,N):
        self.N = N
        self.s = SU(N)
        self.x_min = self._define_x_min() # minimum of superpotential

    def __call__(self,x):
        # x is an num by N-1 array, where num is the number of points
        # and N-1 is the number of complex fields,
        num = x.shape[0]
        result = np.zeros(shape=(num,1),dtype=complex)
        for row in range(num):
            summation = 0
            for i in range(0,self.N):
                summation += exp(np.dot(self.s.alpha[i,:],x[row,:]))
            result[row] = summation
        return result

    def _define_x_min(self):
        ls = []
        #there are N minimum vectors, each is 1 by N-1 dimensional
        for k in range(0,self.N+1):
            ls.append(1j*(2*pi/self.N)*k*self.s.rho)
        x_min = np.array(ls)
        return x_min

class BPS():
    """
    A class representing the BPS equation in SU(N).
    """
    def __init__(self,N,k0,kf):
        """
        k0 = initial vacum index; can be 0,1,...,N (N is the same as 0 actually)
        kf = final vacum index; can be 0,1,...,N
        """
        if k0 is kf:
            # if initial and final vacum are the same, then there is no
            # soliton
            raise Exception
 
        self.s = SU(N)
        self.W = Superpotential(N)
        self.N = N
        self.xmin0, self.xminf = self._define_min(k0,kf)
        numerator = self.W(self.xminf) - self.W(self.xmin0)
        denominator = np.absolute(numerator)
        self.alpha = numerator/denominator
        
    def _define_min(self,k0,kf):
    # if k0 is integer, then it is k0-th minimum of the superpotential
    # if k0 is a string of the form "wn", where n is an integer, then the 
    # minimum is the nth w vector of the sU(N)
        if type(k0) == int:
            xmin0 = np.array([self.W.x_min[k0,:]])
        elif type(k0) == str:
            if self.N == 4:
                w = 1j*2*pi*self.s.w
                w1 = np.array([w[0,:]])
                w2 = np.array([w[1,:]])
                w3 = np.array([w[2,:]])
                dictionary = {"w1":w1,
                              "w2":w2,
                              "w3":w3,
                              "w1+w2":w1+w2,
                              "w2+w3":w2+w3,
                              "w1+w3":w2+w3,
                              "x1+w1":np.array([self.W.x_min[1,:]]) + w1
                                }
            elif self.N == 3:
                dictionary = {"w1":np.array([1j*2*pi*self.s.w[0,:]]),
                              "w2":np.array([1j*2*pi*self.s.w[1,:]]),
                              "w1+w2":np.array([1j*2*pi*self.s.w[0,:] +
                                                1j*2*pi*self.s.w[1,:]])}
            elif self.N == 2:
                dictionary = {"w1":np.array([1j*2*pi*self.s.w[0,:]])}
            elif self.N == 5:
                w = 1j*2*pi*self.s.w
                w1 = np.array([w[0,:]])
                w2 = np.array([w[1,:]])
                w3 = np.array([w[2,:]])
                w4 = np.array([w[3,:]])
                dictionary = {"w1":w1,
                              "w2":w2,
                              "w3":w3,
                              "w4":w4,
                              "w1+w2":w1+w2,
                              "w1+w3":w1+w3,
                              "w1+w4":w1+w4,
                              "w2+w3":w2+w3,
                              "w2+w4":w2+w4,
                              "w3+w4":w3+w4,
                              "w1+w2+w3":w1+w2+w3,
                              "w1+w2+w4":w1+w2+w4,
                              "w1+w3+w4":w1+w3+w4,
                              "w2+w3+w4":w2+w3+w4,
                              }
            elif self.N == 6:
                w1=np.array([1j*2*pi*self.s.w[0,:]])
                w2=np.array([1j*2*pi*self.s.w[1,:]])
                w3=np.array([1j*2*pi*self.s.w[2,:]])
                w4=np.array([1j*2*pi*self.s.w[3,:]])
                w5=np.array([1j*2*pi*self.s.w[4,:]])
                dictionary = {"w1":w1,
                              "w2":w2,
                              "w3":w3,
                              "w4":w4,
                              "w5":w5,
                              "w1+w2+w3":w1+w2+w3,
                              "w1+w3+w5":w1+w3+w5,
                              "w2+w4":w2+w4,
                              "1+w1":np.array([self.W.x_min[1,:]]) + np.array([1j*2*pi*self.s.w[0,:]])
                              }
            elif self.N == 8:
                w1=np.array([1j*2*pi*self.s.w[0,:]])
                w2=np.array([1j*2*pi*self.s.w[1,:]])
                w3=np.array([1j*2*pi*self.s.w[2,:]])
                w4=np.array([1j*2*pi*self.s.w[3,:]])
                w5=np.array([1j*2*pi*self.s.w[4,:]])
                w6=np.array([1j*2*pi*self.s.w[5,:]])
                w7=np.array([1j*2*pi*self.s.w[6,:]])
                dictionary = {"w1":w1,
                              "w2":w2,
                              "w3":w3,
                              "w4":w4,
                              "w5":w5,
                              "w6":w6,
                              "w7":w7,
                              "w2+w3+w6+w7":w2+w3+w6+w7,
                              "w1+w4+w5":w1+w4+w5,
                              "w1+w2+w5+w6":w1+w2+w5+w6,
                              "w3+w4+w7":w3+w4+w7
                              }      
            xmin0 = dictionary[k0]
        elif isinstance(k0,np.ndarray):
            #if user passes in the actual numpy array of the minimum
            xmin0 = k0
    
        if type(kf) == int:
            xminf = np.array([self.W.x_min[kf,:]])
        elif type(kf) == str:
            kf = kf.translate({ord('w'): None})
            kf = int(kf)
            xminf = np.array([1j*2*pi*self.s.w[kf-1,:]])
        elif isinstance(kf,np.ndarray):
            #if user passes in the actual numpy array of the minimum
            xminf = kf
        return (xmin0, xminf)

    def dx(self,x):
        """
        BPS equations
        """
        x_ = np.conj(x)
        dWdx_ = grad(self.W,x_)
        return (self.alpha/2)*dWdx_

    def _Hessian(self,x):
        """
        Hessian Matrix for W*
        """
        m = x.shape[0] # there are m points
        # for each point, there is a Hessian matrix
        # so we will return a (length m) list of Hessian
        ls = []
        for row in range(m):
            ls.append(self._define_Hessian(x[row]))
        return ls

    def _define_Hessian(self,x):
        # x is now a row array
        #initialize Hessian
        H = np.zeros(shape=(self.N-1,self.N-1),dtype=complex)
        for i in range(self.N-1):
            for j in range(self.N-1):
                summation = 0j
                for a in range(self.N):
                    summation += self.s.alpha[a][i] * self.s.alpha[a][j] * \
                    exp(np.dot(self.s.alpha[a],np.conj(x)))
                H[i][j] = summation
        return H

    def ddx(self,x):
        """
        Second order BPS equations
        """
        dWdx = grad(self.W,x)
        m = x.shape[0]
        result = np.zeros(shape=(m,self.N-1),dtype=complex)
        H_ls = self._Hessian(x)
        for (row,H) in enumerate(H_ls):
            result[row] = np.matmul(H,dWdx[row])/4
        return result
