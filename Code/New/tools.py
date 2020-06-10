# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
import numpy as np
from numpy import sqrt, pi, exp
from copy import deepcopy

"""
===============================================================================
                                    Math
===============================================================================
    Conventions
                        
1. The general complex field, x, is stored as a (m x n) array. Each of the m
rows is a new point. So if x is only storing the field at one point, then its
shape is [1,n], ie., x = array([[x_1, x_2, ..., x_n]]), where x_i are complex. 
Each of the n columns are the component of the fields. In SU(N), n = N-1.

2. One exception is when we are storing a few constant vectors in SU(N) class.
In this case, even though there can be n components to these vectors, and there
might be m of these at a time, if m = 1, we store it as array([c1,c2,...,cn]).
For example, SU.rho = array([rho_1, rho_2, ...., rho_n]), since there is only
one rho ever.
===============================================================================
"""
def delta(i,j):
    #kronecker delta
    if i==j:
        return 1
    else:
        return 0


class SU():
    """
    A class representing an SU(N) gauge group. It is used to compute and store
    constants such as fundamental weights and weights of the fundamental 
    representation.
    
    Convention: sets of vectors are stored as rows of a matrix. To acess the
    first nu vector, for example, call SU.nu[0,:].
    """
    def __init__(self,N):
        self.N = N #rank of gauge group
        self.nu = self._define_nu() # weight of fundamental representation
        self.alpha = self._define_alpha() # simple roots and affine root
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
        # add 1 since python starts counting at 0 but lambda starts at 1
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

def dot_vec_with_vec_field(vec,vec_field):
    #assume vec is an array of shape (n,)
    #vector field is an array of shape (n,x,y), where the field has 
    #n components, and lives on a x-y grid.
    #return the dot product the the vector and the field at every point
    #on the grid. The return is a (x,y) grid object
    return np.sum((vec*(vec_field.T)).T,axis=0)

def grad(f, points, dx = 1e-5):
    """
    NAME:
        grad
    PURPOSE:
        Calculate the numerical value of gradient for an array of points, using
        a function that is able to take an array of points
    INPUT:
        f = a differentiable function that takes an array of m points, each
        with n dimensions, and returns an (m,1) array
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


class Superpotential():
    """
    Superpotential in SU(N).
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
    
    def potential_term_on_grid_fast_optimized(self,x):
        #to optimize, absorb all for loop over grid into numpy vectorization
        #ok to loop over N, since they are small
        x_conj = np.conjugate(x)
        summation = np.zeros(shape=x.shape,dtype=complex)
        #loop over a first instead of b, so that terms like exp_B only
        #computes once
        #the following is the equation we get from applying the idenity of
        #alpha_a dot alpha_b in terms of 3 delta function
        for a in range(self.N):
            dot_A=dot_vec_with_vec_field(self.s.alpha[a],x)
            dot_B=dot_vec_with_vec_field(self.s.alpha[a],x_conj)
            dot_C=dot_vec_with_vec_field(self.s.alpha[(a-1)%self.N],x_conj)
            dot_D=dot_vec_with_vec_field(self.s.alpha[(a+1)%self.N],x_conj)
            A = np.exp(dot_A)
            exp_B = np.exp(dot_B)
            exp_C = np.exp(dot_C)
            exp_D = np.exp(dot_D)
            #the a-th term in the vector field summation
            vec_a = np.zeros(shape=x.shape,dtype=complex)
            for b in range(self.N-1):
                #the b-th component of the resulting vector field
                B = exp_B*self.s.alpha[a][b]
                C = exp_C*self.s.alpha[(a-1)%self.N][b]
                D = exp_D*self.s.alpha[(a+1)%self.N][b]
                vec_a[b,:,:] += A*(2*B-C-D)
            summation += vec_a
        return summation/4
