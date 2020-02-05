# -*- coding: utf-8 -*-
"""
File Name: Math.py
Purpose: math tools
Author: Samuel Wong
"""

"""
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

import numpy as np
from numpy import sqrt, pi, exp
from copy import deepcopy
import matplotlib.pyplot as plt

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

def dot_vec_with_vec_field(vec,vec_field):
    #assume vec is an array of shape (n,)
    #vector field is an array of shape (n,x,y), where the field has 
    #n components, and lives on a x-y grid.
    #return the dot product the the vector and the field at every point
    #on the grid. The return is a (x,y) grid object
    return np.sum((vec*(vec_field.T)).T,axis=0)

"""
================================  Classes   ===================================
"""

class SU():
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
        
    def dWdx(self,x):
        """
        First derivative vector, ie. gradient
        
        Input
        -------------------------------------------
        x (array) = the value of the field at m points;an array of shape
                    (m,N-1); where m is number of points,
                    N-1 is number of field components
    
        Output
        --------------------------------------------
        dWdx (array) = an array of shape (m,N-1), where each row is the
                        gradient vector of the corresponding point
        """
        m = x.shape[0]
        dWdx = np.zeros(shape=(m,self.N-1),dtype=complex)
        for i in range(m):
            #fix the ith point
            for j in range(self.N-1):
            #fix the jth partial derivative
                dWdx[i][j] = self._jth_parital_for_dWdx(x[i,:],j)
        return dWdx
    
    def _jth_parital_for_dWdx(self,x_row,j):
        #intialize sum of exponential
        summation = 0
        for k in range(self.N):
            summation += exp(np.dot(self.s.alpha[k,:],x_row))*self.s.alpha[k][j]
        return summation
    
    def dWdx_absolute_square(self,x):
        """
        Absolute squared of gradient.
        
        Input
        -------------------------------------------
        x (array) = the value of the field at m points;an array of shape
                    (m,N-1); where m is number of points,
                    N-1 is number of field components
    
        Output
        --------------------------------------------
        dWdx (array) = an array of shape (m,)
        """
        dWdx = self.dWdx(x)
        dWdx_conj = np.conjugate(dWdx)
        return np.sum(dWdx*dWdx_conj,axis=1)
    
    def dWdx_absolute_square_on_grid(self,x):
        """
        Absolute squared of gradient, evaluted on a grid.
        
        Input
        -------------------------------------------
        x (array) = a grid of shape (layers,n,m)
    
        Output
        --------------------------------------------
        dWdx (array) = a grid of shape (n,m)
        """
        layers,row_num,col_num = x.shape
        result = np.zeros(shape=(row_num,col_num),dtype=complex)
        for row in range(row_num):
            #take the row from each layer, transpose it such that the different fields
            #of the same point are in a row vector
            result[row,:] = self.dWdx_absolute_square(x[:,row,:].T)
        return result
    
    def ddWddx(self,x):
        """
        Diagonal of second derivative matrix, ie. d^2(W)/d(x^a)^2
        """
        #x is a point of shape (m,N-1); where m is number of points, N-1 is field
        m = x.shape[0] # number of points
        #initialize the second derivative array, which has m row, each row
        #is the diagonal of second derivative matrix of the m-th point
        #Each vector has N-1 dimensions (for N-1 fields)
        result = np.zeros(shape=(m,self.N-1),dtype=complex)
        #loop over each point (ie, row in x)
        for (m,x_row) in enumerate(x):
            #loop over each of N-1 field, each corresponding to the independent
            #variable of the second derivative
            for i in range(self.N-1):
                result[m][i] = self._sum_of_exp_for_ddW(x_row,i)
        return result

    def _sum_of_exp_for_ddW(self,x_row,i):
        summation = 0j
        for a in range(self.N):
            coef = (self.s.alpha[a][i])**2
            summation +=  coef * exp(np.dot(self.s.alpha[a],x_row))
        return summation
    
#    def ddW_dxb_dxa(self,x,b,a):
#        """
#        Mixed second partial derivative, d^2(W)/(dx^b dx^a)
#        
#        Input
#        -------------------------------------------
#        x (array) = the value of the field at m points;an array of shape
#                    (m,N-1); where m is number of points,
#                    N-1 is number of field components
#        b (int) = index of field
#        a (int) = index of field
#    
#        Output
#        --------------------------------------------
#        second_derivative (array) = an array of shape (m,)
#        """
#        m = x.shape[0] # number of points
#        #initialize
#        second_derivative = np.zeros(shape=(m,),dtype=complex)
#        #loop over each point (ie, row in x)
#        for (j,x_row) in enumerate(x):
#            second_derivative[j] = self._ddW_dxb_dxa_point(x_row,a,b)
#        return second_derivative
#        
#    def _ddW_dxb_dxa_point(self,x_row,a,b):
#        summation = 0j
#        for i in range(self.N):
#            coeff = self.s.alpha[i][a]*self.s.alpha[i][b]
#            summation += coeff * exp(np.dot(self.s.alpha[i],x_row))
#        return summation
#    
#    def sum_over_dWdxa_ddWdxbdxa_conj(self,x,b):
#        """        
#        Input
#        -------------------------------------------
#        x (array) = the value of the field at m points;an array of shape
#                    (m,N-1); where m is number of points,
#                    N-1 is number of field components
#        b (int) = index of field
#    
#        Output
#        --------------------------------------------
#        row vector of dW dotted ddW (array) = an array of shape (m,)
#        """
#        #each row of dw_list is gradient of a point
#        #each column is each partial derivative component
#        dW_list = self.dWdx(x)
#        ddW_list = []
#        for a in range(0,self.N-1):
#            ddW_list.append(np.conjugate(self.ddW_dxb_dxa(x,b,a)))
#        #each row of ddW_list is a 1D array of partial derivative,
#        #each parital is for a different point
#        #if we take transpose, each row becomes different partial of the same point
#        ddW_list = np.array(ddW_list).T
#        #multiply component wise and sum over the columns
#        #now each row should be the answer we want for each point
#        #but then the sum function reduces the dimension and transposes it
#        #so now it is a row vector
#        return np.sum(dW_list*ddW_list,axis=1)
#    
#    def sum_over_dWdxa_ddWdxbdxa_conj_on_grid(self,x):
#        #this time x is a grid with N-1 layer, each representing a field
#        layers,row_num,col_num = x.shape
#        x_final = np.zeros(x.shape,dtype=complex)
#        for r in range(row_num):
#            #take a row with multiple layers
#            for b in range(layers):
#                #for each layer, the values of that row is based on b
#                #need to take transpose since x has fields in layers; here
#                #this means the points are in the same row
#                #But in sum function, the points are assumed to be different rows
#                x_final[b,r,:] = self.sum_over_dWdxa_ddWdxbdxa_conj(x[:,r,:].T,b)
#        return x_final
#            
#    def potential_term_on_grid(self,x):
#        return (1/4)*self.sum_over_dWdxa_ddWdxbdxa_conj_on_grid(x)
    
    def _potential_term_on_grid_slow(self,x):
        layers,row_num,col_num = x.shape
        result = np.ones(shape=x.shape,dtype=complex)
        for r in range(row_num):
            for c in range(col_num):
                x_vec= x[:,r,c]
                result[:,r,c] = self._sum_slow(x_vec)
        return result
            
    def _sum_slow(self,x_vec):
        vec = np.zeros(self.N-1,dtype=complex)
        for a in range(self.N-1):
            vec[a] = self._sum_a_slow(x_vec,a)
        return vec
    
    def _sum_a_slow(self,x_vec,a):
        summation = 0j
        x_vec_conj = np.conjugate(x_vec)
        for b in range(self.N-1):
            for k in range(self.N):
                for l in range(self.N):
                   coeff = self.s.alpha[k][a]*self.s.alpha[k][b]*self.s.alpha[l][b]
                   dot_prod_1 = np.dot(self.s.alpha[k],x_vec_conj)
                   dot_prod_2 = np.dot(self.s.alpha[l],x_vec)
                   summation += coeff*np.exp(dot_prod_1 + dot_prod_2)
        return summation/4

    def potential_term_on_grid_fast(self,x):
        layers,row_num,col_num = x.shape
        result = np.ones(shape=x.shape,dtype=complex)
        for r in range(row_num):
            for c in range(col_num):
                x_vec= x[:,r,c]
                result[:,r,c] = self._sum_fast(x_vec)
        return result

    def _sum_fast(self,x_vec):
        vec = np.zeros(self.N-1,dtype=complex)
        for b in range(self.N-1):
            #the b-th component of the resulting vector
            vec[b] = self._term_b_fast(x_vec,b)
        return vec
    
    def _term_b_fast(self,x_vec,b):
        summation = 0j
        x_vec_conj = np.conjugate(x_vec)
        for a in range(self.N):
            #this is the equation we get from applying the idenity of
            #alpha_a dot alpha_b in terms of 3 delta function
            A = np.exp(np.dot(self.s.alpha[a],x_vec))
            B = np.exp(np.dot(self.s.alpha[a],x_vec_conj))*self.s.alpha[a][b]
            C = np.exp(np.dot(self.s.alpha[(a-1)%self.N],x_vec_conj)) \
            *self.s.alpha[(a-1)%self.N][b]
            D = np.exp(np.dot(self.s.alpha[(a+1)%self.N],x_vec_conj)) \
            *self.s.alpha[(a+1)%self.N][b]
            summation += A*(2*B-C-D)
        return summation/4
    
    def potential_term_on_grid_fast_optimized(self,x):
        summation = np.zeros(shape=x.shape,dtype=complex)
        x_conj = np.conjugate(x)
        for a in range(self.N):
            #this is the equation we get from applying the idenity of
            #alpha_a dot alpha_b in terms of 3 delta function
            A = np.exp(dot_vec_with_vec_field(self.s.alpha[a],x))
            B = np.exp(dot_vec_with_vec_field(self.s.alpha[a],
                        x_conj)) *self.s.alpha[a]
            C = np.exp(dot_vec_with_vec_field(self.s.alpha[(a-1)%self.N],
                        x_conj)) *self.s.alpha[(a-1)%self.N]
            D = np.exp(dot_vec_with_vec_field(self.s.alpha[(a+1)%self.N],
                        x_conj)) *self.s.alpha[(a+1)%self.N]
            summation += A*(2*B-C-D)
        return (summation.T)/4
        