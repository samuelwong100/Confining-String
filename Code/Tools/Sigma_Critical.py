# -*- coding: utf-8 -*-
"""
File Name: Sigma_Critical.py
Purpose: A class that stores critical points in sigma space and handles 
        their names in string. This includes both charge and field vaccua.
Author: Samuel Wong
"""
import numpy as np
from Math import SU, Superpotential

class Sigma_Critical():
    """
    Full name: Sigma Space Crictical Points
    A class that handles the vector and string name of general ciritical points
    of superpotential. This includes the charge of quarks in the form of linear
    comdinations of fundamental weights and field vaccua, which can be sum of 
    x_min vector with fundamental weights.
    
    Warning: There is a problem of convention. Technically, the sigma space is
    the imaginary component of the complex field, so sigma itself is real. When
    dealing with charge, we often talk about fundamental weights, which are 
    taken to be a real vector. But when we talk about x_min, it is usually in
    a context of it being the vacua of a full complex field, so it is often
    taken to be a purely imaginary vector. Since we want to have a class that
    deals with both generically, I will create two vectors, real and imaginary.
    They only differ by a factor of i.
    
    Variables
    ----------------------------------------
    N (int) = Number of colors
    arg (str) = a string that describes a linear combo of fundamental weights
                and/or x_min vectors;
                Must be in the form of:
               "wi +/-xj +/-wk ..."
               where the first letter of each term is either 'w' or 'x',
               followed by a integer (i,j,k), and it is either "+" or "-" for 
               each sign.
    vector (array) = an array of shape (N-1,)
    """
    def __init__(self,N,arg):
        self.N = N
        self.S = SU(N)
        self.W = Superpotential(N)
        self.name = arg
        self.real_vector = self._define_real_vector(arg)
        self.imaginary_vector = 1j*self.real_vector
            
    def __str__(self):
        return self.name
    
    def _define_real_vector(self,arg):
        str_list = arg.split() #split the string into list; separator is space
        return self._sum_up_terms(str_list)
            
    def _sum_up_terms(self, str_list):
        summation = np.zeros(shape=(self.N-1,),dtype=complex)
        for term in str_list:
            sign, term = self._get_term_sign(term)
            if len(term)==2:
                k = int(term[1])
                if term[0] == 'w':
                    #the usual w is already real
                    #the zeroth w corresponds to w1, need a shift
                    summation += sign*np.real(self.S.w[k-1,:])
                elif term[0] == 'x':
                    #the usual x_min is stored as purely imaginary
                    #want to convert to a real vector
                    #the zeroth x_min corresponds to x0, so no shift
                    summation += sign*np.imag(self.W.x_min[k,:])
                else:
                    raise Exception("Unacceptable sigma crticial points \
                                    specification due to non x or w character")
            else:
                raise Exception("Unacceptable sigma crticial points \
                                    specification due to term too long.")
        return summation
        
    def _get_term_sign(self,term):
        sign = 1
        if term[0] == "+":
            term = term.replace("+","")
        elif term[0] == "-":
            sign = -1
            term = term.replace("-","")
        return sign, term
                