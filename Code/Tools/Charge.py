# -*- coding: utf-8 -*-
"""
File Name: Charge.py
Purpose: A class that stores a charge and handles its name in string
Author: Samuel Wong
"""
import numpy as np
from Math import SU

class Charge():
    """
    
    Variables
    ----------------------------------------
    N (int) = Number of colors
    arg (str) = a string that describes a linear combo of fundamental weights;
                Must be in the form of:
               "wi +/- wj +/- wk ..."
               where i,j,k are integers, and it is either "+" or "-" for each
    vector (array) = an array of shape (N-1,); the sum of fundamental weights
    """
    def __init__(self,N,arg):
        self.N = N
        self.S = SU(N)
        if isinstance(arg,str):
            self.name = arg
            w_str_list = arg.split()
            self.vector = self._sum_up_w(w_str_list)
            
    def __str__(self):
        return self.name
            
    def _sum_up_w(self, w_str_list):
        summation = np.zeros(shape=(self.N-1,),dtype=complex)
        for term in w_str_list:
            sign, term = self._get_term_sign(term)
            if len(term)==2 and term[0] == 'w':
                k = int(term[1])
                summation += sign*self.S.w[k-1,:]
            else:
                raise Exception("Unacceptable charge specification.")
        return summation
        
    def _get_term_sign(self,term):
        sign = 1
        if term[0] == "+":
            term = term.replace("+","")
        elif term[0] == "-":
            sign = -1
            term = term.replace("-","")
        return sign, term
                