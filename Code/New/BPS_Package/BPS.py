# -*- coding: utf-8 -*-
"""
File Name: BPS.py
Purpose: BPS equation class
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import numpy as np
from numpy import exp
from Math import SU, Superpotential, grad

class BPS():
    """
    A class representing the BPS equation in SU(N).
    """
    def __init__(self,N,xmin0,xminf): 
        self.s = SU(N)
        self.W = Superpotential(N)
        self.N = N
        self.xmin0 = np.array([xmin0])
        self.xminf = np.array([xminf])
        numerator = self.W(self.xminf) - self.W(self.xmin0)
        denominator = np.absolute(numerator)
        self.alpha = numerator/denominator

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