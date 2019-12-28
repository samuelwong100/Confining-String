# -*- coding: utf-8 -*-
"""
File Name: test_Charge.py
Purpose: test Charge class
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import numpy as np
from Sigma_Space_Critical_Points import Sigma_Critical
from Math import SU, Superpotential

def test_Charge():
    S = SU(4)
    c = Sigma_Critical(4,"w1 +w2 -w3")
    print("correct name =", "w1 +w2 -w3")
    print("actual name =", c)
    print()
    print("correct vector", str(S.w[0,:] + S.w[1,:] - S.w[2,:]))
    print("actual vector =", c.real_vector)
    
def test_xmin():
    W = Superpotential(4)
    SC = Sigma_Critical(4,"x2 +x3")
    print("correct name =", "x2 +x3")
    print("actual name =", SC)
    print()
    print("correct vector", str(np.imag(W.x_min[2,:]+W.x_min[3,:])))
    print("actual vector =", SC.real_vector)
    
def test_mixed():
    S = SU(4)
    W = Superpotential(4)
    SC = Sigma_Critical(4,"x2 +w1 +w3")
    print("correct name =", "x2 +w1 +w3")
    print("actual name =", SC)
    print()
    print("correct vector", str(np.imag(W.x_min[2,:])
    +np.real(S.w[0,:]+S.w[2,:])))
    print("actual vector =", SC.real_vector)
    print("actual imaginary vector =",SC.imaginary_vector)
    
#test_Charge()#pass
#test_xmin() #pass
#test_mixed() #pass