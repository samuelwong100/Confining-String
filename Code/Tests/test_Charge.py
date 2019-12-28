# -*- coding: utf-8 -*-
"""
File Name: test_Charge.py
Purpose: test Charge class
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
from Charge import Charge
from Math import SU

def test_Charge():
    S = SU(4)
    c = Charge(4,"w1 +w2 -w3")
    print("correct name =", "w1 +w2 -w3")
    print("actual name =", c)
    print()
    print("correct vector", str(S.w[0,:] + S.w[1,:] - S.w[2,:]))
    print("actual vector =", c.vector)
    
test_Charge()