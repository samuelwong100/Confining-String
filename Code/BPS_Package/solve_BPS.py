# -*- coding: utf-8 -*-
"""
File Name: Relaxation_Solution.py
Purpose: Function that calls relaxation and apply it to BPS equations
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
from Sigma_Critical import Sigma_Critical
from Relaxation_1D import Relaxation_1D
from BPS import BPS
from plot_BPS import plot_BPS

def solve_BPS(N,vac0_arg,vacf_arg,z0,zf,h,folder,tol):
    num = int((zf-z0)/h)
    vac0 = Sigma_Critical(N,vac0_arg)
    vacf = Sigma_Critical(N,vacf_arg)
    B=BPS(N,vac0.imaginary_vector,vacf.imaginary_vector)
    R = Relaxation_1D(B.ddx,z0,zf,vac0.imaginary_vector,vacf.imaginary_vector)
    R.solve(num,tol=tol,f0="special kink",diagnose=False)
    z = R.sol_z
    x = R.sol_f        
    plot_BPS(N,z,x,B,h,folder,vac0,vacf)
    return z,x
    