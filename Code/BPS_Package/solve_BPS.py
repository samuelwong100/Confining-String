# -*- coding: utf-8 -*-
"""
File Name: Relaxation_Solution.py
Purpose: Function that calls relaxation and apply it to BPS equations
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
from numpy import pi
from Sigma_Critical import Sigma_Critical
from Relaxation_1D import Relaxation_1D
from BPS import BPS
from plot_BPS import plot_BPS

def solve_BPS(N,vac0_arg,vacf_arg,z0,zf,h,folder,tol,save_plot=True):
    num = int((zf-z0)/h)
    #if vacua0 is one of "x_k", then the sigma critical includes 2pi already
    #if it is one of "w_k", then the sigma cirtical does not include 2pi
    vac0, vac0_vec = _take_care_of_2pi(N,vac0_arg)
    vacf, vacf_vec = _take_care_of_2pi(N,vacf_arg)
    B=BPS(N,vac0_vec,vacf_vec)
    R = Relaxation_1D(B.ddx,z0,zf,vac0_vec,vacf_vec)
    R.solve(num,tol=tol,f0="special kink",diagnose=False)
    z = R.sol_z
    x = R.sol_f        
    plot_BPS(N,z,x,B,h,folder,vac0,vacf,save_plot)
    return z,x

def _take_care_of_2pi(N,vac_arg):
    vac = Sigma_Critical(N,vac_arg)
    if vac_arg[0] == "w":
        vac_vec = vac.imaginary_vector*2*pi
    else:
        vac_vec = vac.imaginary_vector
    return vac, vac_vec
    