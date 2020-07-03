# -*- coding: utf-8 -*-
"""
File Name: Relaxation_Solution.py
Purpose: Function that calls relaxation and apply it to BPS equations
Author: Samuel Wong
"""
import sys
sys.path.append("../")
from Main import Sigma_Critical
from Relaxation_1D import Relaxation_1D
from BPS import BPS

def solve_BPS(N,separation_R,top,vac0_arg,vacf_arg,z0,zf,h,folder,tol,save_plot=True):
    num = int((zf-z0)/h)
    vac0 = Sigma_Critical(N,vac0_arg)
    vacf = Sigma_Critical(N,vacf_arg)
    B=BPS(N,vac0.imaginary_vector,vacf.imaginary_vector)
    R = Relaxation_1D(B.ddx,z0,zf,vac0.imaginary_vector,vacf.imaginary_vector)
    R.solve(separation_R,top,num,tol=tol,f0="kink with predicted width",diagnose=False)
    z = R.sol_z
    x = R.sol_f        
    #plot_BPS(N,z,x,B,h,folder,vac0,vacf,save_plot)
    return z,x,B
