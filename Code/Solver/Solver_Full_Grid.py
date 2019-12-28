# -*- coding: utf-8 -*-
"""
File Name: Solver_Full_Grid.py
Purpose: Solve the equation of motion for full grid method
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import numpy as np
import matplotlib.pyplot as plt
from Grid import Standard_Dipole_Grid
from Sigma_Critical import Sigma_Critical
from Relaxation import Relaxation

def Solver_Full_Grid(N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0=None,
                     diagnose=False):
    charge = Sigma_Critical(N,charge_arg)
    bound = Sigma_Critical(N,bound_arg)
    grid = Standard_Dipole_Grid(L,w,h,R)
    relax = Relaxation(grid,N,bound.imaginary_vector,charge.real_vector,tol,
                       max_loop,x0,diagnose)
    relax.solve()
    x = relax.x
    return x

if __name__ == "__main__":
    x = Solver_Full_Grid(2,"w1",5,5,0.05,1,1E-30,500,diagnose=True)