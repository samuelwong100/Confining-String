# -*- coding: utf-8 -*-
"""
File Name: Solver_Full_Grid.py
Purpose: Solves the equation of motion for full grid method
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import numpy as np
import matplotlib.pyplot as plt
from Grid import Standard_Dipole_Grid
from Charge import Charge
from Relaxation import Relaxation
from Math import Superpotential

def Solver_Full_Grid(N,charge_arg,L,w,h,R,tol,max_loop,x0=None,diagnose=False):
    charge = Charge(N,charge_arg)
    grid = Standard_Dipole_Grid(L,w,h,R)
    W = Superpotential(N)
    bound = W.x_min[0,:]
    relax = Relaxation(grid,N,bound,charge.vector,tol,max_loop,x0,diagnose)
    relax.solve()
    x = relax.x
    return x

if __name__ == "__main__":
    x = Solver_Full_Grid(2,"w1",5,5,0.05,2,1E-30,500,diagnose = True)