# -*- coding: utf-8 -*-
"""
File Name: Solver_Full_Grid.py
Purpose: Solve the equation of motion for full grid method
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import os
from Grid import Standard_Dipole_Grid
from Sigma_Critical import Sigma_Critical
from Relaxation import Relaxation
from Solver_Helpers import get_title, store_solution
from Solution_Viewer import Solution_Viewer

def Solver_Full_Grid(N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0=None,
                     diagnose=False):
    prefix = "../Results/Solutions/"
    title = get_title(prefix,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0,
                      "full grid")
    if os.path.exists(title):
        sol = Solution_Viewer(title)
    else:
        charge = Sigma_Critical(N,charge_arg)
        bound = Sigma_Critical(N,bound_arg)
        grid = Standard_Dipole_Grid(L,w,h,R)
        relax = Relaxation(grid,N,bound.imaginary_vector,charge.real_vector,
                           tol,max_loop,x0,diagnose)
        relax.solve()
        store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,
                       x0,"full grid")
        sol = Solution_Viewer(title)
    sol.display_all()
    return sol
