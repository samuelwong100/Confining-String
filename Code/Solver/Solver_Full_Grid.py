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
from Relaxation import Relaxation, Continue_Relaxation, Relaxation_half_grid
from Solver_Helpers import get_title, store_solution
from Solution_Viewer import Solution_Viewer

def Solver(N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0="BPS",
           half_grid=True,diagnose=False):
    prefix = "../Results/Solutions/"
    title = get_title(prefix,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0)
    if os.path.exists(title):
        sol = Solution_Viewer(title)
    else:
        charge = Sigma_Critical(N,charge_arg)
        bound = Sigma_Critical(N,bound_arg)
        grid = Standard_Dipole_Grid(L,w,h,R)
        if half_grid:
            relax = Relaxation_half_grid(grid,N,bound,charge,tol,max_loop,x0,
                                         diagnose)
        else:
            relax = Relaxation(grid,N,bound,charge,tol,max_loop,x0,diagnose)
        relax.solve()
        store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,
                       x0)
        sol = Solution_Viewer(title)
    #sol.display_all()
    return sol

def Solver_Full_Grid(N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0="BPS",
                     diagnose=False):
    prefix = "../Results/Solutions/"
    title = get_title(prefix,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0)
    if os.path.exists(title):
        sol = Solution_Viewer(title)
    else:
        charge = Sigma_Critical(N,charge_arg)
        bound = Sigma_Critical(N,bound_arg)
        grid = Standard_Dipole_Grid(L,w,h,R)
        relax = Relaxation(grid,N,bound,charge,tol,max_loop,x0,diagnose)
        relax.solve()
        store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,
                       x0)
        sol = Solution_Viewer(title)
    #sol.display_all()
    return sol

def Continue_Solver_Full_Grid(old_title,max_loop,diagnose=True):
    #note the input old_title does not contain path, just folder name
    new_title = _get_new_from_old_title(old_title,max_loop)
    if os.path.exists(new_title):
        sol = Solution_Viewer(new_title)
    else:
        #get old title into a format used by solution viewer
        old_title = "../Results/Solutions/" + old_title + "/"
        old_sol = Solution_Viewer(old_title)
        #get the parameters from the old solution
        N = old_sol.N
        tol = old_sol.tol
        x0 = old_sol.x0
        charge_arg = old_sol.charge_arg
        bound_arg = old_sol.bound_arg
        charge = Sigma_Critical(N,charge_arg)
        bound = Sigma_Critical(N,bound_arg)
        L = old_sol.L
        w = old_sol.w
        h = old_sol.h
        R = old_sol.R
        #create a continuing relaxation object, which starts with old field
        relax = Continue_Relaxation(old_sol,max_loop,charge,bound,diagnose)
        relax.solve()
        store_solution(relax,new_title,N,charge_arg,bound_arg,L,w,h,R,tol,
                       max_loop,x0)
        sol = Solution_Viewer(new_title)
    #sol.display_all()
    return sol

def _get_new_from_old_title(old_title,max_loop):
    #the index of the letter 'm' in 'max_loop='
    index_m = old_title.find("max_loop=")
    index_equal_sign = index_m + 8
    #get the index of the comma immediately after the max loop number
    index_comma = old_title.find(",",index_m) #start search at index_m
    #get the parts before and after the loop number
    first_part = old_title[0:index_equal_sign+1]
    last_part = old_title[index_comma:len(old_title)]
    #insert the new max_loop
    new_title = first_part+str(max_loop)+last_part
    #get new title into proper format for storage
    new_title = "../Results/Solutions/" + new_title + "/"
    return new_title
    