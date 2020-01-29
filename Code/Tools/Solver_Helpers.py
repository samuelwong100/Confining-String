# -*- coding: utf-8 -*-
"""
File Name: Solver_Helpers.py
Purpose: Helper functions for solver
Author: Samuel Wong
"""
import os
import pickle

def get_title(prefix,N,charge,bound,L,w,h,R,tol,max_loop,x0,method):
    title =\
    ('{}CS(N={},charge={},bound={},L={},w={},h={},R={},tol={},'+ \
    'max_loop={},x0={},method={})/').format(prefix,str(N),charge,
    bound,str(L),str(w),str(h),str(R),str(tol),str(max_loop),str(x0),method)
    return title

def store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0,
                   method):
    #store the core result in a dictionary
    core_dict = {"N":N,"charge_arg":charge_arg,"bound_arg":bound_arg,"L":L,
                  "w":w,"h":h,"R":R,"tol":tol,"max_loop":max_loop,"x0":x0,
                  "method":method,"loop":relax.loop,"field":relax.x,
                  "error":relax.error}
    if x0 == "BPS":
        core_dict["BPS_top"] = relax.top_BPS
        core_dict["BPS_bottom"] = relax.bottom_BPS
        core_dict["BPS_y"] = relax.y_half
        core_dict["BPS_slice"] = relax.BPS_slice
        core_dict["initial_grid"] = relax.initial_grid
    #create directory for new folder if it doesn't exist
    if not os.path.exists(title):
        os.makedirs(title)
    #dump dictionary into pickle
    with open(title+"core_dict","wb") as file:
        pickle.dump(core_dict, file)
