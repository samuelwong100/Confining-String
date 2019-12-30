# -*- coding: utf-8 -*-
"""
File Name: Solver_Helpers.py
Purpose: Helper functions for solver
Author: Samuel Wong
"""
import pickle

def get_title(prefix,N,charge,bound,L,w,h,R,tol,max_loop,x0,method):
    title = "{}CS(N={}, charge={}, bound={}, L={}, w={}, h={},\
    R={}, tol={}, max_loop={}, x0={}, method={}/)".format(prefix,str(N),charge,
    bound,str(L),str(w),str(h),str(R),str(tol),str(max_loop),str(x0),method)
    return title

def store_solution(relax,title,N,charge_arg,bound_arg,L,w,h,R,tol,max_loop,x0,
                   method):
    #store the core result in a dictionary
    core_dict = {"N":N,"charge_arg":charge_arg,"bound_arg":bound_arg,"L":L,
                  "w":w,"h":h,"R":R,"tol":tol,"max_loop":max_loop,"x0":x0,
                  "method":method,"loop":relax.loop,"field":relax.x,
                  "error":relax.error}
    pickle_out = open(title+"core_dict.pickle","wb")
    pickle.dump(core_dict, pickle_out)
    pickle_out.close()
    #code for opening file
#    pickle_in = open(title+"core_dict.pickle","rb")
#    core_dict = pickle.load(pickle_in)
    