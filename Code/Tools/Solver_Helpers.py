# -*- coding: utf-8 -*-
"""
File Name: Solver_Helpers.py
Purpose: Helper functions for solver
Author: Samuel Wong
"""

def get_title(N,charge,bound,L,w,h,R,tol,max_loop,x0,method):
    title = "CS(N={}, charge={}, bound={}, L={}, w={}, h={},\
    R={}, tol={}, max_loop={}, x0={}, method={})".format(str(N),charge,bound,
    str(L),str(w),str(h),str(R),str(tol),str(max_loop),str(x0),method)
    return title
