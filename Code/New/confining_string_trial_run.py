# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:56:46 2020

@author: samue
"""
from Source import confining_string_solver

N=2
charge_arg = "w1"
bound_arg = "x1"
L=30
w=15
R=10
h=0.1
tol=1e-5
use_half_grid=True
diagnose=True
initial_kw="BPS"
sol = confining_string_solver(N=N,charge_arg=charge_arg,bound_arg=bound_arg,
                              L=L,w=w,R=R,h=h,tol=tol,initial_kw=initial_kw,
                              use_half_grid=use_half_grid,diagnose=diagnose)


