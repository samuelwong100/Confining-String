# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:56:46 2020

@author: samue
"""
from Source import confining_string_solver

N=7
bound_arg = "x1"
L=30
w=30
h=0.1
sor=1.97
tol=1e-9
use_half_grid=True
diagnose=True
initial_kw="BPS"


for R in range(5,26,5):
    sol = confining_string_solver(N=N,charge_arg="w3",bound_arg=bound_arg,
                                  L=L,w=w,R=R,h=h,sor=sor,tol=tol,
                                  initial_kw=initial_kw,
                                  use_half_grid=use_half_grid,diagnose=diagnose)
    

