# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
from Source import solve_BPS

for N in range(9,13):
    x,z = solve_BPS(N,"x0","x1",401,tol=1e-8)
    for i in range(1,N):
        x,z = solve_BPS(N,"w"+str(i),"x1",401,tol=1e-8)
