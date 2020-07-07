# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
from Source import solve_BPS

for N in range(3,5):
    x,z = solve_BPS(N,"x0","x1",401,tol=1e-9)
    for i in range(1,N):
        x,z = solve_BPS(N,"w"+str(i),"x1",401,tol=1e-9)
        
if N = 10, num=601

