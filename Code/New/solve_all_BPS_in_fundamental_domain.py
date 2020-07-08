# -*- coding: utf-8 -*-
"""
File Name: .py
Purpose: 
Author: Samuel Wong
"""
from Source import solve_BPS

def get_ideal_num(N):
    if N<9:
        num = 401
    else:
        num = 601
    return num

for N in range(9,10):
    num = get_ideal_num(N)
    x,z = solve_BPS(N,"x0","x1",num)
    for i in range(1,N):
        x,z = solve_BPS(N,"w"+str(i),"x1",num)
