# -*- coding: utf-8 -*-
"""
File Name: compute_tension.py
Purpose: compute the tension of a confining string by simulating a range of R
Author: Samuel Wong
"""
import sys
sys.path.append("../Solver")
from Solver import Solver
import numpy as np
import matplotlib.pyplot as plt

#a dictionary that given N, gives the optimal L,w,h,max_loop
N_Lwhm_dict = {3:[10,10,0.1,400]}

def compute_tension(N,p):
    #p is N-ality
    charge_arg='w'+str(p)
    L,w,h,max_loop = N_Lwhm_dict[N]
    #initialze lists
    R_list = []
    energy_list = []
    for R in range(1,L):
        sol = Solver(N=N,charge_arg=charge_arg,bound_arg="x1",
                     L=L,w=w,h=h,R=R,max_loop=max_loop,
                     x0="BPS",half_grid=True)
        #sol.display_all()
        R_list.append(R)
        energy_list.append(sol.get_energy())
    #convert back to array
    R_array = np.array(R_list)
    energy_array = np.array(energy_list)
    #plot energy vs R
    plt.figure()
    plt.scatter(x=R_array,y=energy_array)
    plt.xlabel("R")
    plt.ylabel("Energy")
    plt.title("Energy vs Distance (N={}, p={})".format(str(N),str(p)))
    plt.savefig("Energy vs Distance (N={}, p={}).png".format(str(N),str(p)))
    plt.show()

