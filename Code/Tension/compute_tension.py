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
from scipy.optimize import curve_fit

#a dictionary that given N, gives the optimal L,w,h,max_loop
N_Lwhm_dict = {3:[10,10,0.1,400]}

def compute_energy(N,charge_arg,L,w,h,max_loop):
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
    return R_array, energy_array

def plot_energy_vs_R(R_array,energy_array,m,b,N,p,L):
    plt.figure()
    plt.scatter(x=R_array,y=energy_array)
    x = np.linspace(1,L,1000)
    plt.plot(x,linear_model(x,m,b))
    plt.xlabel("R")
    plt.ylabel("Energy")
    plt.title("Energy vs Distance (N={}, p={})".format(str(N),str(p)))
    plt.savefig("Energy vs Distance (N={}, p={}).png".format(str(N),str(p)))
    plt.show()
    
def linear_model(x,m,b):
    return m*x + b

def compute_tension(N,p):
    #p is N-ality
    charge_arg='w'+str(p)
    L,w,h,max_loop = N_Lwhm_dict[N]
    R_array, energy_array = compute_energy(N,charge_arg,L,w,h,max_loop)
    potp, pcov = curve_fit(linear_model,xdata=R_array,ydata=energy_array)
    m,b = potp
    plot_energy_vs_R(R_array,energy_array,m,b,N,p,L)
    dm = np.sqrt(pcov[0][0])
    return m, dm
    
    

