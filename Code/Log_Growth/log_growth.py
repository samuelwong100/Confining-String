# -*- coding: utf-8 -*-
"""
File Name: Log_Growth.py
Purpose: check whether the tranverse size grows logarithmically with respect to
         R
Author: Samuel Wong
"""
import sys
sys.path.append("../Solver/")
sys.path.append("../Results/Solutions/")
import numpy as np
import matplotlib.pyplot as plt
from Solution_Viewer import Solution_Viewer

sol_list = []
d_list = []
R_list = []
str_list = ["../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=30,w=30,h=0.1,R=5,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=30,w=30,h=0.1,R=10,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=30,w=30,h=0.1,R=15,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=30,w=30,h=0.1,R=20,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=30,w=30,h=0.1,R=25,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=50,w=50,h=0.1,R=40,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=100,w=50,h=0.1,R=90,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=150,w=50,h=0.1,R=140,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=300,w=50,h=0.1,R=290,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=600,w=40,h=0.1,R=590,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=1000,w=30,h=0.1,R=980,max_loop=400,x0=BPS)/",
            "../Results/Solutions/CS(N=2,charge=w1,bound=x1,L=1500,w=30,h=0.1,R=1450,max_loop=400,x0=BPS)/"]

for title in str_list:
    sol = Solution_Viewer(title)
    sol.plot_potential_energy_density()
    sol_list.append(sol)
    pe = sol.get_potential_energy_density()
    center = sol.grid.y_axis
    pe_top = pe[0:sol.grid.z_axis,center]
    pe_bottom = pe[sol.grid.z_axis:,center]
    top_arg = np.argmax(pe_top)
    bottom_arg = np.argmax(pe_bottom) + pe_top.size
    d = sol.grid.y[bottom_arg]-sol.grid.y[top_arg]
    R_list.append(sol.R)
    d_list.append(d)
R = np.array(R_list)
d = np.array(d_list)

plt.scatter(R,d)
plt.ylim(0)

