# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:01:10 2020

@author: samue
"""
from confinepy import Solution_Viewer, get_all_fully_solved_N_p_folder_names,\
    confining_string_solver

#code for getting a list of solutions and displaying one
#print(get_all_fully_solved_N_p_folder_names(7,"w3"))
#sol = Solution_Viewer("CS(N=7,charge=w3,bound=x1,L=30,w=30,h=0.1,R=25,sor=1.97,tol=1e-09,initial_kw=BPS,use_half_grid=True)")
#sol.display_all()

sol = confining_string_solver(N=2,charge_arg="w1",bound_arg="x1",L=30,w=25,R=21,
                        h=0.1,tol=1e-9,initial_kw="BPS",use_half_grid=True,
                            check_point_limit=200,diagnose=True)